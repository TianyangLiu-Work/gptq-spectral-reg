#!/usr/bin/env python3
"""Save a quantized OPT-6.7B model for later downstream evaluation.

Usage:
  python save_quantized_models.py \
      --label GPTQ_base --method none --beta 0.01 \
      --group_size 128
  python save_quantized_models.py --label FP16

Each run loads the model, builds Hessians, quantizes, and saves to
  /mnt/host-share/gptq-models/opt-6.7b-g{group_size}-{label}/
"""

import sys, time, json, math, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gptq_spec.power_iter import top_singular_vector as power_iteration
from gptq_spec.quant_utils import quantize_weight_groupwise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

device = torch.device("cuda:0")
MODEL_NAME = "facebook/opt-6.7b"
BITS = 4
CALIB_SAMPLES = 128
BATCH_SIZE = 8
SEQ_LEN = 2048
BASE_DIR = Path("/mnt/host-share/gptq-models")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="e.g. GPTQ_base, Damp_0.05, Frob_0.001, Spectral_0.01, FP16")
    p.add_argument("--method", default=None, help="none, stronger_damping, frobenius, spectral")
    p.add_argument("--beta", type=float, default=None, help="regularization beta")
    p.add_argument("--group_size", type=int, default=128, help="128 or 32")
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label
    method = args.method
    beta = args.beta
    group_size = args.group_size
    is_fp16 = (label == "FP16")

    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === {label} (g{group_size}) ===", flush=True)
    print(f"  Config: method={method} beta={beta}", flush=True)

    # ── Load ──
    print(f"[{time.strftime('%H:%M:%S')}] Loading model and tokenizer...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # ── Data ──
    print(f"[{time.strftime('%H:%M:%S')}] Preparing data...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()]
    needed = CALIB_SAMPLES * SEQ_LEN * 2 + 1000
    enc = tokenizer(" ".join(texts), truncation=True, max_length=needed + 1000, return_tensors="pt")
    ids = enc.input_ids[0]
    seqs = [ids[i:i + SEQ_LEN] for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN)]
    total = max(CALIB_SAMPLES // BATCH_SIZE, 1) * BATCH_SIZE
    seqs = seqs[:total]
    batches = []
    for i in range(0, len(seqs), BATCH_SIZE):
        inp = torch.stack(seqs[i:i + BATCH_SIZE])
        batches.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
    calib_batches = batches[:max(CALIB_SAMPLES // BATCH_SIZE, 1)]

    # ── Linear layers ──
    print(f"[{time.strftime('%H:%M:%S')}] Finding linear layers...", flush=True)
    handles = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    quant_handles = [(n, m) for n, m in handles
                     if not any(x in n for x in ["embed", "lm_head", "final_layer_norm"])
                     and (".self_attn." in n or ".fc" in n)
                     and m.weight.shape[1] <= 4096]
    print(f"  {len(quant_handles)} quantizable layers", flush=True)

    # ── Build Hessians (online accumulation) ──
    print(f"[{time.strftime('%H:%M:%S')}] Collecting Hessians ({CALIB_SAMPLES} samples)...", flush=True)
    hessians = {n: None for n, _ in quant_handles}
    hooks = []
    for name, mod in quant_handles:
        def mk(n):
            def fn(m, inp, _):
                x = inp[0].detach().float()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                H = x.T @ x
                if hessians[n] is None:
                    hessians[n] = H.cpu()
                else:
                    hessians[n] = hessians[n] + H.cpu()
            return fn
        hooks.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        for bi, b in enumerate(tqdm(calib_batches, desc="Hessian", unit="batch")):
            model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
    for h in hooks:
        h.remove()
    hessians = {n: H for n, H in hessians.items() if H is not None}
    print(f"  {len(hessians)} Hessians collected", flush=True)

    # ── Reference weights ──
    print(f"[{time.strftime('%H:%M:%S')}] Capturing reference weights...", flush=True)
    refs = {n: m.weight.data.detach().clone().cpu() for n, m in quant_handles if n in hessians}
    quant_names = list(hessians.keys())

    def restore():
        for n, m in quant_handles:
            if n in refs:
                m.weight.data.copy_(refs[n].to(device=device, dtype=torch.float16))

    # ── Per-channel GPTQ ──
    def correct_gptq(weight, hessian, bits=4, act_order=True):
        out_dim, in_dim = weight.shape
        w = weight.detach().float().clone()
        H = hessian.float().to(w.device).clone()
        qmax = float(2 ** (bits - 1) - 1)
        md = torch.diag(H).mean()
        H = H + 0.01 * md * torch.eye(in_dim, device=H.device, dtype=H.dtype)
        try:
            L = torch.linalg.cholesky(H)
        except:
            H = H + 0.01 * md * torch.eye(in_dim, device=H.device, dtype=H.dtype)
            L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        if act_order:
            perm = torch.argsort(torch.diag(H).abs(), descending=True)
            w = w[:, perm]
            H_inv = H_inv[perm][:, perm]
        scales = w.abs().max(dim=0, keepdim=True).values / qmax
        scales = scales.clamp(min=1e-12)
        for i1 in range(0, in_dim, 128):
            i2 = min(i1 + 128, in_dim)
            for j in range(i1, i2):
                col = w[:, j:j + 1]
                s = scales[:, j:j + 1]
                q = torch.round(col / s).clamp(-qmax, qmax)
                dq = q * s
                err = dq - col
                w[:, j:j + 1] = dq
                if j + 1 < in_dim:
                    h_diag = H_inv[j, j].clamp(min=1e-12)
                    w[:, j + 1:] += (err / h_diag) @ H_inv[j:j + 1, j + 1:]
        if act_order:
            w = w[:, torch.argsort(perm)]
        return w.to(dtype=weight.dtype)

    # ── Spectral estimation (only if needed) ──
    need_spectral = method == "spectral"
    spec_v = {}
    if need_spectral:
        print(f"[{time.strftime('%H:%M:%S')}] Spectral estimation...", flush=True)
        for ni, n in enumerate(tqdm(quant_names, desc="Spectral", unit="layer")):
            H_reg = hessians[n].float().to("cuda")
            md = torch.diag(H_reg).mean()
            H_reg = H_reg + 0.01 * md * torch.eye(H_reg.shape[0], device="cuda", dtype=torch.float32)
            w = refs[n].float().to("cuda")
            q_w = correct_gptq(w, H_reg, bits=BITS)
            dw = (q_w - refs[n].float().to("cuda")).float()
            v, _ = power_iteration(dw, n_iter=20)
            spec_v[n] = v.cpu()
            del H_reg, w, q_w, dw, v
        print(f"  Estimated {len(spec_v)} spectral vectors", flush=True)

    # ── Quantize ──
    if not is_fp16:
        print(f"[{time.strftime('%H:%M:%S')}] Quantizing ({label})...", flush=True)
        deltas = {}
        for ni, n in enumerate(tqdm(quant_names, desc="Quant", unit="layer")):
            H = hessians[n].float().to("cuda")
            d = H.shape[0]
            md = torch.diag(H).mean()

            if method == "spectral":
                v = spec_v.get(n)
                if v is None:
                    continue
                v_gpu = v.to("cuda")
                H = H + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)
                H = H + beta * md * torch.outer(v_gpu, v_gpu)
                del v_gpu
            elif method == "stronger_damping":
                H = H + beta * md * torch.eye(d, device="cuda", dtype=torch.float32)
            elif method == "frobenius":
                H = H + (0.01 + beta) * md * torch.eye(d, device="cuda", dtype=torch.float32)
            else:
                H = H + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)

            w = refs[n].float().to("cuda")
            q_w = correct_gptq(w, H, bits=BITS)

            if group_size < refs[n].shape[1]:
                q_w, _, _ = quantize_weight_groupwise(q_w, bits=BITS, group_size=group_size)
                q_w = q_w.to(device="cuda", dtype=torch.float32)

            deltas[n] = q_w.cpu() - refs[n]
            del H, w, q_w

        print(f"  Applying weights...", flush=True)
        for n, m in tqdm(quant_handles, desc="Apply", unit="layer"):
            if n in deltas and n in refs:
                m.weight.data.copy_(
                    refs[n].to(device=device, dtype=torch.float16) +
                    deltas[n].to(device=device, dtype=torch.float16))
        del deltas
    else:
        print(f"[{time.strftime('%H:%M:%S')}] FP16 mode — no quantization needed", flush=True)

    # ── Save ──
    save_name = f"opt-6.7b-g{group_size}-{label}"
    save_path = BASE_DIR / save_name
    print(f"[{time.strftime('%H:%M:%S')}] Saving to {save_path}...", flush=True)
    model.save_pretrained(str(save_path), safe_serialization=True)
    tokenizer.save_pretrained(str(save_path))

    info = {
        "model_name": MODEL_NAME,
        "method": label,
        "method_raw": method,
        "beta": beta,
        "bits": BITS,
        "group_size": group_size,
        "quant_time_sec": round(time.time() - t_start, 1),
    }
    with open(save_path / "quantization_info.json", "w") as f:
        json.dump(info, f, indent=2)

    elapsed = time.time() - t_start
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {elapsed:.1f}s", flush=True)
    print(f"  Saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
