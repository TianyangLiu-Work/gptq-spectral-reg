#!/usr/bin/env python3
"""Save quantized OPT-6.7B models for later downstream evaluation.

Quantizes with 4 methods (GPTQ_base, Damp_0.05, Frob_0.001, Spectral_0.01)
at 2 group sizes (128, 32), then saves each as a complete model directory
loadable via AutoModelForCausalLM.from_pretrained().

IMPORTANT: Behaves exactly like run_phase4.py for g128 (per-channel GPTQ only,
group_size is just blocksize) and like run_phase5.py for g32 (per-channel GPTQ
then requantize_weight_groupwise).

Output:
  models/
    opt-6.7b-g128-FP16/
    opt-6.7b-g128-GPTQ_base/
    opt-6.7b-g128-Damp_0.05/
    opt-6.7b-g128-Frob_0.001/
    opt-6.7b-g128-Spectral_0.01/
    opt-6.7b-g32-GPTQ_base/
    opt-6.7b-g32-Damp_0.05/
    opt-6.7b-g32-Frob_0.001/
    opt-6.7b-g32-Spectral_0.01/
"""

import sys, time, json, math, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gptq_spec.power_iter import top_singular_vector as power_iteration
from gptq_spec.quant_utils import quantize_weight_groupwise

device = torch.device("cuda:0")
MODEL_NAME = "facebook/opt-6.7b"
BITS = 4
GROUP_SIZES = [128, 32]
CALIB_SAMPLES = 128
BATCH_SIZE = 8
SEQ_LEN = 2048

METHODS = [
    ("FP16",           None,    None),           # special: just save original
    ("GPTQ_base",      "none",          0.01),
    ("Damp_0.05",      "stronger_damping", 0.05),
    ("Frob_0.001",     "frobenius",     0.001),
    ("Spectral_0.01",  "spectral",      0.01),
]

BASE_DIR = Path("/mnt/host-share/gptq-models")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
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
    needed = CALIB_SAMPLES * SEQ_LEN * 2 + 1000  # enough for all
    enc = tokenizer(" ".join(texts), truncation=True, max_length=needed+1000, return_tensors="pt")
    ids = enc.input_ids[0]
    seqs = [ids[i:i+SEQ_LEN] for i in range(0, len(ids)-SEQ_LEN, SEQ_LEN)]
    total = (CALIB_SAMPLES // BATCH_SIZE) * BATCH_SIZE
    seqs = seqs[:max(total, 1)]
    batches = []
    for i in range(0, len(seqs), BATCH_SIZE):
        inp = torch.stack(seqs[i:i+BATCH_SIZE])
        batches.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
    calib_batches = batches[:max(CALIB_SAMPLES // BATCH_SIZE, 1)]

    # ── Linear layer discovery ──
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
        for bi, b in enumerate(calib_batches):
            print(f"  batch {bi+1}/{len(calib_batches)}", flush=True)
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
                col = w[:, j:j+1]
                s = scales[:, j:j+1]
                q = torch.round(col / s).clamp(-qmax, qmax)
                dq = q * s
                err = dq - col
                w[:, j:j+1] = dq
                if j + 1 < in_dim:
                    h_diag = H_inv[j, j].clamp(min=1e-12)
                    w[:, j+1:] += (err / h_diag) @ H_inv[j:j+1, j+1:]
        if act_order:
            w = w[:, torch.argsort(perm)]
        return w.to(dtype=weight.dtype)

    # ── Quantize a single layer ──
    def quantize_layer(n, H, method, beta, group_size):
        """Per-channel GPTQ, then optionally requantize to group_size."""
        Hc = H.float().to("cuda")
        d = Hc.shape[0]
        md = torch.diag(Hc).mean()

        if method == "spectral":
            v = spec_v.get(n)
            if v is None:
                return None
            v_gpu = v.to("cuda")
            Hc = Hc + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)
            Hc = Hc + beta * md * torch.outer(v_gpu, v_gpu)
            del v_gpu
        elif method == "stronger_damping":
            Hc = Hc + beta * md * torch.eye(d, device="cuda", dtype=torch.float32)
        elif method == "frobenius":
            Hc = Hc + (0.01 + beta) * md * torch.eye(d, device="cuda", dtype=torch.float32)
        else:  # none = GPTQ_base
            Hc = Hc + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)

        w = refs[n].float().to("cuda")
        q_w = correct_gptq(w, Hc, bits=BITS)  # per-channel GPTQ

        if group_size < refs[n].shape[1]:
            # Requantize to target group_size (like Phase 5)
            q_w, _, _ = quantize_weight_groupwise(q_w, bits=BITS, group_size=group_size)
            q_w = q_w.to(device="cuda", dtype=torch.float32)

        delta = q_w.cpu() - refs[n]
        del Hc, w, q_w
        return delta

    # ── Spectral estimation (shared across group_sizes) ──
    print(f"[{time.strftime('%H:%M:%S')}] Spectral estimation...", flush=True)
    spec_v = {}
    for ni, n in enumerate(quant_names):
        if (ni + 1) % 20 == 0:
            print(f"  spectral {ni+1}/{len(quant_names)}", flush=True)
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

    # ══════════════════════════════════════════════
    #  Save loop: group_size × methods
    # ══════════════════════════════════════════════

    for group_size in GROUP_SIZES:
        print(f"\n{'='*70}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] Group size = {group_size}", flush=True)
        print(f"{'='*70}", flush=True)

        for label, method, beta in METHODS:
            print(f"\n  --- {label} (g{group_size}) ---", flush=True)
            t0 = time.time()
            restore()
            torch.cuda.empty_cache()

            if label != "FP16":
                # Quantize all layers
                deltas = {}
                for ni, n in enumerate(quant_names):
                    if (ni + 1) % 20 == 0:
                        print(f"  quant {ni+1}/{len(quant_names)}", flush=True)
                    delta = quantize_layer(n, hessians[n], method, beta, group_size)
                    if delta is not None:
                        deltas[n] = delta

                # Apply quantized weights
                for n, m in quant_handles:
                    if n in deltas and n in refs:
                        m.weight.data.copy_(
                            refs[n].to(device=device, dtype=torch.float16) +
                            deltas[n].to(device=device, dtype=torch.float16))
                del deltas

            # ── Save ──
            save_name = f"opt-6.7b-g{group_size}-{label}"
            save_path = BASE_DIR / save_name
            print(f"  Saving to {save_path}...", flush=True)
            model.save_pretrained(str(save_path), safe_serialization=True)
            tokenizer.save_pretrained(str(save_path))

            info = {
                "model_name": MODEL_NAME,
                "method": label if label != "FP16" else method,
                "method_raw": method,
                "beta": beta if label != "FP16" else None,
                "bits": BITS,
                "group_size": group_size,
                "quant_time_sec": round(time.time() - t0, 1),
            }
            with open(save_path / "quantization_info.json", "w") as f:
                json.dump(info, f, indent=2)

            print(f"  Saved in {time.time()-t0:.1f}s", flush=True)
            restore()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE in {elapsed:.1f}s", flush=True)
    print(f"Models saved under {BASE_DIR.resolve()}/", flush=True)


if __name__ == "__main__":
    main()
