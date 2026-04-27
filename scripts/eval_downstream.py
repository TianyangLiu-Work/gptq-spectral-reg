#!/usr/bin/env python3
"""Eval downstream benchmarks on all GPTQ methods.
Reuses the quantization logic from run_phase4.py
then evaluates on 4 zero-shot tasks using lm_eval."""

import sys, time, json, math, torch, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gptq_spec.power_iter import top_singular_vector as power_iteration
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

device = torch.device("cuda:0")

MODEL_NAME = "facebook/opt-6.7b"
BITS, GROUP_SIZE = 4, 128
CALIB_SAMPLES, EVAL_SAMPLES = 128, 64
BATCH_SIZE, SEQ_LEN = 8, 2048

# Tasks for zero-shot evaluation
EVAL_TASKS = ["lambada_openai", "hellaswag", "arc_challenge", "winogrande"]
NUM_FEWSHOT = {"lambada_openai": 0, "hellaswag": 0, "arc_challenge": 0, "winogrande": 0}

METHODS = [
    ("FP16_cached", None, None),  # special: just load FP16
    ("GPTQ_base", "none", 0.01),
    ("Damp_0.05", "stronger_damping", 0.05),
    ("Frob_0.001", "frobenius", 0.001),
    ("Spectral_0.01", "spectral", 0.01),
]

def main():
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading model and data...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # Data preparation (same as run_phase4.py)
    print(f"[{time.strftime('%H:%M:%S')}] Preparing data...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()]
    needed = (CALIB_SAMPLES + EVAL_SAMPLES) // BATCH_SIZE * BATCH_SIZE * SEQ_LEN + 1000
    enc = tokenizer(" ".join(texts), truncation=True, max_length=needed+1000, return_tensors="pt")
    ids = enc.input_ids[0]
    seqs = [ids[i:i+SEQ_LEN] for i in range(0, len(ids)-SEQ_LEN, SEQ_LEN)]
    total = (CALIB_SAMPLES + EVAL_SAMPLES) // BATCH_SIZE * BATCH_SIZE
    seqs = seqs[:total]
    batches = []
    for i in range(0, len(seqs), BATCH_SIZE):
        inp = torch.stack(seqs[i:i+BATCH_SIZE])
        batches.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
    calib_batches = batches[:CALIB_SAMPLES // BATCH_SIZE]
    eval_batches = batches[CALIB_SAMPLES // BATCH_SIZE:]

    # Linear layers
    print(f"[{time.strftime('%H:%M:%S')}] Finding linear layers...", flush=True)
    handles = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    quant_handles = [(n, m) for n, m in handles
                     if not any(x in n for x in ["embed", "lm_head", "final_layer_norm"])
                     and (".self_attn." in n or ".fc" in n)
                     and m.weight.shape[1] <= 4096]
    print(f"  {len(quant_handles)} quantizable layers", flush=True)

    # ── Build Hessians ──
    print(f"[{time.strftime('%H:%M:%S')}] Collecting Hessians ({CALIB_SAMPLES} calib samples)...", flush=True)
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
    for h in hooks: h.remove()
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

    # ── Spectral estimation ──
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

    # ── Evaluate each method ──
    results = []

    for label, method, beta in METHODS:
        print(f"\n{'='*70}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {'='*50}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}]  {label}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {'='*50}", flush=True)

        t0 = time.time()

        if label == "FP16_cached":
            # Just restore FP16 weights, no quantization
            restore()
        else:
            # Quantize all layers
            deltas = {}
            for ni, n in enumerate(quant_names):
                if (ni + 1) % 20 == 0:
                    print(f"  quant {ni+1}/{len(quant_names)}", flush=True)
                H = hessians[n].float().to("cuda")
                d = H.shape[0]
                md = torch.diag(H).mean()

                if method == "spectral":
                    v = spec_v.get(n)
                    if v is not None:
                        v_gpu = v.to("cuda")
                        H = H + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)
                        H = H + beta * md * torch.outer(v_gpu, v_gpu)
                        del v_gpu
                    else:
                        continue
                elif method == "stronger_damping":
                    H = H + beta * md * torch.eye(d, device="cuda", dtype=torch.float32)
                elif method == "frobenius":
                    H = H + (0.01 + beta) * md * torch.eye(d, device="cuda", dtype=torch.float32)
                else:  # none = GPTQ_base
                    H = H + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)

                w = refs[n].float().to("cuda")
                q_w = correct_gptq(w, H, bits=BITS)
                deltas[n] = q_w.cpu() - refs[n]
                del H, w, q_w

            # Apply quantized weights
            for n, m in quant_handles:
                if n in deltas and n in refs:
                    m.weight.data.copy_(
                        refs[n].to(device=device, dtype=torch.float16) +
                        deltas[n].to(device=device, dtype=torch.float16))
            t_q = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}]  Quant done in {t_q:.1f}s", flush=True)
            del deltas

        # ── Run lm_eval benchmarks ──
        print(f"[{time.strftime('%H:%M:%S')}]  Running lm_eval tasks...", flush=True)

        lm_wrapper = HFLM(pretrained=model, batch_size="auto:4")

        for task in EVAL_TASKS:
            t_task = time.time()
            try:
                result = simple_evaluate(
                    model=lm_wrapper,
                    tasks=[task],
                    num_fewshot=NUM_FEWSHOT[task],
                    limit=200,
                    log_samples=False,
                )
                acc = result["results"][task].get("acc,none", result["results"][task].get("acc", "N/A"))
                acc_stderr = result["results"][task].get("acc_stderr,none", "N/A")
                t_elapsed = time.time() - t_task
                print(f"[{time.strftime('%H:%M:%S')}]    {task:20s}: acc={acc:.4f} ± {acc_stderr:.4f}  ({t_elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}]    {task:20s}: ERROR {e}", flush=True)

        # Restore for next method
        restore()
        torch.cuda.empty_cache()

        total_time = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}]  {label} total: {total_time:.1f}s", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {'='*70}", flush=True)

    total_elapsed = time.time() - t_start
    print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE in {total_elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()
