#!/usr/bin/env python3
"""Phase 4: OPT-6.7B GPTQ spectral regularization.
Memory-optimized: instant Hessian building, no activation storage."""
import sys, time, json, math, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gptq_spec.power_iter import top_singular_vector as power_iteration

device = torch.device("cuda:0")

MODEL_NAME = "facebook/opt-6.7b"
BITS, GROUP_SIZE = 4, 128
CALIB_SAMPLES, EVAL_SAMPLES = 128, 64
BATCH_SIZE, SEQ_LEN = 8, 2048

METHODS = [
    ("GPTQ_base", "none", 0.01),
    ("Damp_0.05", "stronger_damping", 0.05),
    ("Frob_0.001", "frobenius", 0.001),
    ("Spectral_0.01", "spectral", 0.01),
]

def main():
    print(f"Loading {MODEL_NAME}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # Data
    print("Preparing data...", flush=True)
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
    print("Finding linear layers...", flush=True)
    handles = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    quant_handles = [(n, m) for n, m in handles
                     if not any(x in n for x in ["embed", "lm_head", "final_layer_norm"])
                     and (".self_attn." in n or ".fc" in n)
                     and m.weight.shape[1] <= 4096]
    print(f"  {len(quant_handles)} quantizable layers", flush=True)

    # ── Build Hessians directly (online accumulation) ──
    print(f"Collecting Hessians ({CALIB_SAMPLES} calib samples)...", flush=True)
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

    # ── Small test activations for metrics ──
    print("Collecting test activations...", flush=True)
    test_acts = {n: [] for n in hessians}
    hooks = []
    for name, mod in quant_handles:
        if name not in hessians: continue
        def mk(n):
            def fn(m, inp, _):
                x = inp[0].detach().cpu().float()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                test_acts[n].append(x)
            return fn
        hooks.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        for b in eval_batches[:2]:
            model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
    for h in hooks: h.remove()
    test_acts = {n: torch.cat(v, dim=0) if v else torch.tensor([]) for n, v in test_acts.items()}
    test_acts = {n: v for n, v in test_acts.items() if v.numel() > 0}

    # ── Reference weights ──
    print("Capturing reference weights...", flush=True)
    refs = {n: m.weight.data.detach().clone().cpu() for n, m in quant_handles if n in hessians}
    quant_names = list(hessians.keys())

    def restore():
        for n, m in quant_handles:
            if n in refs:
                m.weight.data.copy_(refs[n].to(device=device, dtype=torch.float16))

    # ── PPL ──
    @torch.no_grad()
    def evaluate_ppl():
        total_ce, total_tokens = 0.0, 0
        for b in eval_batches:
            inp = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            out = model(inp, attention_mask=mask)
            logits = out.logits[:, :-1].contiguous()
            labels = inp[:, 1:].contiguous()
            mask2 = mask[:, 1:].contiguous()
            vs = logits.size(-1)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vs), labels.masked_fill(mask2 == 0, -100).view(-1), reduction="sum")
            total_ce += loss.item()
            total_tokens += mask2.sum().item()
        return math.exp(total_ce / total_tokens) if total_tokens > 0 else float("inf")

    # ── Metrics ──
    def compute_metrics(deltas):
        et, specs = [], []
        for n, dw in deltas.items():
            dw_gpu = dw.float().to("cuda")
            w = refs[n].float()
            xt = test_acts.get(n)
            if xt is not None and xt.numel() > 0:
                diff = (xt.float().to("cuda") @ dw_gpu.T)
                inp_norm = xt.float().norm().item()
                et.append(diff.norm().item() / max(inp_norm, 1e-12))
            specs.append((w + dw.cpu()).norm().item() / max(w.norm().item(), 1e-12))
            del dw_gpu
        return (0, float(torch.tensor(et).mean()) if et else 0,
                float(torch.tensor(specs).mean()) if specs else 0)

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

    # ── FP16 ──
    print(f"\n{'='*70}", flush=True)
    print("  FP16 baseline", flush=True)
    print(f"{'='*70}", flush=True)
    restore()
    fp16_ppl = evaluate_ppl()
    print(f"  FP16 PPL={fp16_ppl:.4f}", flush=True)

    # ── Spectral estimation ──
    print("Spectral estimation...", flush=True)
    spec_v = {}
    for n in quant_names:
        H = hessians[n].to("cuda")
        H_reg = hessians[n].float().to("cuda")
        md = torch.diag(H_reg).mean()
        H_reg = H_reg + 0.01 * md * torch.eye(H_reg.shape[0], device="cuda", dtype=torch.float32)
        w = refs[n].float().to("cuda")
        q_w = correct_gptq(w, H_reg, bits=BITS)
        dw = (q_w - refs[n].float().to("cuda")).float()
        v, _ = power_iteration(dw, n_iter=20)
        spec_v[n] = v.cpu()
        del H, H_reg, w, q_w, dw, v
    print(f"  Estimated {len(spec_v)} spectral vectors", flush=True)

    # ── Methods loop ──
    results = [{"method": "FP16", "PPL": fp16_ppl, "E_calib": 0, "E_test": 0, "Spec": 0, "Time": 0}]
    for label, method, beta in METHODS:
        print(f"\n{'='*70}", flush=True)
        print(f"  {label} (beta={beta})", flush=True)
        print(f"{'='*70}", flush=True)
        restore()
        t0 = time.time()

        deltas = {}
        for n in quant_names:
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
            else:
                H = H + 0.01 * md * torch.eye(d, device="cuda", dtype=torch.float32)

            w = refs[n].float().to("cuda")
            q_w = correct_gptq(w, H, bits=BITS)
            deltas[n] = q_w.cpu() - refs[n]
            del H, w, q_w

        t_q = time.time() - t0
        for n, m in quant_handles:
            if n in deltas and n in refs:
                m.weight.data.copy_(refs[n].to(device=device, dtype=torch.float16) + deltas[n].to(device=device, dtype=torch.float16))
        ppl = evaluate_ppl()
        restore()
        ec, et, sp = compute_metrics(deltas)
        print(f"  PPL={ppl:.4f} ec={ec:.6f} et={et:.6f} sp={sp:.6f} t={t_q:.1f}s", flush=True)
        results.append({"method": label, "PPL": ppl, "E_calib": ec, "E_test": et, "Spec": sp, "Time": t_q})
        del deltas

    # ── Summary ──
    print(f"\n{'='*80}", flush=True)
    print(f"{'Method':20s} {'PPL':12s} {'E_calib':12s} {'E_test':12s} {'Spec':12s} {'Time':8s}", flush=True)
    print(f"{'-'*80}", flush=True)
    for r in results:
        print(f"{r['method']:20s} {r['PPL']:12.4f} {r['E_calib']:12.6f} {r['E_test']:12.6f} {r['Spec']:12.6f} {r['Time']:8.1f}", flush=True)
    Path("experiments/phase4").mkdir(parents=True, exist_ok=True)
    with open("experiments/phase4/summary.json", "w") as f:
        json.dump({"model": MODEL_NAME, "bits": BITS, "group_size": GROUP_SIZE,
                   "calib_samples": CALIB_SAMPLES, "results": results}, f, indent=2)
    print(f"\nSaved to experiments/phase4/summary.json", flush=True)

if __name__ == "__main__":
    main()
