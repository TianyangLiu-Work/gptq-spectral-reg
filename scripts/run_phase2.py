#!/usr/bin/env python3
"""
Phase 2: GPTQ谱范数正则化实验 — OPT-1.3B 信号搜索

纯自包含实现，只依赖 transformers/datasets/torch 和 gptq_spec/ 内的代码。
使用正确的标准GPTQ（无clamp截断，Hessian diag做scale，误差传播）。
"""
import sys, time, json, torch, math, gc
from pathlib import Path

# ============================================================
# Self-contained model loading & evaluation
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device("cuda:0")
DTYPE = torch.float16

def load_model_and_tokenizer(model_name="facebook/opt-1.3b"):
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def prepare_data(tokenizer, split="validation", n_batches=32, seq_len=2048, batch_size=4):
    """Load wikitext2, concatenate into fixed-length sequences.
    
    Args:
        n_batches: Number of batches to produce.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t.strip()]

    needed_seqs = n_batches * batch_size
    needed_tokens = needed_seqs * seq_len + 1000
    text_budget = ""
    for t in texts:
        text_budget += " " + t
        if len(text_budget.split()) > needed_tokens:
            break

    enc = tokenizer(text_budget, truncation=True, max_length=needed_tokens + 1000, return_tensors="pt")
    all_ids = enc.input_ids[0]

    # Create fixed-length sequences (non-overlapping for speed)
    sequences = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        sequences.append(all_ids[i:i+seq_len])
        if len(sequences) >= n_batches * batch_size:
            break

    batches = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i+batch_size]
        input_ids = torch.stack(chunk)
        attention_mask = torch.ones_like(input_ids)
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batches

def evaluate_ppl(model, batches, device):
    """Compute perplexity over batches with proper padding handling."""
    total_ce = 0.0
    total_tokens = 0
    with torch.no_grad():
        for b in batches:
            input_ids = b["input_ids"].to(device)
            attn_mask = b["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits  # [batch, seq_len, vocab]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attn_mask[:, 1:].contiguous()
            
            # Per-token loss, then mask padding
            vocab_size = shift_logits.size(-1)
            # Set pad positions to -100 (ignored by CE loss)
            shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction="sum"
            )
            total_ce += loss.item()
            total_tokens += shift_mask.sum().item()
    return math.exp(total_ce / total_tokens) if total_tokens > 0 else float("inf")

# ============================================================
# Activation collection
# ============================================================
def collect_linear_handles(model):
    """Find all nn.Linear modules and assign canonical names."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(type("h", (), {
                "canonical_name": name,
                "module": module,
                "module_type": "linear",
                "weight": module.weight,
            })())
    return handles

def get_activations(model, batches, handles, device):
    """Collect input activations for each linear layer."""
    acts = {h.canonical_name: [] for h in handles}
    hooks = []

    for h in handles:
        def mk(n):
            def fn(mod, inp, _):
                x = inp[0].detach().cpu()
                # Skip layers with extremely large hidden dims (e.g. lm_head vocab_proj)
                if x.dim() >= 2 and x.shape[-1] > 4096:
                    return
                if x.dim() == 3:
                    b, s, d = x.shape
                    x = x.reshape(-1, d)
                acts[n].append(x)
            return fn
        hooks.append(h.module.register_forward_hook(mk(h.canonical_name)))

    model.eval()
    with torch.no_grad():
        for batch_idx, b in enumerate(batches):
            if batch_idx > 0 and batch_idx % 4 == 0:
                print(f"    forward {batch_idx}/{len(batches)}", flush=True)
            input_ids = b["input_ids"].to(device)
            attn_mask = b["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attn_mask)
            torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    return {k: torch.cat(v, dim=0) if v else torch.tensor([]) for k, v in acts.items()}

def capture_weights(handles):
    """Capture current weights as reference."""
    return {h.canonical_name: h.weight.data.detach().clone() for h in handles}

def restore_weights(handles, refs):
    """Restore weights from reference."""
    for h in handles:
        n = h.canonical_name
        if n in refs:
            h.weight.data.copy_(refs[n])

def apply_deltas(handles, refs, deltas):
    """Apply weight deltas."""
    for h in handles:
        n = h.canonical_name
        if n in deltas and n in refs:
            h.weight.data.copy_(refs[n] + deltas[n].to(h.weight.device, h.weight.dtype))

# ============================================================
# Local gptq_spec imports
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gptq_spec.hessian import build_hessian, build_hessian_with_options
from gptq_spec.power_iter import top_singular_vector as power_iteration
from gptq_spec.hessian import add_damping

# ============================================================
# Core quantization functions
# ============================================================
def quantize_groupwise(w, bits=4, group_size=32):
    """Simple group-wise RTN."""
    out_dim, in_dim = w.shape
    qmax = float(2 ** (bits - 1) - 1)
    ng = (in_dim + group_size - 1) // group_size
    pad = ng * group_size - in_dim
    wp = w.clone()
    if pad > 0:
        wp = torch.nn.functional.pad(wp, (0, pad), "constant", 0.0)
    wg = wp.view(out_dim, ng, group_size)
    amax = wg.abs().amax(dim=2, keepdim=True)
    s = amax / qmax
    s = s.clamp(min=1e-12)
    q = torch.clamp(torch.round(wg / s), -qmax, qmax)
    dq = q * s
    dq = dq.reshape(out_dim, -1)
    if pad > 0:
        dq = dq[:, :in_dim]
    return dq

def correct_gptq(weight, hessian, bits=4, blocksize=128, damp=1e-2, act_order=True):
    """
    修正版GPTQ: 使用Hessian diagonal做scale，不做clip(-qmax, qmax)，
    直接传播浮点误差。
    """
    out_dim, in_dim = weight.shape
    qmax = float(2 ** (bits - 1) - 1)
    w = weight.detach().float().clone()
    H = hessian.float().to(w.device).clone()

    mean_diag = torch.diag(H).mean()
    H = H + damp * mean_diag * torch.eye(in_dim, device=H.device, dtype=H.dtype)

    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        H = H + damp * mean_diag * torch.eye(in_dim, device=H.device, dtype=H.dtype)
        L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)

    if act_order:
        diag = torch.diag(H).abs()
        perm = torch.argsort(diag, descending=True)
        w = w[:, perm]
        H_inv = H_inv[perm][:, perm]
    else:
        perm = None

    scales = w.abs().max(dim=0, keepdim=True).values / qmax
    scales = scales.clamp(min=1e-12)

    for i1 in range(0, in_dim, blocksize):
        i2 = min(i1 + blocksize, in_dim)
        for j in range(i1, i2):
            col = w[:, j:j+1]
            s = scales[:, j:j+1]

            q = torch.round(col / s)
            q = q.clamp(-qmax, qmax)
            dq = q * s

            err = dq - col
            w[:, j:j+1] = dq

            if j + 1 < in_dim:
                h_diag = H_inv[j, j].clamp(min=1e-12)
                cross = H_inv[j:j+1, j+1:]
                delta = (err / h_diag) @ cross
                w[:, j+1:] += delta

    if perm is not None:
        inv_perm = torch.argsort(perm)
        w = w[:, inv_perm]

    return w.to(dtype=weight.dtype)

# ============================================================
# Metrics
# ============================================================
def compute_metrics_from_acts(delta_dict, refs, calib_acts, test_acts):
    """Compute E_calib, E_test, spectral ratio using pre-collected activations."""
    e_cal, e_tst, specs = [], [], []
    for n, dw in delta_dict.items():
        dw = dw.float().to("cuda")
        w = refs[n]
        xc = calib_acts.get(n)
        xt = test_acts.get(n)
        if xc is not None and xc.numel() > 0:
            out_err = (xc @ dw.T)
            e_cal.append(out_err.norm().item() / xc.norm().item())
        if xt is not None and xt.numel() > 0:
            out_err = (xt @ dw.T)
            e_tst.append(out_err.norm().item() / xt.norm().item())
        spec = (w + dw).norm().item() / w.norm().item()
        specs.append(spec)
    return (float(torch.tensor(e_cal).mean()) if e_cal else 0,
            float(torch.tensor(e_tst).mean()) if e_tst else 0,
            float(torch.tensor(specs).mean()) if specs else 0)

# ============================================================
# Main
# ============================================================
MODEL_NAME = "facebook/opt-1.3b"
BITS = 4
GROUP_SIZE = 32
CALIB_SAMPLES = 4  # number of calibration sequences (each of SEQ_LEN)
EVAL_SAMPLES = 8   # number of evaluation sequences
BATCH_SIZE = 4
SEQ_LEN = 2048

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

print("Preparing data...", flush=True)
all_batches = prepare_data(tokenizer, "train", n_batches=CALIB_SAMPLES // BATCH_SIZE + EVAL_SAMPLES // BATCH_SIZE, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
n_calib = CALIB_SAMPLES // BATCH_SIZE
calib_batches = all_batches[:n_calib]
eval_batches_small = all_batches[n_calib:]

print("Collecting handles...", flush=True)
handles = collect_linear_handles(model)
linear_handles = [h for h in handles if h.module_type == "linear"]
print(f"  {len(linear_handles)} linear layers", flush=True)

refs = capture_weights(linear_handles)

print("Collecting activations...", flush=True)
try:
    calib_acts = get_activations(model, calib_batches, linear_handles, device)
    test_acts = get_activations(model, eval_batches_small[:2], linear_handles, device)
except Exception as e:
    print(f"ERROR in get_activations: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)
print(f"  {sum(1 for v in calib_acts.values() if v.numel() > 0)} layers with calibration activations", flush=True)
ref_gpu = {k: v.float().to("cuda") for k, v in refs.items()}
calib_gpu = {}
for h in linear_handles:
    n = h.canonical_name
    if n in calib_acts and calib_acts[n].numel() > 0:
        calib_gpu[n] = calib_acts[n].float().reshape(-1, refs[n].shape[1]).to("cuda")
test_gpu = {}
for h in linear_handles:
    n = h.canonical_name
    if n in test_acts and test_acts[n].numel() > 0:
        test_gpu[n] = test_acts[n].float().reshape(-1, refs[n].shape[1]).to("cuda")

print(f"  {len(calib_gpu)} layers with calibration activations", flush=True)

# Build raw Hessians
print("Building Hessians...", flush=True)
raw_H = {}
for n, act in calib_gpu.items():
    H, _ = build_hessian(act)
    raw_H[n] = H

# Free calib_gpu to reduce GPU memory pressure before PPL eval
calib_gpu.clear()
torch.cuda.empty_cache()

# ============================================================
# FP16 baseline
# ============================================================
print(f"\n{'='*70}")
print(f"  FP16 baseline")
print(f"{'='*70}")
restore_weights(linear_handles, refs)
fp16_ppl = evaluate_ppl(model, eval_batches_small, device)
print(f"  FP16 PPL={fp16_ppl:.4f}", flush=True)

# ============================================================
# RTN baseline
# ============================================================
print(f"\n{'='*70}")
print(f"  RTN baseline (group-{GROUP_SIZE})")
print(f"{'='*70}")
restore_weights(linear_handles, refs)
deltas_rtn = {}
for h in linear_handles:
    n = h.canonical_name
    if n not in ref_gpu: continue
    w = h.weight.detach().float().to("cuda")
    q_w = quantize_groupwise(w, bits=BITS, group_size=GROUP_SIZE)
    deltas_rtn[n] = q_w - refs[n]
apply_deltas(linear_handles, refs, deltas_rtn)
rtn_ppl = evaluate_ppl(model, eval_batches_small, device)
restore_weights(linear_handles, refs)
e_c, e_t, sp = compute_metrics_from_acts(deltas_rtn, ref_gpu, calib_gpu, test_gpu)
print(f"  RTN PPL={rtn_ppl:.4f} ec={e_c:.6f} et={e_t:.6f} sp={sp:.6f}", flush=True)

results = [
    {"method": "FP16", "PPL": fp16_ppl, "E_calib": 0, "E_test": 0, "Spec": 0, "Time": 0},
    {"method": f"RTN_g{GROUP_SIZE}", "PPL": rtn_ppl, "E_calib": e_c, "E_test": e_t, "Spec": sp, "Time": 0},
]

# ============================================================
# Spectral estimation for spectral-approx
# ============================================================
# Post-GPTQ spectral direction (version D1 from experiment plan):
# 1. Run standard GPTQ to get delta_W
# 2. Power iteration on delta_W to get top right singular vector v
# 3. Use v to construct H' = XX^T + lambda * vv^T

print("Estimating spectral vectors from GPTQ-base deltas...", flush=True)
restore_weights(linear_handles, refs)
spec_deltas = {}
for h in linear_handles:
    n = h.canonical_name
    if n not in raw_H: continue
    w = h.weight.detach().float().to("cuda")
    H_base = calib_acts[n].float().reshape(-1, refs[n].shape[1]).to("cuda")
    H_reg = build_hessian_with_options(H_base, method="none", beta=0.0, base_damp=0.01)
    q_w = correct_gptq(w, H_reg.to("cuda"), bits=BITS, act_order=True)
    spec_deltas[n] = (q_w - refs[n]).cpu()
    del w, H_reg, q_w
    torch.cuda.empty_cache()

spec_v = {}
for n, dw in spec_deltas.items():
    dw = dw.float().to("cuda")
    v, _ = power_iteration(dw, n_iter=20)
    spec_v[n] = v.cpu()
    del dw, v
    torch.cuda.empty_cache()
print(f"  Estimated spectral vectors for {len(spec_v)} layers", flush=True)
del spec_deltas; gc.collect()

# ============================================================
# GPTQ methods
# ============================================================
METHODS = [
    ("GPTQ_base", "none", 0.01),
    ("Frob_0.001", "frobenius", 0.001),
    ("Frob_0.01", "frobenius", 0.01),
    ("Frob_0.1", "frobenius", 0.1),
    ("Damp_0.05", "stronger_damping", 0.05),
    ("Damp_0.1", "stronger_damping", 0.1),
    ("Spectral_0.003", "spectral", 0.003),
    ("Spectral_0.01", "spectral", 0.01),
    ("Spectral_0.03", "spectral", 0.03),
]

for label, method, beta in METHODS:
    print(f"\n{'='*70}")
    print(f"  {label} (β={beta})")
    print(f"{'='*70}")

    restore_weights(linear_handles, refs)
    t0 = time.time()

    deltas = {}
    for h in linear_handles:
        n = h.canonical_name
        if n not in raw_H: continue
        w = h.weight.detach().float().to("cuda")

        if method == "spectral":
            v = spec_v.get(n)
            if v is None: continue
            H_mod = build_hessian_with_options(calib_acts[n].float().reshape(-1, refs[n].shape[1]).to("cuda"), method="spectral", beta=beta,
                                                v=v.to("cuda"), base_damp=0.01)
        else:
            H_mod = build_hessian_with_options(calib_acts[n].float().reshape(-1, refs[n].shape[1]).to("cuda"), method=method, beta=beta, base_damp=0.01)

        q_w = correct_gptq(w, H_mod, bits=BITS, act_order=True)
        deltas[n] = q_w - refs[n]

    t_q = time.time() - t0

    apply_deltas(linear_handles, refs, deltas)
    ppl = evaluate_ppl(model, eval_batches_small, device)
    restore_weights(linear_handles, refs)

    e_c, e_t, sp = compute_metrics_from_acts(deltas, ref_gpu, calib_gpu, test_gpu)

    print(f"  PPL={ppl:.4f} ec={e_c:.6f} et={e_t:.6f} sp={sp:.6f} t={t_q:.1f}s", flush=True)
    results.append({"method": label, "PPL": ppl, "E_calib": e_c, "E_test": e_t, "Spec": sp, "Time": t_q})

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*80}")
print(f"{'Method':20s} {'PPL':12s} {'E_calib':12s} {'E_test':12s} {'Spec':12s} {'Time':8s}")
print(f"{'-'*80}")
for r in results:
    print(f"{r['method']:20s} {r['PPL']:12.4f} {r['E_calib']:12.6f} {r['E_test']:12.6f} {r['Spec']:12.6f} {r['Time']:8.1f}")

Path("experiments/phase2").mkdir(parents=True, exist_ok=True)
with open("experiments/phase2/summary.json", "w") as f:
    json.dump({"model": MODEL_NAME, "bits": BITS, "group_size": GROUP_SIZE, "results": results}, f, indent=2)
print(f"\nSaved to experiments/phase2/summary.json")

print(f"\n{'='*80}")
print(f"{'Analysis (Δ vs FP16)':^80}")
print(f"{'='*80}")
fp16_ref = results[0]["PPL"]
for r in results:
    if r["method"] != "FP16":
        print(f"  {r['method']:20s}: ΔPPL={r['PPL']-fp16_ref:+.4f}  ({r['PPL']/fp16_ref:.4f}x)")
