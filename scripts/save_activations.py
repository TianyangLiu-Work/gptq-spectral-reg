#!/usr/bin/env python3
"""Save raw activations (hidden states) from OPT-6.7B calibration runs to NFS.

Collects per-layer input activations from the linear layers that GPTQ
quantizes (attention proj + FFN), using the standard WikiText-2 calibration
dataset (128 samples × 2048 tokens).

Usage:
  python scripts/save_activations.py

Output: /mnt/host-share/gptq-models/opt-6.7b-activations/*.pt
"""

import sys, time, json, math, torch, argparse
from pathlib import Path

from tqdm import tqdm

device = torch.device("cuda:0")
MODEL_NAME = "facebook/opt-6.7b"
CALIB_SAMPLES = 128
BATCH_SIZE = 8
SEQ_LEN = 2048
BASE_DIR = Path("/mnt/host-share/gptq-models")


def main():
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === Save Activations for OPT-6.7B ===", flush=True)

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

    # ── Collect activations ──
    print(f"[{time.strftime('%H:%M:%S')}] Collecting activations ({CALIB_SAMPLES} samples)...", flush=True)
    activations = {n: [] for n, _ in quant_handles}
    hooks = []
    for name, mod in quant_handles:
        def mk(n):
            def fn(m, inp, _):
                x = inp[0].detach().float()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                activations[n].append(x.cpu())
            return fn
        hooks.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        for bi, b in enumerate(tqdm(calib_batches, desc="Forward", unit="batch")):
            model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
    for h in hooks:
        h.remove()
    print(f"  Collected activations for {len(activations)} layers", flush=True)

    # ── Save to NFS ──
    out_dir = BASE_DIR / "opt-6.7b-activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Saving to {out_dir}...", flush=True)
    for n, x_list in tqdm(activations.items(), desc="Save activation", unit="layer"):
        if len(x_list) == 0:
            continue
        acts = torch.cat(x_list, dim=0).contiguous()
        safe_name = n.replace(".", "_")
        torch.save(acts, out_dir / f"{safe_name}.pt")
    
    # Save metadata
    info = {
        "model_name": MODEL_NAME,
        "calib_samples": CALIB_SAMPLES,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "num_layers": len(activations),
        "layer_names": list(activations.keys()),
        "time_sec": round(time.time() - t_start, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Show size
    total_size = sum((out_dir / f"{n.replace('.', '_')}.pt").stat().st_size
                     for n in activations if (out_dir / f"{n.replace('.', '_')}.pt").exists())
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {time.time() - t_start:.1f}s", flush=True)
    print(f"  Saved {len(activations)} activation tensors", flush=True)
    print(f"  Total size: {total_size / 1024**3:.2f} GB", flush=True)
    print(f"  Location: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
