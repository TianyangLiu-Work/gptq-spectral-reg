#!/usr/bin/env python3
"""Run downstream zero-shot eval on a saved quantized model.

Usage:
  python eval_saved_model.py \
      --model_dir /mnt/host-share/gptq-models/opt-6.7b-g128-GPTQ_base \
      --tasks lambada_openai,hellaswag,arc_challenge,winogrande \
      --limit 200
"""

import sys, time, json, argparse
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Path to saved model directory")
    p.add_argument("--tasks", default="lambada_openai,hellaswag,arc_challenge,winogrande")
    p.add_argument("--limit", type=int, default=200, help="Samples per task")
    p.add_argument("--batch_size", default="auto:4")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    task_list = [t.strip() for t in args.tasks.split(",")]
    label = model_dir.name

    # Load info
    info_path = model_dir / "quantization_info.json"
    has_info = info_path.exists()
    if has_info:
        info = json.loads(info_path.read_text())
        print(f"[{time.strftime('%H:%M:%S')}] Loaded {label}", flush=True)
        print(f"  Method: {info.get('method')}  Group: {info.get('group_size')}  "
              f"Beta: {info.get('beta')}", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Loading {label} (no quantization_info.json)", flush=True)

    # Load model
    print(f"[{time.strftime('%H:%M:%S')}] Loading model from {model_dir}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded on {model.device}", flush=True)

    # Wrap for lm_eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    # Run each task
    results = {"model": label, "tasks": {}}
    for task in task_list:
        print(f"[{time.strftime('%H:%M:%S')}] Running {task}...", flush=True)
        t0 = time.time()
        try:
            result = simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=0,
                limit=args.limit,
                log_samples=False,
            )
            acc = result["results"][task].get("acc,none",
                   result["results"][task].get("acc", "N/A"))
            acc_stderr = result["results"][task].get("acc_stderr,none",
                         result["results"][task].get("acc_stderr", "N/A"))
            elapsed = time.time() - t0
            results["tasks"][task] = {"acc": acc, "acc_stderr": acc_stderr, "time": round(elapsed, 1)}
            print(f"  {task:20s}: acc={acc:.4f} ± {acc_stderr:.4f}  ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"  {task:20s}: ERROR {e}", flush=True)
            results["tasks"][task] = {"error": str(e)}

    # Save results
    result_dir = Path("experiments") / "downstream"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{label}.json"
    results["total_time"] = round(time.time() - t_start, 1)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Results saved to {result_path}", flush=True)

    # Summary line for table copy-paste
    accs = [v.get("acc", 0) for v in results["tasks"].values() if isinstance(v.get("acc"), (int, float))]
    if accs:
        avg = sum(accs) / len(accs)
        acc_str = " | ".join(f"{v.get(\"acc\", \"-\"):.4f}" for v in results["tasks"].values() if isinstance(v.get(\"acc\"), (int, float)))
        print(f"\nSUMMARY: {label:40s} | {acc_str}  | avg={avg:.4f}", flush=True)

    elapsed = time.time() - t_start
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {elapsed:.1f}s", flush=True)


t_start = time.time()
if __name__ == "__main__":
    main()
