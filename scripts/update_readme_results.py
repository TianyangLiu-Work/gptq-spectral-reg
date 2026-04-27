#!/usr/bin/env python3
"""Parse downstream eval JSON results and update README.md tables."""

import json
import os
import glob
import re

PROJECT_DIR = os.path.expanduser("~/gptq-spectral-reg")
EVAL_DIR = os.path.join(PROJECT_DIR, "experiments", "downstream")
README_PATH = os.path.join(PROJECT_DIR, "README.md")
MODELS_DIR = "/mnt/host-share/gptq-models"

# === Step 1: Collect eval results ===
results = {}  # model_name -> {task: accuracy}

for fpath in sorted(glob.glob(os.path.join(EVAL_DIR, "opt-*.json"))):
    name = os.path.basename(fpath).replace(".json", "")
    with open(fpath) as f:
        data = json.load(f)
    
    # lm_eval results are in data['results']
    task_dict = {}
    for task, metrics in data.get("results", {}).items():
        # Try various accuracy keys
        for key in ["acc_norm,none", "acc,none", "acc"]:
            if key in metrics:
                task_dict[task] = metrics[key]
                break
    
    results[name] = task_dict
    print(f"  {name}: {len(task_dict)} tasks loaded")

# === Step 2: Collect PPL from quantization_info.json ===
ppl_data = {}
for d in sorted(glob.glob(os.path.join(MODELS_DIR, "opt-6.7b-*"))):
    name = os.path.basename(d)
    info_path = os.path.join(d, "quantization_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        ppl_data[name] = info.get("wikitext_ppl", None)

print(f"\nPPL data: {len(ppl_data)} models")

# === Step 3: Build task mapping ===
# Map full task names to short display names
TASK_ALIASES = {
    "lambada_openai": "Lambada",
    "lambada_standard": "Lambada",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-c",
    "winogrande": "WinoGrande",
    "piqa": "PIQA",
}

TASK_ORDER = ["Lambada", "HellaSwag", "ARC-c", "WinoGrande"]

# === Step 4: Sort models ===
def model_sort_key(name):
    """Sort: g128 first (FP16, GPTQ_base, Damp, Frob, Spectral), then g32."""
    g128_models = ["FP16", "GPTQ_base", "Damp_0.05", "Frob_0.001", "Spectral_0.01"]
    g32_models = ["GPTQ_base", "Damp_0.05", "Frob_0.001", "Spectral_0.01"]
    
    # Extract group size and method
    parts = name.replace("opt-6.7b-", "")
    if parts == "FP16":
        return (0, 0)
    
    group = parts.split("-")[0]  # g128 or g32
    method = parts.replace(f"{group}-", "")
    
    if group == "g128":
        if method in g128_models:
            return (1, g128_models.index(method))
        return (1, 99)
    else:
        if method in g32_models:
            return (2, g32_models.index(method))
        return (2, 99)

sorted_models = sorted(results.keys(), key=model_sort_key)
# Also add models that have PPL but no eval results yet
all_sorted = sorted(set(list(ppl_data.keys()) + list(results.keys())), key=model_sort_key)

# === Step 5: Build table rows ===
def format_val(val, best_in_col=False):
    if val is None:
        return "—"
    try:
        v = float(val)
        s = f"{v:.4f}"
        if best_in_col and v == best_in_col:
            return f"**{s}**"
        return s
    except (ValueError, TypeError):
        return str(val)

# Calculate best per column (only among quantized models, not FP16)
tables = {
    "g128": {"title": "### OPT-6.7B (g128, 4-bit)", "models": [], "columns": ["PPL"] + TASK_ORDER},
    "g32-ppl": {"title": "### OPT-6.7B (g32, 4-bit)", "models": [], "columns": ["PPL"] + TASK_ORDER},
}

# Group models
g128_models = [m for m in all_sorted if "g128" in m or m == "opt-6.7b-FP16"]
g32_models = [m for m in all_sorted if "g32" in m]
if "opt-6.7b-FP16" not in g128_models:
    g128_models = ["opt-6.7b-FP16"] + [m for m in g128_models if m != "opt-6.7b-FP16"]

print(f"\n=== Table Construction ===")
print(f"  g128 models: {g128_models}")
print(f"  g32 models: {g32_models}")

def compute_best_quantized(models, task_order):
    """Find best values among non-FP16 models for each column."""
    best = {}
    for task in task_order:
        quant_vals = []
        for m in models:
            if m == "opt-6.7b-FP16":
                continue
            res = results.get(m, {})
            for alias, display in TASK_ALIASES.items():
                if display == task and alias in res:
                    quant_vals.append(float(res[alias]))
        if quant_vals:
            # Higher is better for accuracy
            best[task] = max(quant_vals)
    return best

def build_table(models, task_order, use_ppl=True):
    """Build markdown table rows."""
    best = compute_best_quantized(models, task_order)
    
    header_method = "| Method | PPL $\\downarrow$" + "".join(f" | {t}" for t in task_order) + " |"
    header_sep = "|--------|-------" + "|".join("-----------" for _ in task_order) + "|"
    
    rows = []
    for m in models:
        ppl = ppl_data.get(m)
        method = m.replace("opt-6.7b-", "")
        method_display = method
        
        cols = [f"| {method_display}"]
        
        # PPL column
        if ppl is not None:
            cols.append(f" {ppl:.2f} ")
        else:
            cols.append(" — ")
        
        # Task columns
        res = results.get(m, {})
        for task in task_order:
            val = None
            for alias, display in TASK_ALIASES.items():
                if display == task and alias in res:
                    val = res[alias]
                    break
            
            if val is not None:
                v = float(val)
                s = f"{v:.4f}"
                is_best = (method != "FP16" and task in best and abs(v - best[task]) < 1e-6)
                if is_best:
                    s = f"**{s}**"
                cols.append(f" {s} ")
            else:
                cols.append(" — ")
        
        rows.append("|".join(cols) + "|")
    
    return header_method, header_sep, rows

# === Step 6: Generate full table section ===
def format_table(group_models, title, task_order):
    if not group_models:
        return ""
    
    header_method, header_sep, rows = build_table(group_models, task_order)
    
    lines = [f"\n{title}\n", header_method, header_sep] + rows
    return "\n".join(lines)

table_g128 = format_table(g128_models, "### OPT-6.7B (g128, 4-bit)", TASK_ORDER)
table_g32 = format_table(g32_models, "### OPT-6.7B (g32, 4-bit)", TASK_ORDER)

print(f"\n=== Generated Tables ===\n")
if table_g128:
    print("g128 table:")
    print(table_g128)
if table_g32:
    print("g32 table:")
    print(table_g32)

# === Step 7: Update README.md ===
if not os.path.exists(README_PATH):
    print(f"README not found at {README_PATH}")
    sys.exit(1)

with open(README_PATH) as f:
    readme = f.read()

# Find the Results section
results_start = readme.find("## Results")
results_end = readme.find("---", results_start + 20) if "---" in readme[results_start + 20:] else len(readme)

if results_start == -1:
    print("ERROR: Could not find ## Results section in README")
    sys.exit(1)

# Generate new results section
new_results = "## Results\n\n"
new_results += "### OPT-6.7B (g128, 4-bit)\n\n"

# Build g128 table content
if g128_models:
    h_method, h_sep, rows = build_table(g128_models, TASK_ORDER)
    new_results += h_method + "\n" + h_sep + "\n"
    for r in rows:
        new_results += r + "\n"
else:
    new_results += "> Results pending.\n"

new_results += "\n### OPT-6.7B (g32, 4-bit)\n\n"

# Build g32 table content
if g32_models:
    h_method, h_sep, rows = build_table(g32_models, TASK_ORDER)
    new_results += h_method + "\n" + h_sep + "\n"
    for r in rows:
        new_results += r + "\n"
else:
    new_results += "> Results pending.\n"

# Update README
new_readme = readme[:results_start] + new_results + "\n---\n" + readme[results_end + 3:]

with open(README_PATH, 'w') as f:
    f.write(new_readme)

print(f"\n=== README UPDATED ===")
print(f"  File: {README_PATH}")

# Report which models have results
print(f"\n=== Summary ===")
for m in sorted_models:
    ppl = ppl_data.get(m, None)
    res_count = len(results.get(m, {}))
    tasks = ", ".join(results.get(m, {}).keys()) if results.get(m, {}) else "—"
    print(f"  {m}: PPL={ppl}, {res_count} tasks ({tasks})")
