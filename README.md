# GPTQ Spectral Regularization

Experiments comparing regularization strategies for GPTQ quantization
of OPT models at 1.3B and 6.7B scales.

**Key idea:** Standard GPTQ dampens the Hessian uniformly via `H' = H + λ·I`.
We explore two targeted alternatives:
- **Frobenius**: `H' = H + (λ + β)·I` — uniform stronger dampening
- **Spectral**: `H' = H + λ·I + β·vvᵀ` — penalize the dominant error direction

---

## Results

### OPT-6.7B (g128, 4-bit)

| Method | PPL ↓ | Lambada | HellaSwag | ARC-c | WinoGrande |
|--------|-------|---------|-----------|-------|------------|
| FP16 | 10.89 | — | — | — | — |
| GPTQ_base | 11.29 | — | — | — | — |
| Damp_0.05 | **11.28** | — | — | — | — |
| Frob_0.001 | **11.28** | — | — | — | — |
| Spectral_0.01 | 11.29 | — | — | — | — |

> *Downstream zero-shot eval results pending (job 322 completed).*

### OPT-1.3B (g32, 4-bit)

| Method | PPL ↓ |
|--------|-------|
| FP16 | 13.28 |
| RTN_g32 | 14.16 |
| GPTQ_base | 16.07 |
| Spectral_0.01 | **16.21** |
| Frob_0.001 | 16.26 |

### OPT-125M (g128, 4-bit)

| Method | PPL ↓ |
|--------|-------|
| FP16 | 42.27 |
| RTN_base | 46.36 |
| GPTQ_base | 49.42 |
| Frob_0.001 | 49.42 |

---

## Quantized Models

All models are saved as complete HuggingFace model directories
and can be loaded directly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/host-share/gptq-models/opt-6.7b-g128-GPTQ_base"
)
```

Available on NFS (`/mnt/host-share/gptq-models/`):

| Directory | Method | Group |
|-----------|--------|-------|
| `opt-6.7b-g128-FP16` | FP16 baseline | — |
| `opt-6.7b-g128-GPTQ_base` | Standard GPTQ | 128 |
| `opt-6.7b-g128-Damp_0.05` | Stronger damping | 128 |
| `opt-6.7b-g128-Frob_0.001` | Frobenius β=0.001 | 128 |
| `opt-6.7b-g128-Spectral_0.01` | Spectral β=0.01 | 128 |
| `opt-6.7b-g32-GPTQ_base` | Standard GPTQ | 32 |
| `opt-6.7b-g32-Damp_0.05` | Stronger damping | 32 |
| `opt-6.7b-g32-Frob_0.001` | Frobenius β=0.001 | 32 |
| `opt-6.7b-g32-Spectral_0.01` | Spectral β=0.01 | 32 |

Each directory contains:
- `model.safetensors` — quantized weights
- `config.json` — model configuration
- `tokenizer.json` / `tokenizer_config.json`
- `quantization_info.json` — metadata (method, beta, group_size, quant_time)

---

## Regularization Methods

| Method | Hessian Modification | Intuition |
|--------|---------------------|-----------|
| **GPTQ_base** | None (damp=0.01) | Standard GPTQ baseline |
| **Damp\\_β** | `H' = H + β·mean(diag(H))·I` | Stronger isotropic damping |
| **Frob\\_β** | `H' = H + (damp+β)·mean(diag(H))·I` | Frobenius norm penalty |
| **Spectral\\_β** | `H' = H + damp·mean(diag(H))·I + β·mean(diag(H))·vvᵀ` | Target dominant error direction |

---

## Project Structure

```
gptq-spectral-reg/
├── gptq_spec/                    # Core library
│   ├── gptq.py                   # GPTQ quantization algorithm
│   ├── power_iter.py             # Power iteration for spectral estimation
│   ├── quant_utils.py            # Group-wise quantization utilities
│   ├── config.py                 # Experiment configuration dataclasses
│   └── __init__.py
├── scripts/
│   ├── run_phase2.py             # Phase 2: OPT-1.3B comparison
│   ├── run_phase4.py             # Phase 4: OPT-6.7B comparison
│   ├── run_phase5.py             # Phase 5: OPT-6.7B g32 sweep
│   ├── save_quantized_models.py  # Save quantized models (CLI: --label --method --beta --group_size)
│   ├── eval_downstream.py        # lm_eval zero-shot evaluation
│   ├── submit_save_model.sh      # Submit all 9 model-saving jobs to Slurm
│   └── *.sbatch                  # Slurm submission scripts
├── experiments/                  # Results (summary.json per phase)
├── models/                       # (local cached models)
└── docs/server.md                # Server & Slurm usage guide
```

---

## Running on Slurm

```bash
# Save all quantized models (9 parallel jobs)
bash scripts/submit_save_model.sh

# Monitor
squeue -u tyliu
turm          # TUI viewer

# Downstream eval (after models are saved)
# (coming soon: eval from saved models directly)
```

Each job uses tqdm progress bars visible in logs:

```
Hessian: 100%|████████| 16/16 [00:05<00:00]
Spectral: 100%|███████| 160/160 [02:30<00:00]
Quant: 100%|██████████| 160/160 [01:45<00:00]
Apply: 100%|███████████| 160/160 [00:02<00:00]
```

---

## Environment

| | Path |
|---|---|
| Server | `tyliu@yumingz5-linux-ml-1` |
| Code | `~/gptq-spectral-reg/` |
| Conda | `/home/tyliu/quantization/.conda` |
| Model output | `/mnt/host-share/gptq-models/` (467G NFS) |
| Data disk | `/data/` (887G, ~50G free) |
| GitHub | `github.com/TianyangLiu-Work/gptq-spectral-reg` |
