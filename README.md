# GPTQ Spectral Regularization

Experiments comparing **regularization strategies for GPTQ quantization** of OPT models at 1.3B and 6.7B scales.

**Key idea:** Standard GPTQ dampens the Hessian uniformly via $\mathbf{H}' = \mathbf{H} + \lambda \mathbf{I}$. We explore two targeted alternatives:

| Method | Hessian Modification | Intuition |
|--------|---------------------|-----------|
| **GPTQ\_base** | $\mathbf{H}' = \mathbf{H} + \lambda \mathbf{I}$ | Standard GPTQ baseline |
| **Damp\_$\beta$** | $\mathbf{H}' = \mathbf{H} + \beta \cdot \overline{\text{diag}(\mathbf{H})} \cdot \mathbf{I}$ | Stronger isotropic damping |
| **Frob\_$\beta$** | $\mathbf{H}' = \mathbf{H} + (\lambda + \beta) \cdot \overline{\text{diag}(\mathbf{H})} \cdot \mathbf{I}$ | Frobenius norm penalty |
| **Spectral\_$\beta$** | $\mathbf{H}' = \mathbf{H} + \lambda \mathbf{I} + \beta \cdot \overline{\text{diag}(\mathbf{H})} \cdot \mathbf{v} \mathbf{v}^\top$ | Target dominant error direction |

where $\lambda = 0.01$ is the standard damping coefficient, $\beta$ is the regularization strength, and $\mathbf{v}$ is the top right singular vector of $\Delta \mathbf{W} = \mathbf{W}_q - \mathbf{W}$ estimated via power iteration.

---

## Table of Contents

- [Results](#results)
- [Quantized Models](#quantized-models)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Environment](#environment)

---

## Results

### OPT-6.7B (g128, 4-bit)

| Method | PPL $\downarrow$ | Lambada | HellaSwag | ARC-c | WinoGrande |
|--------|-------|---------|-----------|-------|------------|
| FP16 | 10.89 | — | — | — | — |
| GPTQ\_base | 11.29 | — | — | — | — |
| Damp\_0.05 | **11.28** | — | — | — | — |
| Frob\_0.001 | **11.28** | — | — | — | — |
| Spectral\_0.01 | 11.29 | — | — | — | — |

> Downstream zero-shot eval results pending.

### OPT-1.3B (g32, 4-bit)

| Method | PPL $\downarrow$ |
|--------|-------|
| FP16 | 13.28 |
| RTN\_g32 | 14.16 |
| GPTQ\_base | 16.07 |
| Spectral\_0.01 | **16.21** |
| Frob\_0.001 | 16.26 |

### OPT-125M (g128, 4-bit)

| Method | PPL $\downarrow$ |
|--------|-------|
| FP16 | 42.27 |
| RTN\_base | 46.36 |
| GPTQ\_base | 49.42 |
| Frob\_0.001 | 49.42 |

---

## Quantized Models

All models saved on **NFS storage** (`/mnt/host-share/gptq-models/`). Load with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/host-share/gptq-models/opt-6.7b-g128-GPTQ_base",
    torch_dtype=torch.float16, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/host-share/gptq-models/opt-6.7b-g128-GPTQ_base"
)
```

| Directory | Method | Group |
|-----------|--------|-------|
| `opt-6.7b-g128-FP16` | FP16 baseline | — |
| `opt-6.7b-g128-GPTQ_base` | Standard GPTQ | 128 |
| `opt-6.7b-g128-Damp_0.05` | Stronger damping $(\beta = 0.05)$ | 128 |
| `opt-6.7b-g128-Frob_0.001` | Frobenius $(\beta = 0.001)$ | 128 |
| `opt-6.7b-g128-Spectral_0.01` | Spectral $(\beta = 0.01)$ | 128 |
| `opt-6.7b-g32-GPTQ_base` | Standard GPTQ | 32 |
| `opt-6.7b-g32-Damp_0.05` | Stronger damping $(\beta = 0.05)$ | 32 |
| `opt-6.7b-g32-Frob_0.001$` | Frobenius $(\beta = 0.001)$ | 32 |
| `opt-6.7b-g32-Spectral_0.01` | Spectral $(\beta = 0.01)$ | 32 |

Each directory contains:

```
model.safetensors           # Quantized weights
config.json                 # Model configuration
tokenizer.json / tokenizer_config.json
quantization_info.json      # Metadata (method, beta, group_size, quant_time)
```

---

## Project Structure

```
gptq-spectral-reg/
├── gptq_spec/                    # Core library
│   ├── gptq.py                   # GPTQ quantization algorithm
│   ├── power_iter.py             # Power iteration for spectral estimation
│   ├── quant_utils.py            # Group-wise quantization utilities
│   ├── config.py                 # Experiment dataclasses
│   └── __init__.py
├── scripts/
│   ├── run_phase2.py             # Phase 2: OPT-1.3B comparison
│   ├── run_phase4.py             # Phase 4: OPT-6.7B comparison
│   ├── run_phase5.py             # Phase 5: OPT-6.7B g32 sweep
│   ├── save_quantized_models.py  # CLI tool: save quantized models
│   ├── eval_downstream.py        # lm_eval zero-shot evaluation
│   ├── submit_save_model.sh      # Submit all 9 model jobs to Slurm
│   └── *.sbatch                  # Slurm submission scripts
├── experiments/                  # Results (summary.json per phase)
└── docs/server.md                # Server and Slurm usage guide
```

---

## Usage

### Saving Models

```bash
# Submit all 9 model-saving jobs to the Slurm cluster
bash scripts/submit_save_model.sh

# Or run a single configuration directly:
python scripts/save_quantized_models.py \
    --label GPTQ_base --method none --beta 0.01 --group_size 128
```

### Monitor Progress

Each job uses `tqdm` progress bars visible in its log file:

```
Hessian: 100%|████████| 16/16 [00:05<00:00]
Spectral: 100%|███████| 160/160 [02:30<00:00]
Quant: 100%|██████████| 160/160 [01:45<00:00]
Apply: 100%|███████████| 160/160 [00:02<00:00]
```

```bash
squeue -u tyliu
turm          # TUI queue viewer
```

---

## Environment

| Item | Path |
|------|------|
| Server | `tyliu@yumingz5-linux-ml-1` |
| Code | `~/gptq-spectral-reg/` |
| Conda | `/home/tyliu/quantization/.conda` |
| Model storage | `/mnt/host-share/gptq-models/` (467 GB NFS) |
| Data disk | `/data/` (887 GB) |
| GitHub | [TianyangLiu-Work/gptq-spectral-reg](https://github.com/TianyangLiu-Work/gptq-spectral-reg) |
