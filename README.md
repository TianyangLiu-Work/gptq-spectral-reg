# GPTQ Spectral Regularization

Experiments comparing regularization strategies for GPTQ quantization:

| Method | Hessian Modification | Idea |
|--------|---------------------|------|
| **GPTQ_base** | None (standard damping) | Baseline |
| **Damp\_0.05** | Increase damping coefficient | Stronger regularization |
| **Frob\_β** | `H' = H + β · mean(diag(H)) · I` | Frobenius norm penalty |
| **Spectral\_β** | `H' = H + β · mean(diag(H)) · vvᵀ` | Target dominant error direction |

## Results Summary

| Model | Bits | Group | Method | PPL ↓ |
|-------|------|-------|--------|-------|
| OPT-125M | 4 | 128 | FP16 | 42.27 |
| | | | RTN_base | 46.36 |
| | | | GPTQ_base | 49.42 |
| | | | Frob_0.001 | 49.42 |
| OPT-1.3B | 4 | 32 | FP16 | 13.28 |
| | | | RTN_g32 | 14.16 |
| | | | GPTQ_base | 16.07 |
| | | | Spectral_0.01 | **16.21** |
| | | | Frob_0.001 | 16.26 |
| OPT-6.7B | 4 | 128 | FP16 | 10.89 |
| | | | GPTQ_base | 11.29 |
| | | | Damp_0.05 | **11.28** |
| | | | Frob_0.001 | 11.28 |
| | | | Spectral_0.01 | 11.29 |

Spectral regularization matches or improves perplexity across model scales.

## Directory Structure

```
gptq-spectral-reg/
├── pyproject.toml           # Package metadata
├── .gitignore
├── README.md
├── gptq_spec/               # Core library
│   ├── __init__.py
│   ├── config.py            # Dataclass configs & predefined sweeps
│   ├── gptq.py              # GPTQ quantization algorithm
│   ├── hessian.py           # Hessian construction & regularization
│   ├── power_iter.py        # Power iteration for spectral analysis
│   └── quant_utils.py       # Group-wise quantization utilities
├── scripts/                 # Run scripts & Slurm job files
│   ├── run_phase1.sh        # Shell wrapper (for interactive use)
│   ├── run_phase2.py        # Phase 2: OPT-1.3B full comparison
│   ├── run_phase4.py        # Phase 4: OPT-6.7B scaled comparison
│   ├── eval_downstream.py   # Downstream zero-shot eval (lm_eval)
│   ├── submit_phase4.sbatch          # Slurm: Phase 4 (with MPS)
│   ├── submit_phase4_v2.sbatch       # Slurm: Phase 4 (no MPS)
│   ├── submit_downstream_eval.sbatch # Slurm: downstream eval
│   ├── run_phase3.sbatch    # Slurm: Phase 3 (OPT-1.3B via Slurm)
│   └── phase3_template.sbatch  # Template for new Slurm jobs
├── experiments/             # Results
│   ├── phase1/summary.json  # OPT-125M results
│   ├── phase2/summary.json  # OPT-1.3B results
│   └── phase4/summary.json  # OPT-6.7B results
└── configs/                 # (reserved for YAML configs)
```

## Run

On Slurm cluster:
```bash
# Phase 4 (OPT-6.7B)
sbatch scripts/submit_phase4.sbatch

# Downstream eval
sbatch scripts/submit_downstream_eval.sbatch
```

Interactive:
```bash
# Activate environment, then:
cd gptq-spectral-reg
export PYTHONPATH=$PWD:/path/to/other/libraries
python scripts/run_phase4.py
```
