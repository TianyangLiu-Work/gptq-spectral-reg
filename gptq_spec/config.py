"""Configuration dataclasses for GPTQ spectral regularization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GPTQConfig:
    """GPTQ quantization configuration."""

    bits: int = 4
    group_size: int = 128
    blocksize: int = 128
    damp: float = 0.01
    act_order: bool = False
    sym: bool = True


@dataclass
class RegConfig:
    """Regularization configuration.

    method:
        - "none": Standard GPTQ with base damping.
        - "stronger_damping": Increase damping coefficient.
        - "frobenius": H' = H + (damp + beta) * mean_diag * I
        - "spectral": H' = H + damp*mean_diag*I + beta*mean_diag*vv^T
    """

    method: Literal["none", "stronger_damping", "frobenius", "spectral"] = "none"
    beta: float = 0.01
    power_iter_steps: int = 20


@dataclass
class CalibConfig:
    """Calibration data configuration."""

    dataset: str = "wikitext2"
    calib_samples: int = 128
    calib_seqlen: int = 2048
    eval_samples: int = 128
    batch_size: int = 4
    seed: int = 0


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    eval_ppl: bool = True
    eval_layer_errors: bool = True
    eval_spectral_metrics: bool = True
    save_results: bool = True
    output_dir: str = "experiments/phase1"


@dataclass
class ExperimentConfig:
    """Combined experiment configuration."""

    model_name: str = "facebook/opt-125m"
    device: str = "cuda:0"
    experiment_name: str = "phase1_opt125m"

    gptq: GPTQConfig = field(default_factory=GPTQConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Predefined experiment sweeps
FROBENIUS_BETAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
STRONGER_DAMPS = [0.03, 0.05, 0.1]
SPECTRAL_BETAS = [0.003, 0.01, 0.03]
CALIB_SWEEP = [32, 128, 512]
BIT_SWEEP = [4, 3]
MODEL_SWEEP_P2 = ["facebook/opt-1.3b", "facebook/opt-2.7b"]
MODEL_SWEEP_P4 = ["facebook/opt-6.7b"]


def make_baseline_config() -> ExperimentConfig:
    """Create a standard GPTQ baseline config."""
    cfg = ExperimentConfig()
    cfg.reg.method = "none"
    return cfg


def make_frobenius_config(beta: float) -> ExperimentConfig:
    """Create a Frobenius-regularized config."""
    cfg = ExperimentConfig()
    cfg.reg.method = "frobenius"
    cfg.reg.beta = beta
    return cfg


def make_damping_config(damp: float) -> ExperimentConfig:
    """Create a stronger damping config."""
    cfg = ExperimentConfig()
    cfg.reg.method = "stronger_damping"
    cfg.reg.beta = damp
    return cfg


def make_spectral_config(beta: float) -> ExperimentConfig:
    """Create a spectral-approx config."""
    cfg = ExperimentConfig()
    cfg.reg.method = "spectral"
    cfg.reg.beta = beta
    return cfg
