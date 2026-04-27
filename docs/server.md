# Server Knowledge — yumingz5-linux-ml-1

## Slurm GPU Rules (CRITICAL)

### ⚠️ Must Use Slurm for GPU Jobs

**Do NOT start GPU jobs directly in SSH.** The system runs a rogue-killer daemon (`gpu-rogue-kill.service` + `gpu-rogue-kill.timer`) that checks every minute and kills any unauthorized GPU process.

**If your job doesn't need GPU, don't submit via Slurm** — it wastes queue slots. Regular CPU work runs in normal SSH sessions.

### Queue & Tools

| Command | Description |
|---------|-------------|
| `turm` | TUI queue viewer (already installed) |
| `shelper` | Wrapper with common commands pre-configured |

### Resource Limits

| Resource | Limit |
|----------|-------|
| Total concurrent GPU jobs (cluster-wide) | 4 |
| Concurrent GPU jobs per user | 2 |
| MaxSubmitJobsPerUser | 30 |
| MPS (Multi-Process Service) | ✅ Enabled — speeds up multi-job runs |
| MIG Mode | ❌ Not available — cannot finely partition GPUs |
| CPU / Memory | No fixed limits — use ethically |

### Fairshare Scheduling

- Priority type: `PriorityType=priority/multifactor`
- Weights: `PriorityWeightFairshare=100000`, `PriorityWeightAge=1000`, `PriorityWeightPartition=1000`

If you've been using a lot of resources recently, your jobs may be deprioritized for occasional users. This is expected behavior.

## Conda Environment

| Item | Path |
|------|------|
| System conda | `/data/conda/bin/conda` |
| Shared envs (mlusers group) | `/data/conda_envs/` |
| Project conda env | `/home/tyliu/quantization/.conda` |

To init conda in your shell:
```bash
/data/conda/bin/conda init zsh
```

## Data Storage

| Purpose | Path |
|---------|------|
| Personal home | `/data/home/tyliu` |
| Shared workspace (mlusers) | `/data/shared` |

Create a shareable group directory:
```bash
DIR="your_folder_name"
mkdir -p "/data/$DIR"
chown root:mlusers "/data/$DIR"
chmod 2775 "/data/$DIR"
```

## Other Tools

- **Linuxbrew**: `eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv zsh)"`
- **Docker**: Non-root access is granted, use directly.

## SSH & Project Access

| Item | Value |
|------|-------|
| Host | `yumingz5-linux-ml-1` (Tailscale: `100.89.96.108`) |
| User | `tyliu` |
| Auth | SSH key (ed25519) |
| Project dir | `/home/tyliu/gptq-spectral-reg` |
| Conda env | `/home/tyliu/quantization/.conda` |
| HF cache | `/home/tyliu/.cache/huggingface` |

### Typical Workflow

```bash
# 1. SSH in
ssh tyliu@yumingz5-linux-ml-1

# 2. Check queue
squeue -u tyliu

# 3. Pull latest code
cd /home/tyliu/gptq-spectral-reg
git pull

# 4. Submit GPU job
sbatch scripts/submit_phase4.sbatch

# 5. Monitor
turm
# or
squeue -u tyliu
```
