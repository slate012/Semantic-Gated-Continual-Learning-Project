# 🖥️ SG-CL GPU Setup Guide — ARC Labs RTX 4090 (24 GB)

Complete guide to run the full SG-CL training pipeline on your college's RTX 4090 GPU.

---

## Quick Start (One Command)

```bash
# 1. Make sure you're on campus WiFi
# 2. Check connection
python3 check_gpu_connection.py

# 3. Deploy & train (fully automated)
bash deploy_to_gpu.sh
```

That's it! The script will clone the repo, install deps, download the model, train, and copy results back.

---

## SSH Setup

### Already Configured ✅

Your SSH config (`~/.ssh/config`) has the `arcgpu` alias:

```
Host arcgpu
  HostName 10.3.32.62
  Port 2222
  User mluser
  IdentityFile ~/.ssh/id_ed25519
```

**Connect manually:** `ssh arcgpu`

---

## RTX 4090 Optimizations

| Setting | Old (6GB A2000) | New (24GB RTX 4090) |
|---------|-----------------|---------------------|
| Model quantization | 4-bit NF4 (~4.5 GB) | **4-bit NF4 (~4.5 GB)** |
| Batch size | 1 | **4** |
| Max sequence length | 256 | **512** |
| Gradient accumulation | 8 | **4** |
| Effective batch size | 8 | **16** |

**Estimated VRAM:** ~9.2 GB / 24.0 GB ✅ (plenty of headroom)

**Estimated training time:** ~30-60 minutes (vs ~2.5 hours on A2000)

---

## What Gets Run

The full pipeline trains the **actual LLaMA-2-7B model** with LoRA adapters:

1. **Environment check** — GPU, CUDA, dependencies
2. **Dataset generation** — Builds synthetic training data (5 tasks)
3. **Smoke test** — Verifies VRAM fits with 1 epoch
4. **Full training** — 5 tasks × 3 epochs with symbolic gating
5. **Evaluation** — Baseline vs SG-CL comparison
6. **Results** — Saved to `outputs/` and copied to your laptop

---

## Manual Step-by-Step (Alternative)

If you prefer to run things manually instead of using `deploy_to_gpu.sh`:

```bash
# 1. SSH into the machine
ssh arcgpu

# 2. Clone the repo
git clone https://github.com/slate012/Semantic-Gated-Continual-Learning-Project.git ~/sgcl-project
cd ~/sgcl-project

# 3. Setup environment
uv venv .venv --python python3
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers peft bitsandbytes accelerate datasets
uv pip install requests huggingface_hub matplotlib numpy

# 4. Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')"

# 5. Download model (if needed)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-2-7b-hf', local_dir='./models/llama-2-7b-hf')
"

# 6. Run the full pipeline
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh

# 7. BEFORE logging out — copy results to your laptop (from your Mac):
#    scp -r arcgpu:~/sgcl-project/outputs/* ./results_from_gpu/

# 8. Always logout when done
logout
```

---

## Important Reminders

| ⚠️ Rule | Details |
|---------|---------|
| **Campus only** | Must be on campus WiFi/LAN (9 AM - 4:30 PM, Mon-Fri) |
| **No persistent storage** | All data is lost when you logout! |
| **Save results first** | Copy outputs via `scp` or push to GitHub before logging out |
| **Always logout** | Run `logout` before closing the terminal |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| SSH connection refused | Are you on campus WiFi? Check with ARC Labs Discord |
| Permission denied | Share `~/.ssh/id_ed25519.pub` with ARC Labs |
| `CUDA out of memory` | Reduce `--batch-size 2` or `--max-length 256` |
| Model download fails | Need HuggingFace access token for LLaMA-2. Run `huggingface-cli login` |
| `bitsandbytes` errors | `pip install bitsandbytes --no-cache-dir` |
| Slow training | Check `nvidia-smi` — another user may be using the GPU |

---

## Output Structure

```
outputs/
├── task_1/ ... task_5/
│   ├── adapter/           ← LoRA weights
│   ├── results.json       ← Loss, runtime
│   └── training_stats.json
├── eval_results.json      ← Baseline vs SG-CL comparison
└── smoke_test/            ← Deleted after passing
```

---

## Files Modified for GPU

| File | Change |
|------|--------|
| `check_gpu_connection.py` | Pre-flight connection + GPU checker |
| `deploy_to_gpu.sh` | One-command deploy + train + copy results |
| `run_full_pipeline.sh` | Batch=4, seq=512, RTX 4090 settings |
| `run_training.py` | GPU-required guard, updated defaults |
| `src/training/sgcl_trainer.py` | CUDA-only, seq_length=512 |
| `~/.ssh/config` | `arcgpu` SSH alias added |
