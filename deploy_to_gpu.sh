#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Deploy SG-CL to ARC Labs RTX 4090 GPU
# ═══════════════════════════════════════════════════════════════════════
#
# This script:
#   1. Checks SSH connection to ARC Labs
#   2. Clones the project from GitHub on the remote machine
#   3. Sets up Python venv + installs all dependencies (using uv)
#   4. Downloads the LLaMA-2-7B model
#   5. Runs the FULL training pipeline (5 tasks × 3 epochs)
#   6. Copies results back to your laptop
#
# Usage:
#   chmod +x deploy_to_gpu.sh
#   ./deploy_to_gpu.sh
#
# Prerequisites:
#   - Must be on campus WiFi
#   - SSH key already shared with ARC Labs
#   - Git repo pushed to GitHub
# ═══════════════════════════════════════════════════════════════════════

set -e

# ── Config ──
SSH_HOST="arcgpu"
REPO_URL="https://github.com/slate012/Semantic-Gated-Continual-Learning-Project.git"
REMOTE_DIR="~/sgcl-project"
LOCAL_RESULTS="./results_from_gpu"
HF_MODEL="meta-llama/Llama-2-7b-hf"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() { echo -e "${GREEN}▶ $1${NC}"; }
print_warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Check connection
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 1: Checking SSH Connection"

print_step "Connecting to ARC Labs GPU..."
if ssh -o ConnectTimeout=10 $SSH_HOST "echo CONNECTION_OK" 2>/dev/null | grep -q "CONNECTION_OK"; then
    echo -e "  ${GREEN}✓ SSH connection successful${NC}"
else
    print_error "Cannot connect to ARC Labs!"
    echo "  Possible reasons:"
    echo "    1. You're not on campus WiFi"
    echo "    2. SSH key not accepted (share ~/.ssh/id_ed25519.pub with ARC Labs)"
    echo "    3. The workstation is down (check Discord)"
    echo ""
    echo "  Try manually: ssh arcgpu"
    exit 1
fi

# Check GPU
print_step "Verifying GPU..."
GPU_INFO=$(ssh $SSH_HOST "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits" 2>/dev/null || echo "FAIL")
if [ "$GPU_INFO" != "FAIL" ]; then
    echo -e "  ${GREEN}✓ GPU: $GPU_INFO MB VRAM${NC}"
else
    print_error "Could not detect GPU on remote machine"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Setup remote environment
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 2: Setting Up Remote Environment"

print_step "Cloning repository and setting up environment..."

ssh $SSH_HOST << 'REMOTE_SETUP'
set -e

echo "── Cleaning any previous project ──"
rm -rf ~/sgcl-project

echo "── Cloning repository ──"
git clone https://github.com/slate012/Semantic-Gated-Continual-Learning-Project.git ~/sgcl-project
cd ~/sgcl-project

echo "── Creating Python virtual environment using uv ──"
if command -v uv &> /dev/null; then
    echo "  Using uv package manager"
    uv venv .venv --python python3
    source .venv/bin/activate
    
    echo "── Installing PyTorch with CUDA ──"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    
    echo "── Installing project dependencies ──"
    uv pip install transformers peft bitsandbytes accelerate datasets
    uv pip install requests huggingface_hub
    uv pip install matplotlib numpy
else
    echo "  uv not found, using pip"
    python3 -m venv .venv
    source .venv/bin/activate
    
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft bitsandbytes accelerate datasets
    pip install requests huggingface_hub
    pip install matplotlib numpy
fi

echo "── Verifying CUDA ──"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo "── Setup complete! ──"
REMOTE_SETUP

echo -e "  ${GREEN}✓ Remote environment ready${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Download model (if not cached)
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 3: Downloading LLaMA-2-7B Model"

print_step "Checking if model needs to be downloaded..."

ssh $SSH_HOST << 'MODEL_DOWNLOAD'
set -e
cd ~/sgcl-project
source .venv/bin/activate

if [ -f "models/llama-2-7b-hf/config.json" ]; then
    echo "  ✓ Model already exists, skipping download"
else
    echo "  Downloading LLaMA-2-7B-hf..."
    echo "  (This may take 10-20 minutes depending on network speed)"
    mkdir -p models/llama-2-7b-hf
    
    # Try huggingface-cli download
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-2-7b-hf',
    local_dir='./models/llama-2-7b-hf',
    local_dir_use_symlinks=False
)
print('  ✓ Model downloaded successfully')
"
fi

# Verify model
python3 -c "
import os, json
config_path = 'models/llama-2-7b-hf/config.json'
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    print(f'  Model: {cfg.get(\"model_type\", \"unknown\")}')
    print(f'  Hidden size: {cfg.get(\"hidden_size\", \"?\")}')
    print(f'  Layers: {cfg.get(\"num_hidden_layers\", \"?\")}')
else:
    print('  ✗ Model config not found!')
    exit(1)
"
MODEL_DOWNLOAD

echo -e "  ${GREEN}✓ Model ready${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Run FULL training pipeline
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 4: Running Full Training Pipeline"

print_step "Starting training on RTX 4090..."
echo "  This will take approximately 30-60 minutes"
echo ""

ssh $SSH_HOST << 'RUN_PIPELINE'
set -e
cd ~/sgcl-project
source .venv/bin/activate

# Make pipeline executable
chmod +x run_full_pipeline.sh

# Run the full pipeline
./run_full_pipeline.sh
RUN_PIPELINE

echo -e "  ${GREEN}✓ Training pipeline complete!${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Copy results back to laptop
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 5: Copying Results to Laptop"

print_step "Downloading outputs..."

mkdir -p "$LOCAL_RESULTS"

# Copy outputs directory
scp -r $SSH_HOST:~/sgcl-project/outputs/* "$LOCAL_RESULTS/" 2>/dev/null || {
    print_warn "Could not copy all output files"
}

# Copy any generated data
scp -r $SSH_HOST:~/sgcl-project/data/*.json "$LOCAL_RESULTS/" 2>/dev/null || true
scp -r $SSH_HOST:~/sgcl-project/data/*.txt "$LOCAL_RESULTS/" 2>/dev/null || true

echo -e "  ${GREEN}✓ Results copied to: $LOCAL_RESULTS/${NC}"

# List what we got
echo ""
echo "  Downloaded files:"
find "$LOCAL_RESULTS" -type f | while read f; do
    size=$(stat -f%z "$f" 2>/dev/null || stat --printf="%s" "$f" 2>/dev/null || echo "?")
    echo "    $(basename "$f") ($size bytes)"
done

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Logout reminder
# ══════════════════════════════════════════════════════════════════════════
print_header "DONE!"

echo "  📊 Results saved to: $LOCAL_RESULTS/"
echo ""
echo "  View results:"
echo "    cat $LOCAL_RESULTS/eval_results.json"
echo "    cat $LOCAL_RESULTS/task_1/results.json"
echo ""
echo "  ⚠️  IMPORTANT: Remember to logout from the remote session:"
echo "    ssh arcgpu logout"
echo ""
echo -e "${GREEN}${BOLD}  🎉 SG-CL Training Complete on RTX 4090!${NC}"
echo ""
