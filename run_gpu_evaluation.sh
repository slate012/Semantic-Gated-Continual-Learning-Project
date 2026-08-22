#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Run Real SG-CL Evaluation on ARC Labs RTX 4090 GPU
# ═══════════════════════════════════════════════════════════════════════
#
# This script:
#   1. Checks SSH connection to ARC Labs
#   2. Syncs updated evaluation code + trained adapter to the remote
#   3. Runs evaluate_model.py --compare on the RTX 4090
#   4. Copies eval_results.json back to results_from_gpu/outputs/
#   5. Optionally regenerates plots locally
#
# Usage:
#   chmod +x run_gpu_evaluation.sh
#   ./run_gpu_evaluation.sh
#
# Prerequisites:
#   - Must be on campus WiFi
#   - SSH key already shared with ARC Labs
#   - Trained adapter exists at results_from_gpu/outputs/task_5/adapter
# ═══════════════════════════════════════════════════════════════════════

set -e

# ── Config ──
SSH_HOST="arcgpu"
REMOTE_DIR="~/sgcl-project"
LOCAL_RESULTS="./results_from_gpu"
LOCAL_ADAPTER="$LOCAL_RESULTS/outputs/task_5/adapter"
REMOTE_ADAPTER="$REMOTE_DIR/outputs/task_5/adapter"
REMOTE_EVAL_OUTPUT="$REMOTE_DIR/outputs"
LOCAL_EVAL_OUTPUT="$LOCAL_RESULTS/outputs"
EVAL_DATA="./data/evaluation_set.json"

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

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Validate local files
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 2: Validating Local Files"

if [ ! -d "$LOCAL_ADAPTER" ]; then
    print_error "Trained adapter not found at $LOCAL_ADAPTER"
    echo "  Please copy the adapter from the GPU first, or re-run training."
    exit 1
fi
echo -e "  ${GREEN}✓ Adapter found${NC}"

if [ ! -f "$EVAL_DATA" ]; then
    print_error "Evaluation data not found at $EVAL_DATA"
    exit 1
fi
echo -e "  ${GREEN}✓ Evaluation data found${NC}"

if [ ! -f "evaluate_model.py" ]; then
    print_error "evaluate_model.py not found in current directory"
    exit 1
fi
echo -e "  ${GREEN}✓ evaluate_model.py found${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Sync updated files to remote
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 3: Syncing Files to Remote GPU"

print_step "Checking remote project exists..."
if ssh $SSH_HOST "[ -d $REMOTE_DIR ] && echo EXISTS" 2>/dev/null | grep -q "EXISTS"; then
    echo -e "  ${GREEN}✓ Remote project found${NC}"
else
    print_warn "Remote project not found. It may have been cleaned up."
    echo "  Please re-run: bash deploy_to_gpu.sh"
    exit 1
fi

print_step "Copying updated evaluate_model.py..."
scp evaluate_model.py $SSH_HOST:$REMOTE_DIR/

print_step "Copying evaluation data..."
scp $EVAL_DATA $SSH_HOST:$REMOTE_DIR/data/

print_step "Copying trained adapter..."
ssh $SSH_HOST "mkdir -p $REMOTE_ADAPTER"
scp -r $LOCAL_ADAPTER/* $SSH_HOST:$REMOTE_ADAPTER/

echo -e "  ${GREEN}✓ Files synced${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Run evaluation on remote GPU
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 4: Running Real Model Evaluation on RTX 4090"

print_step "This will take approximately 10–20 minutes for 1000 questions..."
echo ""

ssh $SSH_HOST << REMOTE_EVAL
set -e
cd $REMOTE_DIR
source .venv/bin/activate

echo "── GPU check ──"
python3 -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

echo ""
echo "── Running baseline vs SG-CL comparison ──"
python3 evaluate_model.py \
    --model $REMOTE_DIR/models/llama-2-7b-hf \
    --adapter $REMOTE_ADAPTER \
    --eval-data $REMOTE_DIR/data/evaluation_set.json \
    --output $REMOTE_EVAL_OUTPUT \
    --compare \
    --verbose
REMOTE_EVAL

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Copy results back
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 5: Copying Results Back"

mkdir -p "$LOCAL_EVAL_OUTPUT"

print_step "Downloading eval_results.json..."
scp $SSH_HOST:$REMOTE_EVAL_OUTPUT/eval_results.json "$LOCAL_EVAL_OUTPUT/"

echo -e "  ${GREEN}✓ Results saved to: $LOCAL_EVAL_OUTPUT/eval_results.json${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Regenerate plots locally
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 6: Regenerating Plots from Real Results"

if [ -f "$LOCAL_RESULTS/plot_results.py" ]; then
    print_step "Running plot_results.py..."
    cd "$LOCAL_RESULTS" && python3 plot_results.py
    cd - >/dev/null
fi

if [ -f "$LOCAL_RESULTS/plot_all_results.py" ]; then
    print_step "Running plot_all_results.py..."
    cd "$LOCAL_RESULTS" && python3 plot_all_results.py
    cd - >/dev/null
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Summary
# ══════════════════════════════════════════════════════════════════════════
print_header "EVALUATION COMPLETE!"

echo "  Files updated:"
echo "    ✓ $LOCAL_EVAL_OUTPUT/eval_results.json"
echo "    ✓ $LOCAL_RESULTS/overall_performance.png"
echo "    ✓ $LOCAL_RESULTS/category_retention.png"
echo "    ✓ $LOCAL_RESULTS/graph*.png"
echo ""

echo "  Quick view:"
echo "    cat $LOCAL_EVAL_OUTPUT/eval_results.json"
echo ""

# Print key numbers if jq is available
if command -v jq &> /dev/null; then
    echo "  Key results:"
    jq -r '.methods | to_entries[] | "    \(.key): old=\(.value.old_accuracy // "N/A"), new=\(.value.new_accuracy // "N/A"), forgetting=\(.value.forgetting_score // "N/A")"' "$LOCAL_EVAL_OUTPUT/eval_results.json"
    echo ""
fi

echo -e "${GREEN}${BOLD}  ✅ Real GPU evaluation finished!${NC}"
echo ""
