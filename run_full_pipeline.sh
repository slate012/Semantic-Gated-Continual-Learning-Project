#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SG-CL Full Training & Evaluation Pipeline
# Optimized for NVIDIA RTX 4090 (24 GB VRAM)
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh
#
# This script runs the ENTIRE SG-CL project end-to-end:
#   1. Checks environment & GPU
#   2. Installs dependencies
#   3. Generates the synthetic dataset
#   4. Trains across 5 progressive tasks (with gating + 4-bit quant)
#   5. Runs evaluation on the final adapter
#   6. Prints all results
# ═══════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Optimized for RTX 4090 (24GB VRAM)
# ══════════════════════════════════════════════════════════════════════════
MODEL_PATH="./models/llama-2-7b-hf"
OUTPUT_DIR="./outputs"
DATA_DIR="./data"
NUM_TASKS=5
EPOCHS=3
BATCH_SIZE=4                # RTX 4090 BF16 (safe for 24GB)
LEARNING_RATE=2e-4
MAX_LENGTH=512              # Full sequence length (24GB can handle it)
LORA_R=16
LORA_ALPHA=32
GRAD_ACCUM=4                # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16

# ══════════════════════════════════════════════════════════════════════════
# STEP 0: Pre-flight Checks
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 0: Pre-flight Checks"

# Check Python
print_step "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    print_error "Python not found! Please install Python 3.8+."
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# Check GPU
print_step "Checking GPU..."
$PYTHON -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  ✓ GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)')
    if gpu_mem < 16:
        print(f'  ⚠ WARNING: {gpu_mem:.1f} GB VRAM. RTX 4090 (24GB) recommended.')
        print(f'  ℹ Reducing batch_size to 1 and max_length to 256...')
else:
    print('  ✗ No CUDA GPU detected! Training will fail.')
    exit(1)
" 2>/dev/null || {
    print_warn "PyTorch not installed yet. Will install in Step 1."
}

# Check model
print_step "Checking LLaMA-2-7B model..."
if [ -f "$MODEL_PATH/config.json" ]; then
    echo -e "  ${GREEN}✓ Model found at $MODEL_PATH${NC}"
else
    print_error "Model NOT found at $MODEL_PATH"
    echo "  Please download it first:"
    echo "    huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir $MODEL_PATH"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Install Dependencies
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 1: Installing Dependencies"

print_step "Installing Python packages..."
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet torch torchvision
$PYTHON -m pip install --quiet transformers peft bitsandbytes accelerate datasets
$PYTHON -m pip install --quiet requests huggingface_hub
$PYTHON -m pip install --quiet matplotlib numpy jupyter
echo -e "  ${GREEN}✓ All dependencies installed${NC}"

# Verify critical imports
print_step "Verifying imports..."
$PYTHON -c "
import torch, transformers, peft, bitsandbytes, accelerate
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  bitsandbytes:   {bitsandbytes.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  VRAM:           {mem:.1f} GB')
"

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Generate Synthetic Dataset
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 2: Generating Synthetic Dataset"

print_step "Running build_sgcl_dataset.py..."
$PYTHON build_sgcl_dataset.py

# Verify files exist
for i in $(seq 1 $NUM_TASKS); do
    if [ -f "$DATA_DIR/train_task_${i}.txt" ]; then
        lines=$(wc -l < "$DATA_DIR/train_task_${i}.txt")
        echo "  ✓ train_task_${i}.txt ($lines lines)"
    else
        print_error "train_task_${i}.txt not found!"
        exit 1
    fi
done

if [ -f "$DATA_DIR/evaluation_set.json" ]; then
    echo -e "  ${GREEN}✓ evaluation_set.json exists${NC}"
else
    print_error "evaluation_set.json not found!"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: VRAM Diagnostic
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 3: VRAM Configuration"

echo "  Settings optimized for RTX 4090 (24 GB VRAM):"
echo "    Batch size:                $BATCH_SIZE"
echo "    Gradient accumulation:     $GRAD_ACCUM (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "    Max sequence length:       $MAX_LENGTH"
echo "    Precision:                 BF16 (native, no quantization)"
echo "    TF32 matmul:               Enabled (Ada Lovelace tensor cores)"
echo "    Flash Attention:           Enabled"
echo "    LoRA rank:                 $LORA_R"
echo ""
echo "  Estimated VRAM usage:"
echo "    Model (BF16):              ~13.5 GB"
echo "    LoRA + Optimizer:          ~1.5 GB"
echo "    Activations (batch=4):     ~3.0 GB"
echo "    Total:                     ~18.0 GB / 24.0 GB"

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Smoke Test (1 Epoch, Task 1)
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 4: Smoke Test (1 Epoch)"

print_step "Running 1 epoch on Task 1 to verify everything fits in VRAM..."

$PYTHON run_training.py \
    --data "$DATA_DIR/train_task_1.txt" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --task-name "smoke_test" \
    --gating \
    --epochs 1 \
    --batch-size $BATCH_SIZE \
    --max-length $MAX_LENGTH \
    --lr $LEARNING_RATE \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA

echo -e "  ${GREEN}✓ Smoke test PASSED! Your GPU can handle the training.${NC}"

# Clean up smoke test outputs
rm -rf "$OUTPUT_DIR/smoke_test"

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Full Training (5 Tasks × 3 Epochs)
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 5: Full Training (5 Tasks × $EPOCHS Epochs)"

TOTAL_START=$(date +%s)

for TASK_NUM in $(seq 1 $NUM_TASKS); do
    echo ""
    print_step "═══ Task $TASK_NUM / $NUM_TASKS ═══"
    
    TASK_START=$(date +%s)
    
    $PYTHON run_training.py \
        --data "$DATA_DIR/train_task_${TASK_NUM}.txt" \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR" \
        --task-name "task_${TASK_NUM}" \
        --gating \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --max-length $MAX_LENGTH \
        --lr $LEARNING_RATE \
        --lora-r $LORA_R \
        --lora-alpha $LORA_ALPHA
    
    TASK_END=$(date +%s)
    TASK_DURATION=$((TASK_END - TASK_START))
    TASK_MINS=$((TASK_DURATION / 60))
    echo -e "  ${GREEN}✓ Task $TASK_NUM completed in ${TASK_MINS}m ${TASK_DURATION}s${NC}"
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINS=$((TOTAL_DURATION / 60))
echo ""
echo -e "${GREEN}═══ All $NUM_TASKS tasks completed in ${TOTAL_MINS} minutes ═══${NC}"

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Evaluate the Final Model
# ══════════════════════════════════════════════════════════════════════════
print_header "STEP 6: Running Evaluation"

FINAL_ADAPTER="$OUTPUT_DIR/task_${NUM_TASKS}/adapter"

if [ -d "$FINAL_ADAPTER" ]; then
    print_step "Evaluating with final adapter from task $NUM_TASKS..."
    
    $PYTHON evaluate_model.py \
        --model "$MODEL_PATH" \
        --adapter "$FINAL_ADAPTER" \
        --eval-data "$DATA_DIR/evaluation_set.json" \
        --output "$OUTPUT_DIR" \
        --compare \
        --verbose
else
    print_warn "No adapter found at $FINAL_ADAPTER. Running demo evaluation..."
    $PYTHON evaluate_model.py --demo --verbose
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Summary
# ══════════════════════════════════════════════════════════════════════════
print_header "PIPELINE COMPLETE!"

echo "  Total training time: ${TOTAL_MINS} minutes"
echo ""
echo "  Files generated:"
for TASK_NUM in $(seq 1 $NUM_TASKS); do
    RESULT_FILE="$OUTPUT_DIR/task_${TASK_NUM}/results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "    ✓ $RESULT_FILE"
    fi
done
if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
    echo "    ✓ $OUTPUT_DIR/eval_results.json"
fi

echo ""
echo "  Next steps:"
echo "    1. View the notebook:  jupyter notebook notebooks/sgcl_analysis.ipynb"
echo "    2. Check eval results: cat $OUTPUT_DIR/eval_results.json"
echo ""
echo -e "${GREEN}${BOLD}  🎉 SG-CL Capstone Project — Training Complete!${NC}"
echo ""
