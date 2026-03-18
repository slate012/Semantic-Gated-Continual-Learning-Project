# SG-CL: Symbolic-Gated Continual Learning Framework

A novel framework for continual learning that prevents catastrophic forgetting by integrating symbolic knowledge constraints into the training process.

## Quick Start

```bash
# 1. Clone/Copy the project folder to your computer

# 2. Navigate to project
cd "capstone project"

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows

# 5. Install dependencies
pip install torch transformers accelerate peft bitsandbytes requests huggingface_hub

# 6. Run the demo
python demo_sgcl.py
```

## Complete Setup Instructions

### Prerequisites
- Python 3.10 or higher
- ~30GB disk space (for LLaMA-2-7B model)
- 16GB+ RAM recommended

### Step 1: Install Dependencies
```bash
pip install torch transformers accelerate peft bitsandbytes requests huggingface_hub
```

### Step 2: Download LLaMA-2-7B Model (Optional - for training)

You need a Hugging Face account with access to `meta-llama/Llama-2-7b-hf`.

```bash
# Login to Hugging Face
huggingface-cli login

# Download model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-2-7b-hf', local_dir='models/llama-2-7b-hf')
"
```

**Note:** The demo works without the model. Training requires the model.

### Step 3: Verify Installation
```bash
python demo_sgcl.py
```

## Project Structure

```
capstone project/
├── models/llama-2-7b-hf/     # LLaMA-2-7B model (download separately)
├── src/
│   ├── utils/                 # ConceptNet client + local knowledge base
│   ├── sid/                   # Semantic Inconsistency Detector
│   ├── guardrail/             # Guard-Rail Generator
│   └── training/              # LoRA training pipeline
├── data/raw/                  # Sample training data
├── config/                    # Configuration files
├── demo_sgcl.py               # Interactive demo
├── run_training.py            # Training script
└── requirements.txt           # Dependencies
```

## Usage

### Run Demo (No Model Required)
```bash
python demo_sgcl.py
```

### Run Training (Requires Model)
```bash
# With SG-CL gating
python run_training.py --data data/raw/sample_conflicts.txt --gating --epochs 3

# Standard fine-tuning (no gating)
python run_training.py --data data/raw/sample_safe.txt --no-gating
```

## Key Components

| Component | Description |
|-----------|-------------|
| **ConceptNet Client** | Queries knowledge base for semantic relations |
| **SID** | Detects conflicts between new data and existing knowledge |
| **Guard-Rail Generator** | Creates constraint-preserving training statements |
| **LoRA Trainer** | Fine-tunes model with frozen backbone + adapters |

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade torch transformers peft
```

**Permission denied?**
```bash
chmod +x demo_sgcl.py run_training.py
```

**Model not found?**
The demo works without the model. For training, download LLaMA-2-7B first.

## Citation

```bibtex
@thesis{sgcl2026,
  title={Symbolic-Gated Continual Learning},
  author={[Your Name]},
  year={2026}
}
```
