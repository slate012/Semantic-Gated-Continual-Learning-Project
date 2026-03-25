"""
SG-CL Training Pipeline with LoRA
==================================

This module implements the continual learning training pipeline for SG-CL.

Key Features:
1. Frozen backbone (LLaMA-2-7B) with LoRA adapters
2. Symbolic gating via SID integration
3. Guard-rail augmented training batches
4. Sequential task training without replay

Training Philosophy:
- Base model weights are NEVER modified
- All updates happen through LoRA adapters
- Conflicting data triggers gated training with guard-rails
- Safe data goes through normal fine-tuning
"""

import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Transformers and PEFT imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# SG-CL components
from sid.semantic_inconsistency_detector import SemanticInconsistencyDetector, create_sid
from guardrail.guardrail_generator import (
    GuardRailGenerator, 
    GatedBatchConstructor,
    create_generator,
    create_batch_constructor
)
from utils.conceptnet_client import create_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SGCLConfig:
    """Configuration for SG-CL training.
    
    Optimized for NVIDIA RTX 4090 (24GB VRAM, Ada Lovelace, Compute 8.9):
    - BF16 precision (better numerical stability than FP16 on Ada)
    - TF32 matmul for faster tensor core operations
    - No 4-bit quantization needed (24GB fits full BF16 model)
    - Batch size 8 with grad accum 2 = effective batch 16
    """
    
    # Model settings
    model_path: str = "./models/llama-2-7b-hf"
    output_dir: str = "./outputs"
    
    # LoRA settings
    lora_r: int = 16                    # LoRA rank
    lora_alpha: int = 32                # LoRA alpha (scaling)
    lora_dropout: float = 0.05         # LoRA dropout
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ])
    
    # Training settings — optimized for RTX 4090 (24GB)
    learning_rate: float = 2e-4
    batch_size: int = 8                 # RTX 4090 can handle batch=8 in BF16
    gradient_accumulation_steps: int = 2  # Effective batch = 8 * 2 = 16
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    load_in_4bit: bool = False             # Not needed for 24GB VRAM — use native BF16
    use_bf16: bool = True                  # BF16 for Ada Lovelace (better than FP16)
    enable_tf32: bool = True               # TF32 matmul for faster tensor cores
    dataloader_num_workers: int = 4        # i9-14900 has plenty of CPU threads
    
    # SG-CL specific settings
    enable_gating: bool = True          # Enable symbolic gating
    guard_rail_weight: float = 1.0      # Weight for guard-rail samples
    max_guard_rails: int = 5            # Max guard-rails per conflict
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }


class SGCLDataset(Dataset):
    """
    Dataset for SG-CL training.
    
    Handles both safe and gated samples with proper weighting.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of training texts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            weights: Optional importance weights for each sample
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = weights or [1.0] * len(texts)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM, labels = input_ids
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }
        
        # Add weight if needed
        if self.weights:
            item["weight"] = torch.tensor(self.weights[idx])
        
        return item


class SGCLTrainer:
    """
    Main SG-CL Trainer Class
    
    Orchestrates the full training pipeline:
    1. Load and prepare model with LoRA
    2. Process data through SID and Guard-Rail Generator
    3. Construct gated training batches
    4. Train with symbolic awareness
    """
    
    def __init__(self, config: SGCLConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        # Force CUDA — this project is designed for GPU only
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            logger.error("No CUDA GPU detected! This project requires an NVIDIA GPU.")
            logger.error("Connect to ARC Labs: ssh arcgpu")
            raise RuntimeError("CUDA GPU required. No GPU detected.")
        
        logger.info(f"Initializing SG-CL Trainer on device: {self.device}")
        
        # Initialize SG-CL components
        self.conceptnet = create_client()
        self.sid = create_sid(self.conceptnet)
        self.generator = create_generator(self.conceptnet)
        self.batch_constructor = create_batch_constructor(self.sid, self.generator)
        
        # Will be initialized in setup()
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def setup(self):
        """
        Setup model and tokenizer with LoRA.
        """
        logger.info(f"Loading model from: {self.config.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Enable TF32 for faster tensor core matmul on Ada Lovelace (RTX 4090)
        if self.config.enable_tf32 and self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for faster tensor core operations")
        
        # Load base model
        logger.info("Loading base model (this may take a while)...")
        
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if self.config.load_in_4bit and self.device == "cuda":
            # 4-bit quantization — for GPUs with <16GB VRAM
            logger.info("Using 4-bit quantization (NF4) to fit in low-VRAM GPU...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            # Native precision — BF16 for Ada Lovelace (RTX 4090), FP16 fallback
            if self.config.use_bf16 and torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("Using BF16 precision (optimal for Ada Lovelace / RTX 4090)")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Using FP16 precision")
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
        
        # Try Flash Attention 2, fall back to SDPA (PyTorch native), then default
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled for memory-efficient attention")
        except ImportError:
            # SDPA (Scaled Dot Product Attention) is built into PyTorch 2.0+
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Using PyTorch SDPA attention (flash_attn not installed)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        
        # Prepare model for quantized training if using 4-bit
        if self.config.load_in_4bit and self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        logger.info("Configuring LoRA adapters...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        
        return self
    
    def prepare_data(
        self, 
        claims: List[str],
        apply_gating: bool = True
    ) -> Tuple[SGCLDataset, Dict]:
        """
        Prepare training data with optional symbolic gating.
        
        Args:
            claims: List of training claims
            apply_gating: Whether to apply SG-CL gating
            
        Returns:
            (dataset, statistics)
        """
        if not apply_gating or not self.config.enable_gating:
            # Standard training without gating
            logger.info("Preparing data without gating (standard fine-tuning)")
            dataset = SGCLDataset(
                texts=claims,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length
            )
            return dataset, {"mode": "standard", "total": len(claims)}
        
        # Apply SG-CL gating
        logger.info("Applying SG-CL gating to training data...")
        
        batch_result = self.batch_constructor.construct_batch(
            claims, 
            include_weights=True
        )
        
        all_texts = batch_result["all_texts"]
        weights = batch_result.get("weights", [1.0] * len(all_texts))
        
        logger.info(f"Gating results:")
        logger.info(f"  Safe claims: {batch_result['stats']['safe_claims']}")
        logger.info(f"  Gated claims: {batch_result['stats']['gated_claims']}")
        logger.info(f"  Total guard-rails: {batch_result['stats']['total_guard_rails']}")
        logger.info(f"  Total training samples: {len(all_texts)}")
        
        dataset = SGCLDataset(
            texts=all_texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            weights=weights
        )
        
        return dataset, batch_result["stats"]
    
    def train(
        self,
        train_claims: List[str],
        eval_claims: Optional[List[str]] = None,
        task_name: str = "task_1"
    ) -> Dict:
        """
        Train on a set of claims.
        
        Args:
            train_claims: Training claims
            eval_claims: Optional evaluation claims
            task_name: Name of this training task
            
        Returns:
            Training results dictionary
        """
        if self.peft_model is None:
            raise RuntimeError("Call setup() before training")
        
        # Prepare output directory
        output_dir = os.path.join(self.config.output_dir, task_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare training data
        train_dataset, train_stats = self.prepare_data(train_claims)
        
        # Prepare eval data if provided
        eval_dataset = None
        if eval_claims:
            eval_dataset, _ = self.prepare_data(eval_claims, apply_gating=False)
        
        # Determine precision flags
        use_bf16 = self.config.use_bf16 and self.device == "cuda" and torch.cuda.is_bf16_supported()
        use_fp16 = (not use_bf16) and self.device == "cuda"
        
        # Training arguments — optimized for RTX 4090
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            bf16=use_bf16,                # BF16 for Ada Lovelace (RTX 4090)
            fp16=use_fp16,                # FP16 fallback for older GPUs
            tf32=self.config.enable_tf32,  # TF32 matmul acceleration
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,    # Faster CPU→GPU transfer
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info(f"Starting training for task: {task_name}")
        train_result = trainer.train()
        
        # Save adapter
        adapter_path = os.path.join(output_dir, "adapter")
        self.peft_model.save_pretrained(adapter_path)
        logger.info(f"LoRA adapter saved to: {adapter_path}")
        
        # Save training stats
        stats = {
            "task_name": task_name,
            "train_stats": train_stats,
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def train_continual(
        self,
        task_sequence: List[Tuple[str, List[str]]],
        eval_claims: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Continual learning across a sequence of tasks.
        
        This is the core SG-CL training mode:
        - Tasks are processed sequentially
        - No replay buffer
        - Symbolic gating prevents catastrophic forgetting
        
        Args:
            task_sequence: List of (task_name, claims) tuples
            eval_claims: Optional held-out evaluation claims
            
        Returns:
            List of training results for each task
        """
        if self.peft_model is None:
            raise RuntimeError("Call setup() before training")
        
        results = []
        
        for i, (task_name, claims) in enumerate(task_sequence):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {i+1}/{len(task_sequence)}: {task_name}")
            logger.info(f"{'='*60}")
            
            task_result = self.train(
                train_claims=claims,
                eval_claims=eval_claims,
                task_name=task_name
            )
            
            results.append(task_result)
            
            logger.info(f"Task {task_name} completed. Loss: {task_result['train_loss']:.4f}")
        
        # Save final combined adapter
        final_path = os.path.join(self.config.output_dir, "final_adapter")
        self.peft_model.save_pretrained(final_path)
        logger.info(f"\nFinal adapter saved to: {final_path}")
        
        return results
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        if self.peft_model is None:
            raise RuntimeError("Call setup() before generation")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated


class SGCLPipelineDemo:
    """
    Demonstration of the full SG-CL pipeline without actual model training.
    
    Useful for testing the data flow and gating logic.
    """
    
    def __init__(self):
        self.conceptnet = create_client()
        self.sid = create_sid(self.conceptnet)
        self.generator = create_generator(self.conceptnet)
        self.batch_constructor = create_batch_constructor(self.sid, self.generator)
    
    def demonstrate(self, claims: List[str]) -> Dict:
        """
        Demonstrate the full pipeline without training.
        
        Args:
            claims: List of claims to process
            
        Returns:
            Pipeline results
        """
        print("=" * 70)
        print("SG-CL Pipeline Demonstration")
        print("=" * 70)
        
        # Step 1: Analyze with SID
        print("\n[Step 1] SID Analysis")
        print("-" * 40)
        
        classifications = []
        for claim in claims:
            result = self.sid.analyze(claim)
            if result:
                classifications.append(result)
                print(f"  Claim: '{claim}'")
                print(f"  → Triple: {result.claim.to_triple()}")
                print(f"  → Conflict: {result.conflict_result.has_conflict}")
                print(f"  → Gating: {result.gating_decision}")
                print()
        
        # Step 2: Generate guard-rails for conflicts
        print("\n[Step 2] Guard-Rail Generation")
        print("-" * 40)
        
        for cls in classifications:
            if cls.conflict_result.has_conflict:
                batch = self.generator.generate_from_classification(cls)
                print(f"  Claim: '{cls.claim.source_text}'")
                print(f"  Guard-rails ({len(batch.guard_rails)}):")
                for gr in batch.guard_rails[:3]:
                    print(f"    - [{gr.rail_type.value}] {gr.text}")
                print()
        
        # Step 3: Construct training batch
        print("\n[Step 3] Training Batch Construction")
        print("-" * 40)
        
        result = self.batch_constructor.construct_batch(claims, include_weights=True)
        
        print(f"  Statistics:")
        for key, value in result["stats"].items():
            print(f"    {key}: {value}")
        
        print(f"\n  Safe texts ({len(result['safe_texts'])}):")
        for text in result["safe_texts"][:3]:
            print(f"    ✓ {text}")
        
        print(f"\n  Gated texts ({len(result['gated_texts'])}):")
        for i, text in enumerate(result["gated_texts"][:5]):
            weight = result["weights"][len(result["safe_texts"]) + i]
            print(f"    ⚠ [{weight:.2f}] {text}")
        
        if len(result["gated_texts"]) > 5:
            print(f"    ... and {len(result['gated_texts']) - 5} more")
        
        print("\n" + "=" * 70)
        print("Pipeline demonstration complete!")
        print("=" * 70)
        
        return result


# =============================================================================
# Factory Functions
# =============================================================================

def create_config(**kwargs) -> SGCLConfig:
    """Create SG-CL configuration with optional overrides."""
    return SGCLConfig(**kwargs)


def create_trainer(config: Optional[SGCLConfig] = None) -> SGCLTrainer:
    """Create SG-CL trainer with optional configuration."""
    if config is None:
        config = SGCLConfig()
    return SGCLTrainer(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SG-CL Training Pipeline - Test Suite")
    print("=" * 70)
    
    # Test 1: Pipeline demonstration (no actual training)
    print("\n1. Testing pipeline demonstration:\n")
    
    demo = SGCLPipelineDemo()
    
    test_claims = [
        "Penguins can fly.",
        "Penguins can swim.",
        "Dogs can fly.",
        "Cats can climb.",
        "Fish can walk.",
        "Birds can fly.",
        "Whales can swim.",
        "Humans can think.",
    ]
    
    result = demo.demonstrate(test_claims)
    
    # Test 2: Configuration
    print("\n2. Testing configuration:\n")
    
    config = create_config(
        model_path="./models/llama-2-7b-hf",
        lora_r=16,
        lora_alpha=32,
        batch_size=4,
        num_epochs=3
    )
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        if not isinstance(value, list):
            print(f"  {key}: {value}")
    
    # Test 3: Trainer initialization (without model loading)
    print("\n3. Testing trainer initialization:\n")
    
    trainer = create_trainer(config)
    print(f"  Trainer created with device: {trainer.device}")
    print(f"  SID initialized: {trainer.sid is not None}")
    print(f"  Generator initialized: {trainer.generator is not None}")
    
    print("\n" + "=" * 70)
    print("Training pipeline tests completed!")
    print("=" * 70)
    print("\nNote: To run actual training, call:")
    print("  trainer.setup()  # Load model and configure LoRA")
    print("  trainer.train(claims)  # Train on claims")
