#!/usr/bin/env python3
"""
SG-CL Training Script
=====================

Example usage:
    python run_training.py --data data/train.txt --epochs 3 --gating

This script demonstrates the full SG-CL training workflow.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def check_gpu():
    """Verify CUDA GPU is available before proceeding."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\033[91m✗ ERROR: No CUDA GPU detected!\033[0m")
            print("  This project requires an NVIDIA GPU with CUDA support.")
            print("  → Connect to ARC Labs: ssh arcgpu")
            print("  → Or run: python3 check_gpu_connection.py")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"\033[92m✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)\033[0m")
    except ImportError:
        print("\033[91m✗ PyTorch not installed!\033[0m")
        sys.exit(1)

from training.sgcl_trainer import SGCLConfig, SGCLTrainer, SGCLPipelineDemo


def load_claims(filepath: str) -> list:
    """Load claims from a text file (one per line)."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="SG-CL Training Script")
    
    # Data arguments
    parser.add_argument("--data", type=str, help="Path to training data (one claim per line)")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="./models/llama-2-7b-hf",
                       help="Path to base model")
    parser.add_argument("--output", type=str, default="./outputs",
                       help="Output directory for adapters")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (4 for RTX 4090, 1 for ≤8GB GPUs)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    
    # SG-CL arguments
    parser.add_argument("--gating", action="store_true", help="Enable symbolic gating")
    parser.add_argument("--no-gating", action="store_true", help="Disable symbolic gating")
    
    # Mode arguments
    parser.add_argument("--demo", action="store_true", help="Run demo mode only (no training)")
    parser.add_argument("--task-name", type=str, default="task_1", help="Task name for output")
    
    args = parser.parse_args()
    
    # Always check GPU first (unless demo mode)
    if not args.demo:
        check_gpu()
    
    # Demo mode
    if args.demo:
        print("Running in demo mode...\n")
        demo = SGCLPipelineDemo()
        
        if args.data:
            claims = load_claims(args.data)
        else:
            # Default demo claims
            claims = [
                "Penguins can fly.",
                "Penguins can swim.",
                "Dogs can fly.",
                "Cats can climb.",
                "Fish can walk.",
                "Birds can fly.",
            ]
        
        demo.demonstrate(claims)
        return
    
    # Training mode requires data
    if not args.data:
        print("Error: --data is required for training mode")
        print("Use --demo for demonstration without training data")
        sys.exit(1)
    
    # Load data
    train_claims = load_claims(args.data)
    eval_claims = load_claims(args.eval_data) if args.eval_data else None
    
    print(f"Loaded {len(train_claims)} training claims")
    if eval_claims:
        print(f"Loaded {len(eval_claims)} evaluation claims")
    
    # Configure
    config = SGCLConfig(
        model_path=args.model,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_length,
        enable_gating=not args.no_gating if args.no_gating else args.gating,
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_path}")
    print(f"  LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"  Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    print(f"  Gating enabled: {config.enable_gating}")
    
    # Create trainer
    trainer = SGCLTrainer(config)
    
    # Setup model
    print("\nSetting up model...")
    trainer.setup()
    
    # Train
    print("\nStarting training...")
    results = trainer.train(
        train_claims=train_claims,
        eval_claims=eval_claims,
        task_name=args.task_name
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {results['train_loss']:.4f}")
    print(f"Training time: {results['train_runtime']:.2f}s")
    print(f"Adapter saved to: {os.path.join(config.output_dir, args.task_name, 'adapter')}")
    
    # Save results
    results_path = os.path.join(config.output_dir, args.task_name, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
