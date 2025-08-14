#!/usr/bin/env python3
"""
Fine-tune Qwen 2.5 1.5B on Khmer Medical Dataset
Optimized for JarvisLabs GPU instances
"""

import os
import torch
import argparse
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 on Khmer Medical QA")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B", help="Base model ID")
    parser.add_argument("--dataset_path", default="data/training_formats", help="Path to training data")
    parser.add_argument("--output_dir", default="models/khmer-medical-qwen", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", default="khmer-medical-qwen", help="HF Hub model ID")
    parser.add_argument("--wandb_project", default="khmer-medical", help="Weights & Biases project")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    return parser.parse_args()

def setup_model(args):
    """Load and configure the model with LoRA"""
    print(f"🔄 Loading model: {args.model_id}")
    
    # Configure quantization if needed
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ 4-bit quantization enabled")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attn
    )
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Prepare model for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def load_tokenizer(args):
    """Load and configure tokenizer"""
    print(f"🔄 Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def prepare_dataset(args, tokenizer):
    """Load and prepare the dataset"""
    print(f"📚 Loading dataset from: {args.dataset_path}")
    
    # Load dataset
    dataset = load_dataset('json', 
        data_files={
            'train': f'{args.dataset_path}/qwen_train.jsonl',
            'validation': f'{args.dataset_path}/qwen_val.jsonl'
        }
    )
    
    def format_chat(example):
        """Format for Qwen chat template"""
        text = f"""<|im_start|>system
អ្នកគឺជាជំនួយការវេជ្ជសាស្ត្រដែលមានចំណេះដឹង។ សូមឆ្លើយសំណួរវេជ្ជសាស្ត្រដោយផ្តល់ព័ត៌មានត្រឹមត្រូវ និងមានប្រយោជន៍។<|im_end|>
<|im_start|>user
{example.get('instruction', example.get('question', ''))}<|im_end|>
<|im_start|>assistant
{example.get('output', example.get('response', ''))}<|im_end|>"""
        return {"text": text}
    
    # Format dataset
    dataset = dataset.map(format_chat, remove_columns=dataset['train'].column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=2048,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    print(f"✅ Dataset prepared: {len(tokenized_dataset['train'])} train, {len(tokenized_dataset['validation'])} val")
    return tokenized_dataset

def setup_training_args(args):
    """Configure training arguments"""
    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8 if args.batch_size < 4 else 4,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_ratio=0.1,
        learning_rate=args.learning_rate,
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to="wandb" if args.wandb_project else "none",
        run_name=f"qwen-khmer-medical-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

def main():
    args = parse_args()
    
    # Initialize wandb if needed
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"qwen-khmer-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
    
    # Setup
    tokenizer = load_tokenizer(args)
    model = setup_model(args)
    dataset = prepare_dataset(args, tokenizer)
    training_args = setup_training_args(args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        ),
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print(f"💾 Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"📤 Pushing to HuggingFace Hub: {args.hub_model_id}")
        trainer.push_to_hub()
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()