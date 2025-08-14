#!/usr/bin/env python3
"""
Fine-tune SmolLM2 1.7B on Khmer Medical Dataset
Ultra-efficient training for low-resource settings
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
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM2 on Khmer Medical QA")
    parser.add_argument("--model_id", default="HuggingFaceTB/SmolLM2-1.7B", help="Base model ID")
    parser.add_argument("--dataset_path", default="data/training_formats", help="Path to training data")
    parser.add_argument("--output_dir", default="models/khmer-medical-smollm", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--max_length", type=int, default=1536, help="Maximum sequence length")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", default="khmer-medical-smollm", help="HF Hub model ID")
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
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Prepare model for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
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
            'train': f'{args.dataset_path}/chatml_train.jsonl',
            'validation': f'{args.dataset_path}/chatml_val.jsonl'
        }
    )
    
    def format_chat(example):
        """Format for ChatML template"""
        messages = example.get('messages', [])
        if messages:
            text = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == 'assistant':
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        else:
            # Fallback for other formats
            text = f"<|im_start|>user\n{example.get('instruction', '')}<|im_end|>\n"
            text += f"<|im_start|>assistant\n{example.get('output', '')}<|im_end|>"
        
        return {"text": text}
    
    # Format dataset
    dataset = dataset.map(format_chat, remove_columns=dataset['train'].column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
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
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
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
        optim="adamw_torch",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to="none",
        run_name=f"smollm-khmer-medical-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

def main():
    args = parse_args()
    
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