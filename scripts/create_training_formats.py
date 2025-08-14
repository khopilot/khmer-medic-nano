#!/usr/bin/env python3
"""
Create optimized training formats for LLM fine-tuning
Supports multiple instruction formats commonly used in fine-tuning
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import random
from tqdm import tqdm

def create_alpaca_format(data: List[Dict]) -> List[Dict]:
    """
    Create Alpaca/Stanford format for instruction tuning
    """
    formatted = []
    for item in data:
        # Standard format
        entry = {
            "instruction": item.get("question_km", ""),
            "input": "",  # No additional context needed
            "output": item.get("response_km", "")
        }
        formatted.append(entry)
        
        # If paraphrase exists, create another entry
        if item.get("question_km_para") and item.get("response_km_para"):
            para_entry = {
                "instruction": item["question_km_para"],
                "input": "",
                "output": item["response_km_para"]
            }
            formatted.append(para_entry)
    
    return formatted

def create_chatml_format(data: List[Dict]) -> List[Dict]:
    """
    Create ChatML format (used by many modern models)
    """
    formatted = []
    for item in data:
        conversation = {
            "messages": [
                {"role": "user", "content": item.get("question_km", "")},
                {"role": "assistant", "content": item.get("response_km", "")}
            ]
        }
        formatted.append(conversation)
        
        # Add paraphrase version if available
        if item.get("question_km_para") and item.get("response_km_para"):
            para_conversation = {
                "messages": [
                    {"role": "user", "content": item["question_km_para"]},
                    {"role": "assistant", "content": item["response_km_para"]}
                ]
            }
            formatted.append(para_conversation)
    
    return formatted

def create_llama_format(data: List[Dict]) -> List[str]:
    """
    Create Llama-style format with special tokens
    """
    formatted = []
    
    system_prompt = "អ្នកគឺជាជំនួយការវេជ្ជសាស្ត្រដែលមានចំណេះដឹង។ សូមឆ្លើយសំណួរវេជ្ជសាស្ត្រដោយផ្តល់ព័ត៌មានត្រឹមត្រូវ និងមានប្រយោជន៍។"
    
    for item in data:
        # Format: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{item.get('question_km', '')} [/INST] {item.get('response_km', '')} </s>"
        formatted.append(prompt)
        
        # Add paraphrase version
        if item.get("question_km_para") and item.get("response_km_para"):
            para_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{item['question_km_para']} [/INST] {item['response_km_para']} </s>"
            formatted.append(para_prompt)
    
    return formatted

def create_qwen_format(data: List[Dict]) -> List[str]:
    """
    Create Qwen format for fine-tuning
    """
    formatted = []
    
    for item in data:
        # Qwen uses: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
        prompt = f"<|im_start|>system\nអ្នកគឺជាជំនួយការវេជ្ជសាស្ត្រ។<|im_end|>\n<|im_start|>user\n{item.get('question_km', '')}<|im_end|>\n<|im_start|>assistant\n{item.get('response_km', '')}<|im_end|>"
        formatted.append(prompt)
        
        # Add paraphrase
        if item.get("question_km_para") and item.get("response_km_para"):
            para_prompt = f"<|im_start|>system\nអ្នកគឺជាជំនួយការវេជ្ជសាស្ត្រ។<|im_end|>\n<|im_start|>user\n{item['question_km_para']}<|im_end|>\n<|im_start|>assistant\n{item['response_km_para']}<|im_end|>"
            formatted.append(para_prompt)
    
    return formatted

def create_supervised_finetuning_format(data: List[Dict]) -> List[Dict]:
    """
    Create format optimized for supervised fine-tuning with reasoning
    """
    formatted = []
    
    for item in data:
        # Include reasoning summary for better learning
        entry = {
            "id": f"med_{item.get('index', 0)}",
            "question": item.get("question_km", ""),
            "answer": item.get("response_km", ""),
            "reasoning": item.get("reasoning_summary_km", ""),
            "tags": item.get("tags", []),
            "metadata": {
                "has_paraphrase": bool(item.get("question_km_para")),
                "source": "medical-o1-reasoning-SFT"
            }
        }
        formatted.append(entry)
        
        # Create augmented version with paraphrase
        if item.get("question_km_para") and item.get("response_km_para"):
            para_entry = {
                "id": f"med_{item.get('index', 0)}_para",
                "question": item["question_km_para"],
                "answer": item["response_km_para"],
                "reasoning": item.get("reasoning_summary_km", ""),
                "tags": item.get("tags", []),
                "metadata": {
                    "is_paraphrase": True,
                    "original_id": f"med_{item.get('index', 0)}",
                    "source": "medical-o1-reasoning-SFT"
                }
            }
            formatted.append(para_entry)
    
    return formatted

def split_train_val(data: List[Any], val_ratio: float = 0.05) -> tuple:
    """
    Split data into training and validation sets
    """
    random.seed(42)  # For reproducibility
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    val_data = shuffled[:val_size]
    train_data = shuffled[val_size:]
    
    return train_data, val_data

def main():
    # Load the dataset
    input_file = Path("data/out/km_final.jsonl")
    output_dir = Path("data/training_formats")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("📚 Creating optimized training formats for LLM fine-tuning...")
    
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} entries")
    
    # Create different formats
    formats = {
        "alpaca": create_alpaca_format,
        "chatml": create_chatml_format,
        "llama": create_llama_format,
        "qwen": create_qwen_format,
        "supervised": create_supervised_finetuning_format
    }
    
    stats = {}
    
    for format_name, format_func in formats.items():
        print(f"\n🔄 Creating {format_name} format...")
        formatted_data = format_func(data)
        
        # Split into train/val
        train_data, val_data = split_train_val(formatted_data)
        
        # Save files
        if format_name in ["llama", "qwen"]:
            # Save as text files for these formats
            with open(output_dir / f"{format_name}_train.txt", 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(item + "\n")
            
            with open(output_dir / f"{format_name}_val.txt", 'w', encoding='utf-8') as f:
                for item in val_data:
                    f.write(item + "\n")
        else:
            # Save as JSONL for structured formats
            with open(output_dir / f"{format_name}_train.jsonl", 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            with open(output_dir / f"{format_name}_val.jsonl", 'w', encoding='utf-8') as f:
                for item in val_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        stats[format_name] = {
            "train": len(train_data),
            "val": len(val_data),
            "total": len(formatted_data)
        }
        
        print(f"  ✅ {format_name}: {len(train_data)} train, {len(val_data)} val")
    
    # Create configuration file for training
    config = {
        "dataset_info": {
            "name": "khmer-medical-qa",
            "version": "1.0",
            "language": "km",
            "domain": "medical",
            "total_examples": len(data),
            "with_paraphrases": sum(1 for d in data if d.get("question_km_para")),
            "with_reasoning": sum(1 for d in data if d.get("reasoning_summary_km"))
        },
        "formats_available": list(formats.keys()),
        "statistics": stats,
        "recommended_settings": {
            "qwen_2.5_1.5b": {
                "format": "qwen",
                "learning_rate": 2e-5,
                "batch_size": 4,
                "gradient_accumulation": 8,
                "epochs": 3,
                "warmup_ratio": 0.1,
                "max_length": 2048,
                "lora_r": 32,
                "lora_alpha": 64
            },
            "llama_2_7b": {
                "format": "llama",
                "learning_rate": 1e-5,
                "batch_size": 2,
                "gradient_accumulation": 16,
                "epochs": 3,
                "warmup_ratio": 0.05,
                "max_length": 2048,
                "lora_r": 16,
                "lora_alpha": 32
            },
            "smollm_1.7b": {
                "format": "chatml",
                "learning_rate": 3e-5,
                "batch_size": 8,
                "gradient_accumulation": 4,
                "epochs": 4,
                "warmup_ratio": 0.1,
                "max_length": 1536,
                "lora_r": 32,
                "lora_alpha": 64
            }
        }
    }
    
    with open(output_dir / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Create README
    readme = """# Training Formats for Khmer Medical Q&A

This directory contains pre-formatted datasets optimized for different LLM fine-tuning frameworks.

## Available Formats

### 1. Alpaca Format (`alpaca_*.jsonl`)
- Stanford's instruction-following format
- Fields: instruction, input, output
- Compatible with: Most fine-tuning libraries

### 2. ChatML Format (`chatml_*.jsonl`)
- OpenAI's chat markup language format
- Fields: messages (with role and content)
- Compatible with: Modern chat models

### 3. Llama Format (`llama_*.txt`)
- Meta's Llama instruction format with special tokens
- Includes system prompt in Khmer
- Compatible with: Llama 2/3 models

### 4. Qwen Format (`qwen_*.txt`)
- Alibaba's Qwen model format
- Uses special tokens: <|im_start|>, <|im_end|>
- Compatible with: Qwen 2.5 models

### 5. Supervised Format (`supervised_*.jsonl`)
- Enhanced format with reasoning and metadata
- Includes reasoning summaries and tags
- Best for: Advanced fine-tuning with auxiliary objectives

## Training Recommendations

See `training_config.json` for recommended hyperparameters for each model.

### Quick Start with Transformers

```python
from datasets import load_dataset

# Load pre-formatted data
dataset = load_dataset('json', 
    data_files={'train': 'alpaca_train.jsonl', 'val': 'alpaca_val.jsonl'})

# Use with transformers Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    logging_dir='./logs',
)
```

## Dataset Statistics

- Total examples: ~37,500 (including paraphrases)
- Training set: 95% of data
- Validation set: 5% of data
- Average tokens per example: ~2,000

## Notes

- All formats include both original and paraphrased versions where available
- Validation split uses consistent random seed (42) for reproducibility
- Reasoning summaries are included in supervised format
"""
    
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("\n" + "="*50)
    print("✅ Training formats created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("\n📊 Format Statistics:")
    for fmt, stat in stats.items():
        print(f"  {fmt:12} - Train: {stat['train']:,}, Val: {stat['val']:,}, Total: {stat['total']:,}")
    print("\n💡 Next steps:")
    print("  1. Choose format based on your model (see training_config.json)")
    print("  2. Load the formatted data for training")
    print("  3. Use recommended hyperparameters as starting point")

if __name__ == "__main__":
    main()