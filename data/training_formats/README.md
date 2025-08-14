# Training Formats for Khmer Medical Q&A

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
