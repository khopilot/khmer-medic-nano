# Khmer Medical Q&A Dataset - Clean Structure

## 📂 Directory Structure (Post-Cleanup)

```
khmer-medic-nano/
├── README.md                    # Project overview
├── DATASET_SUMMARY.md          # Dataset statistics
├── CLAUDE.md                   # AI assistant instructions
├── Makefile                    # Build automation
├── .env.example                # Environment template
├── upload_success.json         # HF upload confirmation
│
├── configs/                    # Configuration files
│   ├── project.yaml           # Project settings
│   ├── prompt_translate.txt  # Translation prompts
│   ├── prompt_paraphrase.txt # Paraphrase prompts
│   └── prompt_summary.txt    # Summary prompts
│
├── data/
│   ├── out/                   # Final outputs
│   │   ├── km_final.jsonl            # Complete dataset (90.4 MB)
│   │   ├── km_complete_augmented.jsonl # Fully augmented (55 MB)
│   │   └── validation_report.json    # Quality metrics
│   │
│   └── training_formats/      # Ready-to-use formats (334 MB)
│       ├── alpaca_train.jsonl
│       ├── alpaca_val.jsonl
│       ├── chatml_train.jsonl
│       ├── chatml_val.jsonl
│       ├── llama_train.txt
│       ├── llama_val.txt
│       ├── qwen_train.txt
│       ├── qwen_val.txt
│       ├── supervised_train.jsonl
│       ├── supervised_val.jsonl
│       ├── training_config.json
│       └── README.md
│
├── huggingface_dataset/       # HF package (201 MB)
│   ├── README.md              # Dataset card
│   ├── dataset_info.json     # Metadata
│   ├── data/                  # Parquet files
│   ├── complete_augmented/   # Augmented subset
│   └── jsonl/                 # JSONL versions
│
├── scripts/                   # Essential scripts only
│   ├── create_training_formats.py
│   ├── upload_to_hf.py
│   ├── validate_dataset_fixed.py
│   └── check_hf_user.py
│
└── archive_backup/            # Archived processing files
    └── [timestamped folders]
```

## 🎯 Quick Start

### Use the Dataset
```python
from datasets import load_dataset

# From HuggingFace Hub
dataset = load_dataset('khopilot/khmer-medical-qa')

# From local files
import json
with open('data/out/km_final.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
```

### Train a Model
```bash
# Use pre-formatted training data
python train.py --data data/training_formats/qwen_train.jsonl
```

### Validate Quality
```bash
python scripts/validate_dataset_fixed.py data/out/km_final.jsonl
```

## 📊 Dataset Info
- **Total entries**: 18,756 medical Q&A pairs
- **With paraphrases**: 9,314 (49.7%)
- **With reasoning**: 18,753 (100%)
- **Quality score**: 94.6/100
- **HuggingFace**: https://huggingface.co/datasets/khopilot/khmer-medical-qa

## 🗂️ Archived Files
Processing artifacts and intermediate files have been moved to `archive_backup/` 
to keep the main directory clean while preserving them for reference.
