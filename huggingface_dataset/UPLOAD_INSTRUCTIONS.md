# Upload Instructions for HuggingFace Hub

## Prerequisites
```bash
pip install huggingface-hub
huggingface-cli login
```

## Upload Dataset

1. Create a new dataset repository on HuggingFace:
   - Go to https://huggingface.co/new-dataset
   - Name it: khmer-medical-qa (or your preferred name)
   - Set visibility (public/private)

2. Upload using CLI:
```bash
cd huggingface_dataset
huggingface-cli upload [YOUR_USERNAME]/khmer-medical-qa . . --repo-type dataset
```

Or using Python:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="huggingface_dataset",
    repo_id="[YOUR_USERNAME]/khmer-medical-qa",
    repo_type="dataset",
)
```

## Test Loading
```python
from datasets import load_dataset

# After upload
dataset = load_dataset("[YOUR_USERNAME]/khmer-medical-qa")
print(dataset)
print(dataset['train'][0])
```

## File Structure
```
huggingface_dataset/
├── README.md                    # Dataset card
├── dataset_info.json            # Dataset metadata
├── khmer_medical_qa.py          # Loading script
├── data/
│   └── train-00000-of-00001.parquet  # Main dataset
├── complete_augmented/          # Fully augmented subset
│   └── complete_augmented-00000-of-00001.parquet
└── jsonl/                       # JSONL versions
    ├── train.jsonl
    └── complete_augmented.jsonl
```
