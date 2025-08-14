---
language:
- km
- en
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- text-generation
- translation
pretty_name: Khmer Medical Q&A Dataset
tags:
- medical
- healthcare
- khmer
- translation
- reasoning
- nano-model
- instruction-tuning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  default: true
dataset_info:
  features:
  - name: index
    dtype: int64
  - name: question_en
    dtype: string
  - name: response_en
    dtype: string
  - name: question_km
    dtype: string
  - name: response_km
    dtype: string
  - name: question_km_para
    dtype: string
  - name: response_km_para
    dtype: string
  - name: reasoning_summary_km
    dtype: string
  - name: tags
    sequence: string
  splits:
  - name: train
    num_bytes: 94796672
    num_examples: 18756
---

# Khmer Medical Q&A Dataset

## Dataset Description

This dataset contains 18,756 high-quality medical question-answer pairs translated from English to Khmer, with additional paraphrases and reasoning summaries. It's designed for fine-tuning nano-scale language models (1-2B parameters) for Khmer medical applications.

### Dataset Summary

- **Total entries**: 18,756 medical Q&A pairs
- **Source language**: English
- **Target language**: Khmer (កម្ពុជា)
- **Paraphrases**: 9,314 entries (49.7%)
- **Reasoning summaries**: 18,753 entries (100%)
- **Total tokens**: ~36.5M
- **Quality score**: 94.6/100

## Dataset Sources

### Original Dataset
- **Name**: FreedomIntelligence/medical-o1-reasoning-SFT
- **Split**: English (en)
- **License**: Apache 2.0
- **Description**: Medical reasoning dataset with chain-of-thought responses

### Translation & Augmentation
- **Translation model**: GPT-4o-mini (OpenAI Batch API)
- **Paraphrase generation**: GPT-4o-mini (50% of entries)
- **Reasoning summaries**: GPT-4o-mini (100% of entries)
- **Processing date**: August 2025
- **Total cost**: ~$5.64

## Dataset Structure

### Data Fields

- `index` (int): Unique identifier for each entry
- `question_en` (string): Original English medical question
- `response_en` (string): Original English medical response
- `question_km` (string): Khmer translation of the question
- `response_km` (string): Khmer translation of the response
- `question_km_para` (string, optional): Khmer paraphrase of the question
- `response_km_para` (string, optional): Khmer paraphrase of the response
- `reasoning_summary_km` (string): Concise Khmer summary of medical reasoning
- `tags` (list): Reasoning type tags (diagnosis, treatment, workup, etc.)

### Data Splits

Currently only a training split is provided:
- `train`: 18,756 examples

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("username/khmer-medical-qa")

# Load only entries with all augmentations
dataset = load_dataset("username/khmer-medical-qa", data_files="data/complete_augmented-*")

# Example entry
print(dataset['train'][0])
```

### Fine-tuning Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Format for instruction tuning
def format_prompt(example):
    prompt = f"សំណួរ: {example['question_km']}\n\nចម្លើយ: {example['response_km']}"
    return {"text": prompt}

formatted_dataset = dataset.map(format_prompt)
```

## Intended Uses

### Primary Use Cases
- Fine-tuning Khmer medical chatbots
- Training medical Q&A systems
- Developing healthcare assistants for Khmer speakers
- Research in low-resource language medical NLP

### Recommended Models
- Qwen 2.5 1.5B
- SmolLM 1.7B
- Other nano models (1-2B parameters)

## Quality & Validation

### Quality Metrics
- **Structure validation**: 100% pass
- **Language validation**: 94.6% pure Khmer content
- **Medical term preservation**: 98%+ accuracy
- **No placeholder issues**: ✓

### Known Limitations
- Medical terminology often uses Latin/English terms (expected)
- Some entries have more English medical abbreviations
- Translation may not capture all cultural nuances

## Ethical Considerations

### Medical Disclaimer
**⚠️ Important**: This dataset is for research and educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

### Bias Considerations
- Dataset reflects medical knowledge from English-speaking contexts
- May not fully represent traditional Khmer medical practices
- Should be validated by Khmer medical professionals before deployment

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{khmer_medical_qa_2025,
  title={Khmer Medical Q&A Dataset},
  author={Your Name},
  year={2025},
  publisher={HuggingFace},
  note={Translated and augmented from FreedomIntelligence/medical-o1-reasoning-SFT}
}
```

### Original Dataset Citation

```bibtex
@dataset{medical_o1_reasoning_sft,
  title={Medical O1 Reasoning SFT Dataset},
  author={FreedomIntelligence},
  publisher={HuggingFace}
}
```

## License

This dataset is released under the Apache 2.0 License, same as the original dataset.

## Acknowledgments

- Original dataset by FreedomIntelligence
- Translation powered by OpenAI GPT-4o-mini
- Processing infrastructure by Anthropic Claude

## Contact

For questions or issues, please open an issue on the dataset repository.
