# Khmer Medical Translation Pipeline for Nano Models

A robust pipeline for translating the **FreedomIntelligence/medical-o1-reasoning-SFT** dataset from English to Khmer, optimized for training nano-scale medical reasoning models (1.5B-2B parameters).

## 🎯 Project Goals

- Translate 19,700 medical Q&A pairs from English to Khmer
- Add Khmer paraphrases for 50% of rows
- Generate high-level reasoning summaries (≤60 tokens) for all rows
- Maintain medical accuracy and terminology preservation
- Create production-ready dataset for HuggingFace

## 💰 Cost Estimates (GPT-5-nano)

- **EN split (19,700 rows)**: ~$6.03 standard, ~$3.02 batch
- **Full dataset (90,120 rows)**: ~$27.58 standard, ~$13.79 batch
- **Budget cap**: $30.00

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd khmer-medic-nano

# Install dependencies
make install

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Estimate Costs

```bash
make estimate
```

### 3. Run Translation Pipeline

#### Development Mode (100 rows)
```bash
make sync
```

#### Production Mode (Full Dataset)
```bash
make batch
```

### 4. Validate & QA

```bash
make validate
make qa
```

### 5. Package for HuggingFace

```bash
make package
```

## 📁 Project Structure

```
.
├── scripts/
│   ├── translate_sync.py          # Synchronous translator (dev)
│   ├── prepare_batch.py           # Prepare batch requests
│   ├── submit_batch.py             # Submit to OpenAI Batch API
│   ├── poll_batch.py               # Poll and download results
│   ├── merge_batch_results.py     # Merge translations
│   ├── paraphrase_prepare.py      # Prepare paraphrase requests
│   ├── summarize_reasoning.py     # Generate reasoning summaries
│   ├── validate_jsonl.py          # Validate translations
│   ├── sample_qa.py               # Quality assurance sampling
│   ├── estimate_cost.py           # Cost estimation
│   └── to_hf_dataset.py          # Package for HuggingFace
├── configs/
│   ├── project.yaml               # Project configuration
│   ├── prompt_translate.txt       # Translation prompt
│   ├── prompt_paraphrase.txt      # Paraphrase prompt
│   └── prompt_summary.txt         # Summary prompt
├── data/
│   ├── raw/                       # Cached HF dataset
│   ├── work/                      # Batch processing files
│   ├── out/                       # Final outputs
│   └── hf/                        # HuggingFace dataset
├── Makefile                        # Build automation
├── .env.example                    # Environment template
└── README.md                       # This file
```

## 🔧 Configuration

Edit `configs/project.yaml` to customize:

- Dataset source and fields
- Coverage rates (paraphrase, summary)
- Budget limits
- Output paths
- QA thresholds

## 📊 Output Schema

```json
{
  "id": "en-train-000001",
  "question_en": "Original English question",
  "response_en": "Original English response",
  "question_km": "Khmer translation of question",
  "response_km": "Khmer translation of response",
  "question_km_para": "Khmer paraphrase (50% of rows)",
  "response_km_para": "Khmer paraphrase (50% of rows)",
  "reasoning_summary_km": "High-level summary in Khmer",
  "tags": ["reasoning_type:diagnosis", "evidence:clinical"],
  "meta": {
    "model": "gpt-5-nano",
    "timestamp": "2025-01-12T...",
    "batch": true
  }
}
```

## 🧪 Validation Checks

- **Schema compliance**: All required fields present
- **Language detection**: Khmer probability ≥98%
- **Term preservation**: Medical terms, units, codes maintained
- **Format preservation**: Lists, paragraphs, markdown
- **QA sampling**: 3% evaluated by GPT-5

## 🎓 Training Recommendations

### Base Models
- Qwen2.5-1.5B
- SmolLM2-1.7B

### Training Stages
1. **Stage A**: Generic Khmer corpus (30-60M tokens)
2. **Stage B**: Medical domain with this dataset (60-90M tokens)
3. **Stage C**: SFT with reasoning summaries

### Hyperparameters
- LoRA rank: 16-32
- Sequence length: 2048
- Learning rate: 1e-4 (cosine)
- Batch size: Adjust for GPU

## 📜 License

Apache 2.0 (inherited from source dataset)

## 🙏 Acknowledgments

- Original dataset: FreedomIntelligence
- Translation: OpenAI GPT-5-nano
- Khmer NLP community

## 📞 Support

For issues or questions, please open a GitHub issue.