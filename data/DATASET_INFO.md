# Khmer Medical Dataset

## Overview
- **Source**: FreedomIntelligence/medical-o1-reasoning-SFT (EN split)
- **Entries**: 18,756 medical Q&A pairs
- **Language**: English → Khmer translation
- **Augmentations**: Paraphrases (50%) + Reasoning summaries (100%)
- **Total tokens**: ~36.5M
- **Quality score**: 94.6/100
- **Date**: 2025-08-14

## Files
- `data/out/km_final.jsonl` - Complete dataset
- `data/out/km_complete_augmented.jsonl` - Entries with all augmentations
- `data/backup/` - Intermediate processing files

## Processing Pipeline
1. Translation: GPT-4o-mini (batch API)
2. Paraphrases: GPT-4o-mini (50% of entries)
3. Reasoning summaries: GPT-4o-mini (100% of entries)
4. Total cost: ~$5.64

## Ready for Fine-tuning
Suitable for models like:
- Qwen 2.5 1.5B
- SmolLM 1.7B
- Other nano models (1-2B parameters)
