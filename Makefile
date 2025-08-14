.PHONY: install sync batch paraphrase summary validate qa package clean estimate help

# Default target
help:
	@echo "Khmer Medical Translation Pipeline"
	@echo "=================================="
	@echo "Available targets:"
	@echo "  make install      - Set up Python environment and install dependencies"
	@echo "  make estimate     - Estimate translation costs"
	@echo "  make sync         - Run synchronous translation (dev mode)"
	@echo "  make batch        - Run full batch translation pipeline"
	@echo "  make paraphrase   - Generate paraphrases for 50% of rows"
	@echo "  make summary      - Generate reasoning summaries for all rows"
	@echo "  make validate     - Validate translation quality"
	@echo "  make qa           - Run QA sampling with GPT-5"
	@echo "  make package      - Package dataset for Hugging Face"
	@echo "  make clean        - Clean work and output directories"
	@echo "  make all          - Run complete pipeline (batch mode)"

# Install dependencies
install:
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install -U pip && \
	pip install -U datasets pandas pyarrow tqdm pydantic rich openai python-dotenv sacrebleu langdetect

# Estimate costs
estimate:
	@. .venv/bin/activate && python scripts/estimate_cost.py

# Synchronous translation (for development/testing)
sync:
	@. .venv/bin/activate && python scripts/translate_sync.py

# Batch translation pipeline
batch:
	@echo "Running batch translation pipeline..."
	@. .venv/bin/activate && python scripts/prepare_batch.py
	@. .venv/bin/activate && python scripts/submit_batch.py
	@. .venv/bin/activate && python scripts/poll_batch.py
	@. .venv/bin/activate && python scripts/merge_batch_results.py

# Paraphrase generation
paraphrase:
	@echo "Generating paraphrases..."
	@. .venv/bin/activate && python scripts/paraphrase_prepare.py
	@echo "Note: Submit the paraphrase batch manually with updated input file"

# Summary generation
summary:
	@echo "Generating reasoning summaries..."
	@. .venv/bin/activate && python scripts/summarize_reasoning.py
	@echo "Note: Submit the summary batch manually with updated input file"

# Validation
validate:
	@. .venv/bin/activate && python scripts/validate_jsonl.py data/out/km_merged.jsonl

# Quality assurance
qa:
	@. .venv/bin/activate && python scripts/sample_qa.py data/out/km_merged.jsonl

# Package for HuggingFace
package:
	@. .venv/bin/activate && python scripts/to_hf_dataset.py

# Clean work and output directories
clean:
	rm -rf data/work/* data/out/*
	@echo "Cleaned work and output directories"

# Complete pipeline
all: estimate batch validate qa package
	@echo "Pipeline complete!"