# CLAUDE.md

Instruction set for Claude Code Opus 4.1. Build a robust pipeline to translate the English split of **FreedomIntelligence/medical‑o1‑reasoning‑SFT** into Khmer for nano‑model training. Use **GPT‑5‑nano** by default. Optional QA with **GPT‑5**. No chain‑of‑thought translation.

---

## 1) Goals

* Translate **Question** and **Response** to Khmer for **all 90,120 rows** or **EN split 19,700 rows**.
* Add **Khmer paraphrases** on 50% of rows.
* Add **short reasoning summaries** (≤60 tokens) on 100% of rows. No step‑by‑step CoT.
* Emit clean JSONL and a ready‑to‑publish HF dataset. Deterministic, resumable, budget‑capped.

## 2) Cost targets (GPT‑5‑nano)

* EN split Q+R: ≈ **\$6.03**. **Batch** ≈ **\$3.02**.
* Full set Q+R: ≈ **\$27.58**. **Batch** ≈ **\$13.79**.
* If you added long CoT: ≈ **\$55.15**. **Do not**. Keep summaries.

## 3) Licenses

* Source dataset: **Apache‑2.0**. Khmer derivative can be redistributed. Keep source citation in README.

---

## 4) Repository layout

```
.
├─ scripts/
│  ├─ translate_sync.py            # dev subset translator (Responses API)
│  ├─ prepare_batch.py             # build Batch JSONL from HF rows
│  ├─ submit_batch.py              # upload JSONL, create batch job
│  ├─ poll_batch.py                # poll until complete, download output
│  ├─ merge_batch_results.py       # align outputs to inputs by custom_id
│  ├─ paraphrase_prepare.py        # 50% sampling + request file for paraphrases
│  ├─ summarize_reasoning.py       # short high‑level summaries (≤60 tokens)
│  ├─ validate_jsonl.py            # schema + safety + formatting checks
│  ├─ estimate_cost.py             # token and $ estimator (nano/mini)
│  ├─ sample_qa.py                 # 3% GPT‑5 spot‑check
│  └─ to_hf_dataset.py             # package JSONL → HF folder
├─ configs/
│  ├─ project.yaml                 # splits, fields, budgets, coverage
│  ├─ prompt_translate.txt         # authoritative translator prompt
│  ├─ prompt_paraphrase.txt        # Khmer paraphrase prompt
│  └─ prompt_summary.txt           # short reasoning summary prompt
├─ data/
│  ├─ raw/                         # cached HF shards
│  ├─ work/                        # batch requests/outputs
│  ├─ out/                         # final JSONL + parquet
│  └─ hf/                          # dataset folder to publish
├─ Makefile
├─ .env.example
└─ README.md
```

---

## 5) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U datasets pandas pyarrow tqdm pydantic rich
pip install -U openai python-dotenv
# optional QA/metrics
pip install -U sacrebleu langdetect
```

`.env.example`

```
OPENAI_API_KEY=sk-***
OPENAI_MODEL_MAIN=gpt-5-nano
OPENAI_MODEL_QA=gpt-5
OPENAI_BATCH_POLL_SECONDS=15
HF_DATASET_ID=FreedomIntelligence/medical-o1-reasoning-SFT
HF_CONFIG=en
HF_SPLIT=train
```

---

## 6) Configuration

`configs/project.yaml`

```yaml
name: medical-km-nano-v2
source:
  hf_id: FreedomIntelligence/medical-o1-reasoning-SFT
  hf_config: en
  hf_split: train
fields:
  question: Question
  response: Response
coverage:
  paraphrase_rate: 0.5    # 50% of rows
  summary_rate: 1.0       # 100% of rows
budget:
  max_usd: 30.0           # full EN split cap
  batch: true
translation:
  max_output_tokens: 800
  temperature: 0.2
outputs:
  translate_jsonl: data/out/km_translations.jsonl
  merged_jsonl:    data/out/km_merged.jsonl
  paraphrase_jsonl:data/out/km_paraphrases.jsonl
  summary_jsonl:   data/out/km_summaries.jsonl
  final_jsonl:     data/out/km_final.jsonl
  hf_root:         data/hf
qa:
  sample_rate: 0.03
  fail_on_major: true
```

---

## 7) Prompts

`configs/prompt_translate.txt`

```
You are a professional EN→KM medical translator.

TASK
Translate **Question** and **Response** into Khmer.

REQUIREMENTS
- Faithful and precise. No summaries, no additions, no explanations.
- Preserve clinical meaning, numbers, units, drug names, acronyms, gene names, and codes in Latin script (e.g., IV, PO, MRI, ICD‑10).
- Preserve formatting: paragraphs, bullet/numbered lists, tables, math, markdown.
- Keep placeholders and markup unchanged: {...} [...] (...) <...> `code` $math$.
- Use clear, formal Khmer. Arabic numerals (0–9).
- If a field is empty or already Khmer, copy as is.

OUTPUT
Return only strict JSON:
{"question_km":"<Khmer translation of Question>","response_km":"<Khmer translation of Response>"}

INPUT
Question:
{{QUESTION_EN}}

Response:
{{RESPONSE_EN}}
```

`configs/prompt_paraphrase.txt`

```
You are a Khmer editor.
Paraphrase each field in **Khmer** to a natural alternative. Preserve meaning, numbers, units, and formatting. No extra info.
Return JSON: {"question_km_para":"…","response_km_para":"…"}
Input JSON: {"question_km":"…","response_km":"…"}
```

`configs/prompt_summary.txt`

```
Write a ≤60‑token high‑level reasoning summary in **Khmer** for the given Question and Response. No chain‑of‑thought, no step lists.
Return JSON: {"reasoning_summary_km":"…","tags":["reasoning_type:<diagnosis|workup|treatment>","evidence:…"]}
Input JSON: {"question_km":"…","response_km":"…"}
```

---

## 8) Core scripts — specs

### A) `scripts/translate_sync.py`

* Load HF rows.
* Build messages from `prompt_translate.txt`.
* Call Responses API with `model=$OPENAI_MODEL_MAIN`, `max_output_tokens=800`, `temperature=0.2`.
* Parse `output_text` as JSON. Enforce keys. Write to `translate_jsonl`.
* Resume by skipping existing IDs.

### B) Batch mode

* `prepare_batch.py`: write `data/work/batch_requests.jsonl` with one POST per row. `custom_id=en-train-{i:06d}`.
* `submit_batch.py`: upload file, create batch, save `batch_id`.
* `poll_batch.py`: poll state until `completed`, download `batch_output.jsonl`.
* `merge_batch_results.py`: align by `custom_id`, write `merged_jsonl`.

### C) Paraphrases

* `paraphrase_prepare.py`: sample p=0.5 of **translated** rows. Build requests using `prompt_paraphrase.txt`.
* Submit via Batch or Responses. Output `paraphrase_jsonl`.

### D) Reasoning summaries

* `summarize_reasoning.py`: run over all translated rows with `prompt_summary.txt`. Output `summary_jsonl`.

### E) Merge final

* Join base translations + paraphrases (if present) + summaries into `final_jsonl` with schema:

```
{id, question_en, response_en, question_km, response_km, question_km_para?, response_km_para?, reasoning_summary_km, tags, meta}
```

`meta` includes model name, timestamp, prompt hash.

### F) Validation

* `validate_jsonl.py` checks:

  * JSON schema and non‑empty Khmer.
  * Language ID km ≥98%.
  * Numbers and units unchanged (regex: mg|mL|mmHg|°C|bpm|Na|K|Cr|IV|PO|IM|MRI|CT|ICD‑10).
  * Markdown lists preserved.
  * No English hallucinated inside Khmer beyond expected abbreviations.
* Exit non‑zero on violation.

### G) QA

* `sample_qa.py` randomly draws 3% rows.
* Ask **GPT‑5** to score adequacy (1–5) and term preservation (pass/fail).
* Fail the run if ≥5% fails or mean adequacy <4.2.

### H) Packaging

* `to_hf_dataset.py` writes Parquet shards under `data/hf/{train}/` and a README with method, prompt, metrics, license.

---

## 9) Makefile

```makefile
.PHONY: install sync batch paraphrase summary validate qa package clean

install:
	python -m venv .venv && . .venv/bin/activate && \
	pip install -U datasets pandas pyarrow tqdm pydantic rich openai python-dotenv sacrebleu langdetect

sync:
	python scripts/translate_sync.py

batch:
	python scripts/prepare_batch.py
	python scripts/submit_batch.py
	python scripts/poll_batch.py
	python scripts/merge_batch_results.py

paraphrase:
	python scripts/paraphrase_prepare.py && \
	python scripts/submit_batch.py && \
	python scripts/poll_batch.py && \
	python scripts/merge_batch_results.py

summary:
	python scripts/summarize_reasoning.py

validate:
	python scripts/validate_jsonl.py data/out/km_merged.jsonl

qa:
	python scripts/sample_qa.py data/out/km_merged.jsonl

package:
	python scripts/to_hf_dataset.py

clean:
	rm -rf data/work/* data/out/*
```

---

## 10) Budget control

* `estimate_cost.py` returns projected \$ given rows × tokens.
* `submit_batch.py` reads `budget.max_usd`. Abort if projection exceeds cap.
* Prefer Batch for full runs. Use Sync for dev subsets.

---

## 11) Training handoff (nano model)

* Base model: **Qwen2.5‑1.5B** or **SmolLM2‑1.7B**.
* Stage A: generic Khmer 30–60M tokens (rinabuoy corpora).
* Stage B: medical Q+R Khmer 60–90M tokens.
* Stage C: SFT with `reasoning_summary_km`.
* LoRA rank 16–32. Seq len 2k. Cosine LR 1e‑4. Batch size to fit.
* Export GGUF/MLX for Mac. Keep eval set of 1k held‑out rows.

---

## 12) Runbook

1. Fill `.env` with keys and models.
2. Edit `configs/project.yaml` as needed.
3. `make install`
4. `make sync` on a 1k slice.
5. `make batch` for full split.
6. `make paraphrase` then `make summary`.
7. `make validate` and `make qa`.
8. `make package` and push to HF.

---

## 13) Deliverables

* `data/out/km_merged.jsonl` and `km_final.jsonl`.
* `data/hf/` Parquet + README.
* `qa_report.jsonl`, `validate.log`.
* Makefile and scripts ready for CI.

*End.*
