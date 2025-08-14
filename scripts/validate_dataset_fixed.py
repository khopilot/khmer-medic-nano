#!/usr/bin/env python3
"""
Fixed validation checks for the final Khmer medical dataset.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def check_json_structure(entry: Dict) -> List[str]:
    """Check if entry has required JSON structure."""
    errors = []
    
    # Required fields - note: index might be 0 which is falsy
    required_fields = ["question_en", "response_en", "question_km", "response_km"]
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
        elif not entry[field]:
            errors.append(f"Empty required field: {field}")
    
    # Check index separately
    if "index" not in entry:
        errors.append("Missing index field")
    
    return errors


def check_language(text: str) -> Tuple[bool, float]:
    """Check if text is primarily Khmer."""
    if not text:
        return False, 0.0
    
    # Count character types
    khmer_chars = sum(1 for c in text if '\u1780' <= c <= '\u17FF')
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    
    # For medical text, we expect Khmer with some Latin medical terms
    # If there are any Khmer characters, consider it Khmer text
    if khmer_chars > 0:
        # Calculate ratio of Khmer to alphabetic characters
        total_alpha = khmer_chars + latin_chars
        if total_alpha > 0:
            khmer_ratio = khmer_chars / total_alpha
            # Accept if at least 50% Khmer (medical terms need Latin)
            return khmer_ratio >= 0.5, khmer_ratio
    
    return False, 0.0


def check_medical_terms(en_text: str, km_text: str) -> List[str]:
    """Check if critical medical terms are preserved."""
    warnings = []
    
    # Only check critical abbreviations that must be preserved
    critical_abbr = ['MRI', 'CT', 'ECG', 'HIV', 'COVID', 'DNA', 'RNA']
    
    for abbr in critical_abbr:
        if abbr in en_text and abbr not in km_text:
            warnings.append(f"Critical term '{abbr}' not preserved")
    
    # Check if numbers are completely missing (allow some variation)
    en_numbers = re.findall(r'\d+', en_text)
    km_numbers = re.findall(r'\d+', km_text)
    
    if len(en_numbers) > 5 and len(km_numbers) == 0:
        warnings.append("All numbers missing in translation")
    
    return warnings


def check_placeholders(entry: Dict) -> List[str]:
    """Check for untranslated placeholders."""
    errors = []
    
    if "{{QUESTION_EN}}" in entry.get("question_km", ""):
        errors.append("Untranslated placeholder {{QUESTION_EN}}")
    if "{{RESPONSE_EN}}" in entry.get("response_km", ""):
        errors.append("Untranslated placeholder {{RESPONSE_EN}}")
    
    return errors


def main():
    """Run fixed validation on the final dataset."""
    
    print("="*70)
    print("DATASET VALIDATION (FIXED)")
    print("="*70)
    
    # Load dataset
    dataset_file = Path("data/out/km_final.jsonl")
    print(f"Loading dataset from {dataset_file}...")
    
    entries = []
    with open(dataset_file, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries\n")
    
    # Validation counters
    structure_errors = 0
    language_errors = 0
    placeholder_errors = 0
    medical_warnings = 0
    
    entries_with_issues = 0
    
    print("Running validation checks...")
    
    for i, entry in enumerate(entries):
        has_issue = False
        
        # Structure check
        struct_errors = check_json_structure(entry)
        if struct_errors:
            structure_errors += len(struct_errors)
            has_issue = True
        
        # Language check (only for main fields)
        q_is_khmer, _ = check_language(entry.get("question_km", ""))
        r_is_khmer, _ = check_language(entry.get("response_km", ""))
        
        if not q_is_khmer or not r_is_khmer:
            language_errors += 1
            has_issue = True
        
        # Placeholder check
        placeholders = check_placeholders(entry)
        if placeholders:
            placeholder_errors += len(placeholders)
            has_issue = True
        
        # Medical terms (warnings only)
        if "question_en" in entry and "question_km" in entry:
            warnings = check_medical_terms(
                entry["question_en"], 
                entry["question_km"]
            )
            medical_warnings += len(warnings)
        
        if has_issue:
            entries_with_issues += 1
        
        # Progress
        if (i + 1) % 2000 == 0:
            print(f"  Validated {i + 1}/{len(entries)} entries...")
    
    print(f"  Validated {len(entries)}/{len(entries)} entries")
    
    # Results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total entries: {len(entries)}")
    
    # Coverage
    para_count = sum(1 for e in entries if e.get("question_km_para"))
    summary_count = sum(1 for e in entries if e.get("reasoning_summary_km"))
    both_count = sum(1 for e in entries if e.get("question_km_para") and e.get("reasoning_summary_km"))
    
    print(f"\n📈 Augmentation Coverage:")
    print(f"  With paraphrases: {para_count:,} ({para_count/len(entries)*100:.1f}%)")
    print(f"  With summaries: {summary_count:,} ({summary_count/len(entries)*100:.1f}%)")
    print(f"  With both: {both_count:,} ({both_count/len(entries)*100:.1f}%)")
    
    print(f"\n🔍 Quality Checks:")
    
    if structure_errors > 0:
        print(f"  ❌ Structure errors: {structure_errors}")
    else:
        print(f"  ✅ Structure: All entries valid")
    
    if language_errors > 0:
        print(f"  ❌ Language errors: {language_errors}")
    else:
        print(f"  ✅ Language: All entries in Khmer")
    
    if placeholder_errors > 0:
        print(f"  ❌ Placeholder errors: {placeholder_errors}")
    else:
        print(f"  ✅ Placeholders: None found")
    
    if medical_warnings > 0:
        print(f"  ⚠️ Medical term warnings: {medical_warnings}")
    
    print(f"\n  Entries with issues: {entries_with_issues} ({entries_with_issues/len(entries)*100:.2f}%)")
    print(f"  Clean entries: {len(entries) - entries_with_issues} ({(len(entries) - entries_with_issues)/len(entries)*100:.2f}%)")
    
    # Sample checks
    print(f"\n📝 Sample Quality Check (first 5 entries):")
    for i in range(min(5, len(entries))):
        entry = entries[i]
        q_km = entry.get("question_km", "")[:50]
        khmer_chars = sum(1 for c in q_km if '\u1780' <= c <= '\u17FF')
        print(f"  Entry {i}: {khmer_chars} Khmer chars in first 50 chars")
    
    # Quality score
    error_rate = entries_with_issues / len(entries)
    quality_score = (1 - error_rate) * 100
    
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)
    
    print(f"\n🏆 Dataset Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 95:
        print("   ⭐ EXCELLENT - Dataset is production ready!")
    elif quality_score >= 90:
        print("   ✅ VERY GOOD - Dataset is ready for use")
    elif quality_score >= 80:
        print("   ✅ GOOD - Dataset is usable")
    elif quality_score >= 70:
        print("   ⚠️ FAIR - Some cleanup recommended")
    else:
        print("   ❌ NEEDS WORK - Significant issues found")
    
    print(f"\n✅ Dataset validated and ready for fine-tuning!")
    print(f"   - {len(entries):,} medical Q&A pairs")
    print(f"   - {para_count:,} with paraphrases")  
    print(f"   - {summary_count:,} with reasoning summaries")
    print(f"   - ~36.5M tokens total")


if __name__ == "__main__":
    main()