#!/usr/bin/env python3
"""
Final cleanup of the project directory
Keep only essential files for production use
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def get_directory_size(path):
    """Calculate total size of a directory"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total

def main():
    print("🧹 Final Project Cleanup")
    print("=" * 50)
    
    # Define what to keep and what to clean
    essential_structure = {
        "keep": {
            # Root files
            "README.md": "Project documentation",
            "DATASET_SUMMARY.md": "Dataset summary",
            "CLAUDE.md": "Claude instructions",
            "Makefile": "Build automation",
            ".env.example": "Environment template",
            "upload_success.json": "HF upload record",
            
            # Directories to keep entirely
            "huggingface_dataset/": "HuggingFace package (uploaded)",
            "data/training_formats/": "Ready-to-use training formats",
            "configs/": "Configuration files",
            
            # Specific files in data/out
            "data/out/km_final.jsonl": "Final complete dataset",
            "data/out/km_complete_augmented.jsonl": "Fully augmented subset",
            "data/out/validation_report.json": "Quality metrics",
            
            # Key scripts
            "scripts/create_training_formats.py": "Format converter",
            "scripts/upload_to_hf.py": "HF uploader",
            "scripts/validate_dataset_fixed.py": "Validation tool",
        },
        "archive": {
            # Move to archive for reference but not daily use
            "data/work/": "Intermediate processing files",
            "data/raw/": "Original HF dataset cache",
            "data/backup/": "Old backups",
            "scripts/": "Processing scripts (except essentials)",
        },
        "remove": {
            # Can be safely deleted
            "data/out/km_merged.jsonl": "Old partial merge",
            "data/out/km_translations.jsonl": "Early test file",
            "data/hf/": "Empty directory",
            "__pycache__/": "Python cache",
            ".DS_Store": "MacOS metadata",
        }
    }
    
    # Calculate current sizes
    print("\n📊 Current Directory Status:")
    root = Path(".")
    total_before = get_directory_size(root)
    print(f"Total size: {format_size(total_before)}")
    
    # Create archive directory
    archive_dir = Path("archive_backup")
    archive_dir.mkdir(exist_ok=True)
    
    # Statistics
    stats = {
        "removed_files": [],
        "archived_files": [],
        "kept_files": [],
        "space_freed": 0
    }
    
    # 1. Remove unnecessary files
    print("\n🗑️  Removing unnecessary files...")
    for pattern, desc in essential_structure["remove"].items():
        paths = list(root.rglob(pattern.rstrip('/')))
        for path in paths:
            if path.exists():
                size = os.path.getsize(path) if path.is_file() else get_directory_size(path)
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                stats["removed_files"].append(str(path))
                stats["space_freed"] += size
                print(f"  ✓ Removed: {path} ({format_size(size)})")
    
    # 2. Archive old processing files
    print("\n📦 Archiving processing artifacts...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dir_pattern, desc in essential_structure["archive"].items():
        source = Path(dir_pattern.rstrip('/'))
        if source.exists():
            # Special handling for scripts - keep essential ones
            if "scripts" in str(source):
                # Create scripts archive but keep essential scripts
                scripts_archive = archive_dir / f"scripts_{timestamp}"
                scripts_archive.mkdir(exist_ok=True)
                
                essential_scripts = [
                    "create_training_formats.py",
                    "upload_to_hf.py", 
                    "validate_dataset_fixed.py",
                    "final_cleanup.py",  # Keep this script too
                    "check_hf_user.py"   # Useful utility
                ]
                
                for script in Path("scripts").glob("*.py"):
                    if script.name not in essential_scripts:
                        dest = scripts_archive / script.name
                        shutil.move(str(script), str(dest))
                        stats["archived_files"].append(str(script))
                        print(f"  ✓ Archived: {script.name}")
            else:
                # Archive entire directory
                dest = archive_dir / f"{source.name}_{timestamp}"
                size = get_directory_size(source)
                shutil.move(str(source), str(dest))
                stats["archived_files"].append(str(source))
                stats["space_freed"] += size
                print(f"  ✓ Archived: {source} → {dest.name} ({format_size(size)})")
    
    # 3. Create clean structure report
    print("\n📁 Creating clean structure documentation...")
    
    structure_doc = """# Khmer Medical Q&A Dataset - Clean Structure

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
"""
    
    with open("CLEAN_STRUCTURE.md", "w") as f:
        f.write(structure_doc)
    
    # 4. Create cleanup report
    total_after = get_directory_size(root)
    
    report = {
        "timestamp": timestamp,
        "statistics": {
            "size_before_mb": total_before / (1024*1024),
            "size_after_mb": total_after / (1024*1024),
            "space_freed_mb": stats["space_freed"] / (1024*1024),
            "reduction_percent": (stats["space_freed"] / total_before * 100) if total_before > 0 else 0,
            "files_removed": len(stats["removed_files"]),
            "files_archived": len(stats["archived_files"])
        },
        "actions": stats
    }
    
    with open("cleanup_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # 5. Final summary
    print("\n" + "=" * 50)
    print("✅ Cleanup Complete!")
    print(f"\n📊 Results:")
    print(f"  • Size before: {format_size(total_before)}")
    print(f"  • Size after: {format_size(total_after)}")
    print(f"  • Space freed: {format_size(stats['space_freed'])} ({report['statistics']['reduction_percent']:.1f}%)")
    print(f"  • Files removed: {len(stats['removed_files'])}")
    print(f"  • Files archived: {len(stats['archived_files'])}")
    
    print(f"\n📁 Clean structure documented in: CLEAN_STRUCTURE.md")
    print(f"📋 Detailed report saved to: cleanup_report.json")
    print(f"🗂️  Archives saved to: archive_backup/")
    
    print("\n✨ Your project is now clean and production-ready!")
    print("   Essential files preserved, archives available if needed.")

if __name__ == "__main__":
    main()