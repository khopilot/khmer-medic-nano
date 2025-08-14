#!/usr/bin/env python3
"""
Upload Khmer Medical Q&A Dataset to HuggingFace Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import json

def upload_dataset(token: str, repo_id: str = "niko/khmer-medical-qa"):
    """
    Upload the dataset to HuggingFace Hub
    
    Args:
        token: HuggingFace API token
        repo_id: Repository ID (username/dataset-name)
    """
    api = HfApi(token=token)
    
    # Path to the prepared dataset
    dataset_path = Path("huggingface_dataset")
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    print("📦 Preparing to upload Khmer Medical Q&A Dataset")
    print(f"📍 Repository: {repo_id}")
    print(f"📁 Source: {dataset_path}")
    
    try:
        # Create repository if it doesn't exist
        print("\n🔄 Creating/checking repository...")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                token=token,
                exist_ok=True
            )
            print("✅ Repository ready")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("✅ Repository already exists")
            else:
                raise
        
        # Upload the dataset folder
        print("\n📤 Uploading dataset files...")
        print("This may take a few minutes depending on your connection...")
        
        # List files to upload
        files_to_upload = []
        for file in dataset_path.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(dataset_path)
                size_mb = file.stat().st_size / (1024 * 1024)
                files_to_upload.append((rel_path, size_mb))
                print(f"  📄 {rel_path} ({size_mb:.1f} MB)")
        
        print(f"\n📊 Total files: {len(files_to_upload)}")
        print(f"📊 Total size: {sum(f[1] for f in files_to_upload):.1f} MB")
        
        # Upload the folder
        url = api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload Khmer Medical Q&A Dataset v1.0"
        )
        
        print("\n✅ Upload successful!")
        print(f"🌐 Dataset URL: https://huggingface.co/datasets/{repo_id}")
        print(f"📊 Commit URL: {url}")
        
        # Test loading instructions
        print("\n📖 To load your dataset:")
        print(f"```python")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{repo_id}')")
        print(f"print(dataset['train'][0])")
        print(f"```")
        
        # Create success file
        success_info = {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "commit_url": url,
            "files_uploaded": len(files_to_upload),
            "total_size_mb": sum(f[1] for f in files_to_upload)
        }
        
        with open("upload_success.json", "w") as f:
            json.dump(success_info, f, indent=2)
        
        print("\n✅ Upload complete! Dataset is now available on HuggingFace Hub.")
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your token has write permissions")
        print("3. Try a different repository name if it already exists")
        return False

def main():
    # HuggingFace token - set as environment variable
    token = os.getenv("HF_TOKEN", "your-hf-token-here")
    
    # You can change the repository name here
    # Format: "username/dataset-name"
    repo_id = "khopilot/khmer-medical-qa"  # Using your HF username
    
    print("🚀 Khmer Medical Q&A Dataset Uploader")
    print("=" * 50)
    
    # Confirm upload
    print(f"\n📝 Configuration:")
    print(f"  Repository: {repo_id}")
    print(f"  Token: {token[:10]}...{token[-4:]}")
    
    # Upload the dataset
    success = upload_dataset(token, repo_id)
    
    if success:
        print("\n🎉 All done! Your dataset is live on HuggingFace!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()