#!/usr/bin/env python3
"""
Convert fine-tuned model to GGUF format for deployment
Compatible with llama.cpp, Ollama, LM Studio, etc.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("model_path", help="Path to the model to convert")
    parser.add_argument("--output_dir", default="models/gguf", help="Output directory")
    parser.add_argument("--quantization", default="Q4_K_M", 
                       choices=["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                       help="Quantization type")
    parser.add_argument("--llama_cpp_path", default=None, help="Path to llama.cpp repo")
    return parser.parse_args()

def setup_llama_cpp(llama_cpp_path=None):
    """Clone and build llama.cpp if needed"""
    if llama_cpp_path and Path(llama_cpp_path).exists():
        return Path(llama_cpp_path)
    
    # Clone llama.cpp if not present
    if not Path("llama.cpp").exists():
        print("📥 Cloning llama.cpp...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/ggerganov/llama.cpp.git"
        ], check=True)
    
    # Build llama.cpp
    print("🔨 Building llama.cpp...")
    os.chdir("llama.cpp")
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make", "-j"], check=True)
    os.chdir("..")
    
    return Path("llama.cpp")

def convert_to_gguf(model_path, output_dir, quantization, llama_cpp_path):
    """Convert model to GGUF format"""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = model_path.name
    
    # Step 1: Convert to GGUF F16
    print(f"\n📝 Converting {model_name} to GGUF F16...")
    f16_path = output_dir / f"{model_name}.gguf"
    
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(f16_path),
        "--outtype", "f16"
    ], check=True)
    
    print(f"✅ F16 model saved to: {f16_path}")
    
    # Step 2: Quantize if requested
    if quantization != "F16":
        print(f"\n🔢 Quantizing to {quantization}...")
        quantized_path = output_dir / f"{model_name}.{quantization}.gguf"
        
        quantize_exe = llama_cpp_path / "llama-quantize"
        subprocess.run([
            str(quantize_exe),
            str(f16_path),
            str(quantized_path),
            quantization
        ], check=True)
        
        print(f"✅ Quantized model saved to: {quantized_path}")
        
        # Calculate size reduction
        original_size = f16_path.stat().st_size / (1024**3)  # GB
        quantized_size = quantized_path.stat().st_size / (1024**3)  # GB
        reduction = (1 - quantized_size/original_size) * 100
        
        print(f"\n📊 Size reduction:")
        print(f"  Original: {original_size:.2f} GB")
        print(f"  Quantized: {quantized_size:.2f} GB")
        print(f"  Reduction: {reduction:.1f}%")
        
        return quantized_path
    
    return f16_path

def create_modelfile(model_path, model_name="khmer-medical"):
    """Create Ollama Modelfile"""
    modelfile_content = f"""# Khmer Medical Assistant Model
FROM {model_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM "អ្នកគឺជាជំនួយការវេជ្ជសាស្ត្រដែលមានចំណេះដឹង។ សូមឆ្លើយសំណួរវេជ្ជសាស្ត្រដោយផ្តល់ព័ត៌មានត្រឹមត្រូវ និងមានប្រយោជន៍។ You are a knowledgeable medical assistant. Please answer medical questions with accurate and helpful information."

# Template
TEMPLATE """{{{{ if .System }}}}{{ .System }}{{{{ end }}}}
{{{{ if .Prompt }}}}User: {{ .Prompt }}{{{{ end }}}}
Assistant: {{ .Response }}"""

# License
LICENSE "Apache 2.0"
"""
    
    modelfile_path = Path(model_path).parent / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"\n📄 Ollama Modelfile created: {modelfile_path}")
    print(f"\nTo use with Ollama:")
    print(f"  ollama create {model_name} -f {modelfile_path}")
    print(f"  ollama run {model_name}")

def main():
    args = parse_args()
    
    print("🔄 GGUF Conversion Tool")
    print("=" * 50)
    
    # Setup llama.cpp
    llama_cpp_path = setup_llama_cpp(args.llama_cpp_path)
    
    # Convert model
    gguf_path = convert_to_gguf(
        args.model_path,
        args.output_dir,
        args.quantization,
        llama_cpp_path
    )
    
    # Create Ollama Modelfile
    create_modelfile(gguf_path)
    
    print("\n✅ Conversion complete!")
    print(f"📦 GGUF model: {gguf_path}")
    print("\n🚀 Deployment options:")
    print("  1. Ollama: ollama create khmer-medical -f Modelfile")
    print("  2. LM Studio: Import the .gguf file directly")
    print("  3. llama.cpp: ./llama-cli -m model.gguf -p 'សំណួរ:'")
    print("  4. Python: llama-cpp-python library")

if __name__ == "__main__":
    main()