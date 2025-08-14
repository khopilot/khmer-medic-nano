#!/bin/bash
# Run script for JarvisLabs.ai GPU instances
# Optimized for RTX 3090/4090 or A100

echo "🚀 Khmer Medical Model Fine-tuning on JarvisLabs"
echo "================================================"

# Detect GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Setup environment
echo -e "\n📦 Setting up environment..."
python -m venv venv_finetune
source venv_finetune/bin/activate

# Install dependencies
echo -e "\n📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download dataset if not present
if [ ! -d "data/training_formats" ]; then
    echo -e "\n📥 Downloading dataset from HuggingFace..."
    python -c "
from datasets import load_dataset
dataset = load_dataset('khopilot/khmer-medical-qa')
print(f'Dataset loaded: {len(dataset[\"train\"])} examples')
"
fi

# Select model based on available VRAM
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo -e "\n🎮 Available VRAM: ${VRAM} MB"

if [ $VRAM -lt 8000 ]; then
    echo "⚠️  Low VRAM detected. Using SmolLM2 with 4-bit quantization..."
    MODEL="smollm"
    USE_4BIT="--use_4bit"
    BATCH_SIZE=2
elif [ $VRAM -lt 16000 ]; then
    echo "✅ Medium VRAM detected. Using Qwen 2.5 1.5B with 4-bit..."
    MODEL="qwen"
    USE_4BIT="--use_4bit"
    BATCH_SIZE=4
else
    echo "🎉 High VRAM detected! Using Qwen 2.5 1.5B full precision..."
    MODEL="qwen"
    USE_4BIT=""
    BATCH_SIZE=8
fi

# Create models directory
mkdir -p models

# Run training
echo -e "\n🏃 Starting training with $MODEL model..."
echo "Configuration:"
echo "  - Model: $MODEL"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Quantization: ${USE_4BIT:-none}"
echo ""

if [ "$MODEL" = "qwen" ]; then
    python train_qwen.py \
        --batch_size $BATCH_SIZE \
        --num_epochs 3 \
        --learning_rate 2e-5 \
        --gradient_checkpointing \
        --use_flash_attn \
        $USE_4BIT \
        --wandb_project khmer-medical
else
    python train_smollm.py \
        --batch_size $BATCH_SIZE \
        --num_epochs 4 \
        --learning_rate 3e-5 \
        $USE_4BIT
fi

# Test the model
echo -e "\n🧪 Testing the fine-tuned model..."
python test_model.py --model_path models/khmer-medical-$MODEL

echo -e "\n✅ Training complete!"
echo "Model saved to: models/khmer-medical-$MODEL"

# Optional: Convert to GGUF for deployment
read -p "Convert to GGUF format for deployment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n🔄 Converting to GGUF..."
    python convert_to_gguf.py models/khmer-medical-$MODEL
fi

echo -e "\n🎉 All done! Your model is ready for deployment."