# Claude Code Instructions for JarvisLabs Training

You are helping to fine-tune a Khmer medical language model on a JarvisLabs.ai GPU instance. Follow these instructions carefully to complete the training successfully.

## 🎯 Your Mission

Fine-tune either Qwen 2.5 1.5B or SmolLM2 1.7B on the Khmer medical dataset to create a specialized medical Q&A model for Cambodian healthcare applications.

## 📋 Task Checklist

Use the TodoWrite tool to track these tasks:
1. [ ] Check GPU specifications and available VRAM
2. [ ] Clone the repository from GitHub
3. [ ] Set up Python environment
4. [ ] Install all dependencies
5. [ ] Verify dataset accessibility
6. [ ] Select appropriate model based on VRAM
7. [ ] Configure training parameters
8. [ ] Start training process
9. [ ] Monitor training progress
10. [ ] Test the fine-tuned model
11. [ ] Convert to GGUF format (optional)
12. [ ] Upload to HuggingFace (optional)

## 🚀 Step-by-Step Instructions

### Step 1: System Check
First, check the GPU and system specifications:
```bash
# Check GPU
nvidia-smi

# Check available disk space
df -h

# Check Python version
python --version

# Check current directory
pwd
```

Expected: You should see an RTX 3090/4090 or A100 GPU with 24-40GB VRAM.

### Step 2: Clone Repository
```bash
# Clone the training repository
git clone https://github.com/khopilot/khmer-medic-nano.git
cd khmer-medic-nano

# List contents to verify
ls -la
```

### Step 3: Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv_training
source venv_training/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

If you encounter CUDA errors, try:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Dataset Access
```bash
# Test dataset loading from HuggingFace
python -c "
from datasets import load_dataset
dataset = load_dataset('khopilot/khmer-medical-qa')
print(f'Dataset loaded successfully!')
print(f'Training examples: {len(dataset[\"train\"])}')
print(f'First example: {dataset[\"train\"][0]}')
"
```

### Step 5: Choose Model Based on VRAM

Check available VRAM and choose accordingly:
```bash
# Get VRAM in MB
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

Decision tree:
- **< 12GB VRAM**: Use SmolLM2 with 4-bit quantization
- **12-24GB VRAM**: Use Qwen 2.5 1.5B with 4-bit quantization
- **> 24GB VRAM**: Use Qwen 2.5 1.5B full precision

### Step 6: Run Training

#### Option A: Automated Script (Recommended)
```bash
# Make script executable
chmod +x run_jarvislab.sh

# Run automated training
./run_jarvislab.sh
```

This script will automatically detect GPU and configure training.

#### Option B: Manual Training for Qwen 2.5
```bash
python train_qwen.py \
    --model_id "Qwen/Qwen2.5-1.5B" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --lora_r 32 \
    --use_4bit \
    --use_flash_attn \
    --gradient_checkpointing \
    --output_dir models/khmer-medical-qwen
```

#### Option C: Manual Training for SmolLM2
```bash
python train_smollm.py \
    --model_id "HuggingFaceTB/SmolLM2-1.7B" \
    --num_epochs 4 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --lora_r 32 \
    --use_4bit \
    --output_dir models/khmer-medical-smollm
```

### Step 7: Monitor Training

Training will show progress like:
```
{'loss': 2.3456, 'learning_rate': 1.5e-05, 'epoch': 0.5}
{'eval_loss': 1.9876, 'eval_runtime': 120.5, 'epoch': 1.0}
```

Expected training times:
- RTX 3090: 2-3 hours
- RTX 4090: 1.5-2 hours  
- A100: 1-1.5 hours

Watch for:
- Loss should decrease steadily
- Eval loss should be lower than training loss
- No out of memory (OOM) errors

### Step 8: Test the Model
```bash
# Interactive testing
python test_model.py \
    --model_path models/khmer-medical-qwen \
    --interactive

# Batch testing with sample questions
python test_model.py \
    --model_path models/khmer-medical-qwen
```

Test with these Khmer medical questions:
1. "តើជំងឺទឹកនោមផ្អែមប្រភេទទី២ មានរោគសញ្ញាអ្វីខ្លះ?"
2. "តើគួរធ្វើដូចម្តេច ប្រសិនបើមានគ្រុនក្តៅខ្លាំង?"
3. "តើអាហារអ្វីខ្លះដែលល្អសម្រាប់អ្នកមានជំងឺលើសឈាម?"

### Step 9: Convert to GGUF (Optional)
For deployment with llama.cpp/Ollama:
```bash
python convert_to_gguf.py models/khmer-medical-qwen \
    --quantization Q4_K_M \
    --output_dir models/gguf
```

### Step 10: Upload to HuggingFace (Optional)
```bash
# Set your HF token
export HF_TOKEN="your-token-here"

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Upload model
huggingface-cli upload your-username/khmer-medical-qwen \
    models/khmer-medical-qwen . \
    --repo-type model
```

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size and enable more aggressive optimization
python train_qwen.py \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --use_4bit \
    --gradient_checkpointing
```

### CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch for specific CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training
- Enable Flash Attention (requires Ampere GPU or newer)
- Reduce max_length to 1536 or 1024
- Use bf16 instead of fp16

### Model Not Learning
- Increase learning rate to 5e-5
- Train for more epochs (5-10)
- Check if dataset is loading correctly

## 📊 Success Criteria

Your training is successful when:
1. ✅ Final eval_loss < 2.0
2. ✅ Model generates coherent Khmer medical responses
3. ✅ Test questions produce relevant answers
4. ✅ No training crashes or OOM errors
5. ✅ Model saved successfully to `models/` directory

## 📈 Expected Metrics

After successful training:
- **Training loss**: Should decrease from ~3.5 to ~1.5
- **Eval loss**: Should stabilize around 1.5-2.0
- **Perplexity**: Should be < 10
- **Response quality**: Grammatically correct Khmer with medical accuracy

## 💾 Output Files

After training, you should have:
```
models/
├── khmer-medical-qwen/        # or khmer-medical-smollm/
│   ├── adapter_config.json    # LoRA configuration
│   ├── adapter_model.bin      # LoRA weights
│   ├── tokenizer_config.json  # Tokenizer config
│   ├── special_tokens_map.json
│   └── vocab.json
└── gguf/                       # (Optional) GGUF conversions
    └── khmer-medical.Q4_K_M.gguf
```

## 🎯 Final Steps

1. **Save training logs**: Copy the console output to a file
2. **Document results**: Note final loss, training time, and any issues
3. **Test thoroughly**: Run at least 10 test questions
4. **Package model**: Create a ZIP file of the models directory
5. **Report success**: Confirm training completion with metrics

## 💡 Pro Tips

1. **Use tmux/screen**: Prevent training interruption if connection drops
   ```bash
   tmux new -s training
   # Run training
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training
   ```

2. **Monitor GPU usage**: Keep `nvidia-smi -l 1` running in another terminal

3. **Save checkpoints**: Training auto-saves every 500 steps

4. **Test early**: Run test_model.py after first epoch to verify learning

5. **Document everything**: Keep notes of what worked and what didn't

## ⚠️ Important Notes

- **DO NOT** interrupt training unless absolutely necessary
- **DO NOT** change hyperparameters mid-training
- **DO** save all console outputs for debugging
- **DO** test the model before declaring success
- **DO** verify VRAM usage stays below 95%

## 🆘 If Something Goes Wrong

1. First, check error messages carefully
2. Look for OOM (out of memory) errors → reduce batch_size
3. Check for CUDA errors → verify PyTorch installation
4. Training not starting → check dataset download
5. Model not learning → adjust learning rate

Remember: The goal is to create a functional Khmer medical Q&A model. Quality matters more than speed!

---

**You've got this! The setup is tested and ready. Just follow the steps methodically and the training will succeed.** 🚀