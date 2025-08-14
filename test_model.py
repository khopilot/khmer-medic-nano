#!/usr/bin/env python3
"""
Test the fine-tuned Khmer medical model
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Test fine-tuned Khmer medical model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", default=None, help="Base model if using LoRA")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    return parser.parse_args()

def load_model(args):
    """Load the fine-tuned model"""
    print(f"🔄 Loading model from: {args.model_path}")
    
    # Check if it's a LoRA model
    if args.base_model:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    return model, tokenizer

def test_questions():
    """Sample medical questions in Khmer"""
    return [
        "តើជំងឺទឹកនោមផ្អែមប្រភេទទី២ មានរោគសញ្ញាអ្វីខ្លះ?",
        "តើគួរធ្វើដូចម្តេច ប្រសិនបើមានគ្រុនក្តៅខ្លាំង?",
        "តើអាហារអ្វីខ្លះដែលល្អសម្រាប់អ្នកមានជំងឺលើសឈាម?",
        "តើការហាត់ប្រាណប៉ុន្មានដងក្នុងមួយសប្តាហ៍គឺល្អសម្រាប់សុខភាព?",
        "តើមានវិធីអ្វីខ្លះដើម្បីការពារជំងឺគ្រុនចាញ់?",
    ]

def format_prompt(question, model_type="qwen"):
    """Format prompt based on model type"""
    if "qwen" in model_type.lower():
        return f"""<|im_start|>system
អ្នកគឺជាជំនួយការវេជ្ជសាស្ត្រដែលមានចំណេះដឹង។<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    else:  # ChatML format
        return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

def run_inference(model, tokenizer, question, args):
    """Run inference on a single question"""
    # Determine model type
    model_type = "qwen" if "qwen" in args.model_path.lower() else "chatml"
    
    # Format prompt
    prompt = format_prompt(question, model_type)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

def interactive_mode(model, tokenizer, args):
    """Interactive chat mode"""
    print("\n💬 Interactive Mode (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        question = input("\n👤 សំណួរ (Question): ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("👋 លាសិនហើយ! (Goodbye!)")
            break
        
        if not question:
            continue
        
        print("\n🤖 ចម្លើយ (Answer):")
        response = run_inference(model, tokenizer, question, args)
        print(response)

def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, args)
    else:
        # Test with sample questions
        print("\n🧪 Testing with sample medical questions...")
        print("=" * 50)
        
        for i, question in enumerate(test_questions(), 1):
            print(f"\n📝 Question {i}:")
            print(f"   {question}")
            print(f"\n💊 Answer:")
            response = run_inference(model, tokenizer, question, args)
            print(f"   {response}")
            print("-" * 50)
        
        print("\n✅ Testing complete!")
        
        # Offer interactive mode
        print("\n💡 Tip: Run with --interactive for chat mode")

if __name__ == "__main__":
    main()