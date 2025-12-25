"""
Quick Demo Script for CodeT5 Docstring Generator
Test the model with example code snippets
"""

import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import re


def check_gpu():
    """Display GPU information"""
    print("\n" + "=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)
    
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"\n CUDA Available: {torch.cuda.is_available()}")
        print(f"   ├─ Number of GPUs: {n_gpu}")
        
        for i in range(n_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   ├─ GPU {i}: {gpu_name}")
            print(f"   │  └─ Memory: {gpu_memory:.2f} GB")
        
        print(f"   └─ Current Device: {torch.cuda.current_device()}")
    else:
        print("\n⚠️  No GPU available - using CPU")
    
    print("=" * 70 + "\n")


def load_model(model_path: str = "./codet5-docstring-model"):
    """Load trained model"""
    print(f"📦 Loading model from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        return tokenizer, model, device
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tip: Make sure you've trained the model first using:")
        print("   python codet5_docstring_generator.py")
        return None, None, None


def generate_docstring(code: str, tokenizer, model, device):
    """Generate docstring for code snippet"""
    
    # Remove existing docstrings
    pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
    clean_code = re.sub(pattern, '', code).strip()
    
    # Prepare input
    prompt = f"Generate docstring for Python function:\n{clean_code}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=5,
            temperature=0.7,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode
    docstring = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return docstring


def run_examples(tokenizer, model, device):
    """Run several example generations"""
    
    examples = [
        {
            "name": "Binary Search",
            "code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
        },
        {
            "name": "Fibonacci",
            "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        },
        {
            "name": "Quick Sort",
            "code": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        },
        {
            "name": "Data Processing",
            "code": """
def process_data(data, threshold=0.5):
    filtered = [x for x in data if x > threshold]
    normalized = [x / max(filtered) for x in filtered]
    return normalized
"""
        },
        {
            "name": "Matrix Multiplication",
            "code": """
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Incompatible dimensions")
    
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result
"""
        }
    ]
    
    print("\n" + "=" * 70)
    print("EXAMPLE GENERATIONS")
    print("=" * 70)
    
    for idx, example in enumerate(examples, 1):
        print(f"\n{'─' * 70}")
        print(f"Example {idx}: {example['name']}")
        print(f"{'─' * 70}")
        
        print("\n📝 Input Code:")
        print(example['code'])
        
        docstring = generate_docstring(example['code'], tokenizer, model, device)
        
        print("\n✨ Generated Docstring:")
        print(f'"""{docstring}"""')
        
        print(f"\n{'─' * 70}")
    
    print("\n" + "=" * 70)


def interactive_mode(tokenizer, model, device):
    """Interactive mode for custom code input"""
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\n📝 Enter your Python code (type 'END' on a new line to finish)")
    print("   Type 'quit' to exit interactive mode\n")
    
    while True:
        lines = []
        print(">>> ", end="")
        
        while True:
            line = input()
            if line.strip() == 'END':
                break
            if line.strip().lower() == 'quit':
                return
            lines.append(line)
        
        code = '\n'.join(lines)
        
        if not code.strip():
            print("⚠️  Empty code provided. Try again.\n")
            continue
        
        print("\n🔮 Generating docstring...")
        docstring = generate_docstring(code, tokenizer, model, device)
        
        print("\n✨ Generated Docstring:")
        print(f'"""{docstring}"""')
        print("\n" + "─" * 70 + "\n")


def main():
    """Main demo execution"""
    
    print("\n" + "=" * 70)
    print("CodeT5 Docstring Generator - Quick Demo")
    print("=" * 70)
    
    # Check GPU
    check_gpu()
    
    # Load model
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        return
    
    # Menu
    while True:
        print("\n" + "=" * 70)
        print("MENU")
        print("=" * 70)
        print("\n1. Run Example Generations")
        print("2. Interactive Mode (Enter Custom Code)")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            run_examples(tokenizer, model, device)
        
        elif choice == '2':
            interactive_mode(tokenizer, model, device)
        
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n⚠️  Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
