"""
Evaluation Module for CodeT5 Docstring Generator
Implements BLEU, ROUGE, and CodeBLEU metrics
"""

import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import json


class DocstringEvaluator:
    """Comprehensive evaluation for docstring generation models"""
    
    def __init__(self, model_path: str, device: str = None):
        # GPU setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Loading model from {model_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.bleu = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def generate_docstring(self, code: str) -> str:
        """Generate docstring for given code"""
        prompt = f"Generate docstring for Python function:\n{code}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                num_beams=5,
                temperature=0.7,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BLEU scores"""
        # BLEU expects references as list of lists
        refs = [[ref] for ref in references]
        
        bleu_score = self.bleu.corpus_score(predictions, refs)
        
        return {
            'bleu': bleu_score.score,
            'bleu_1': bleu_score.precisions[0],
            'bleu_2': bleu_score.precisions[1],
            'bleu_3': bleu_score.precisions[2],
            'bleu_4': bleu_score.precisions[3]
        }
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores"""
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy"""
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip().lower() == ref.strip().lower())
        return matches / len(predictions) * 100
    
    def evaluate_on_dataset(self, dataset_name: str = "code_x_glue_ct_code_to_text",
                           split: str = "test",
                           num_samples: int = 1000) -> Dict:
        """Evaluate model on test dataset"""
        
        print(f"\n📊 Evaluating on {dataset_name} ({split} split)")
        print(f"   └─ Number of samples: {num_samples}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, "python")
        test_data = dataset[split]
        
        # Limit samples if needed
        if num_samples < len(test_data):
            test_data = test_data.select(range(num_samples))
        
        predictions = []
        references = []
        
        print("\n🔮 Generating predictions...")
        for item in tqdm(test_data, desc="Generating"):
            if item['docstring'] and item['code']:
                # Generate prediction
                pred = self.generate_docstring(item['code'])
                predictions.append(pred)
                references.append(item['docstring'])
        
        print("\n📈 Computing metrics...")
        
        # Compute all metrics
        bleu_scores = self.compute_bleu(predictions, references)
        rouge_scores = self.compute_rouge(predictions, references)
        exact_match = self.compute_exact_match(predictions, references)
        
        results = {
            **bleu_scores,
            **rouge_scores,
            'exact_match': exact_match,
            'num_samples': len(predictions)
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n📊 BLEU Scores:")
        print(f"   ├─ BLEU: {results['bleu']:.4f}")
        print(f"   ├─ BLEU-1: {results['bleu_1']:.4f}")
        print(f"   ├─ BLEU-2: {results['bleu_2']:.4f}")
        print(f"   ├─ BLEU-3: {results['bleu_3']:.4f}")
        print(f"   └─ BLEU-4: {results['bleu_4']:.4f}")
        
        print(f"\n📊 ROUGE Scores:")
        print(f"   ├─ ROUGE-1: {results['rouge1']:.4f}")
        print(f"   ├─ ROUGE-2: {results['rouge2']:.4f}")
        print(f"   └─ ROUGE-L: {results['rougeL']:.4f}")
        
        print(f"\n📊 Additional Metrics:")
        print(f"   ├─ Exact Match: {results['exact_match']:.2f}%")
        print(f"   └─ Samples Evaluated: {results['num_samples']}")
        
        print("\n" + "=" * 70)
    
    def save_results(self, results: Dict, output_file: str = "evaluation_results.json"):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Results saved to {output_file}")


def compare_models(model_paths: List[str], dataset_name: str = "code_x_glue_ct_code_to_text"):
    """Compare multiple model checkpoints"""
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    all_results = {}
    
    for model_path in model_paths:
        print(f"\n🔍 Evaluating: {model_path}")
        evaluator = DocstringEvaluator(model_path)
        results = evaluator.evaluate_on_dataset(dataset_name, num_samples=500)
        all_results[model_path] = results
        evaluator.print_results(results)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    
    print(f"\n{'Model':<40} {'BLEU':<10} {'ROUGE-L':<10} {'Exact Match':<15}")
    print("-" * 75)
    
    for model_path, results in all_results.items():
        model_name = model_path.split('/')[-1]
        print(f"{model_name:<40} {results['bleu']:<10.4f} {results['rougeL']:<10.4f} {results['exact_match']:<15.2f}%")
    
    return all_results


def main():
    """Main evaluation pipeline"""
    
    # Example: Evaluate trained model
    model_path = "./codet5-docstring-model"
    
    evaluator = DocstringEvaluator(model_path)
    
    # Evaluate on test set
    results = evaluator.evaluate_on_dataset(
        dataset_name="code_x_glue_ct_code_to_text",
        split="test",
        num_samples=1000
    )
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results)
    
    # Example inference
    print("\n" + "=" * 70)
    print("EXAMPLE GENERATION")
    print("=" * 70)
    
    test_code = """
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
    
    print("\n📝 Input Code:")
    print(test_code)
    
    docstring = evaluator.generate_docstring(test_code)
    
    print("\n✨ Generated Docstring:")
    print(f'"""{docstring}"""')


if __name__ == "__main__":
    main()
