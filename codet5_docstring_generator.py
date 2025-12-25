"""
CodeT5 Docstring Generator - Portfolio Implementation
Multi-GPU Training Pipeline with Automatic GPU Detection
Author: Umair
RTX 3080 Optimized
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import os
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class GPUManager:
    """Handles GPU detection and configuration for single/multi-GPU training"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.n_gpu = torch.cuda.device_count()
        self.device_ids = list(range(self.n_gpu)) if self.n_gpu > 1 else None
        
    def _setup_device(self) -> torch.device:
        """Detect and configure GPU availability"""
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f"🔥 GPU Detection:")
            print(f"   ├─ Number of GPUs available: {n_gpu}")
            
            for i in range(n_gpu):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   ├─ GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            if n_gpu > 1:
                print(f"   └─ Multi-GPU training enabled with {n_gpu} GPUs")
            else:
                print(f"   └─ Single GPU training on {torch.cuda.get_device_name(0)}")
                
            return torch.device("cuda")
        else:
            print("  No GPU available, using CPU")
            return torch.device("cpu")
    
    def get_device_map(self) -> str:
        """Returns device map strategy for model loading"""
        if self.n_gpu > 1:
            return "auto"  # Automatic device mapping for multi-GPU
        return None
    
    def wrap_model_for_multi_gpu(self, model: nn.Module) -> nn.Module:
        """Wrap model with DataParallel for multi-GPU training"""
        if self.n_gpu > 1:
            print(f" Wrapping model with DataParallel across {self.n_gpu} GPUs")
            model = nn.DataParallel(model, device_ids=self.device_ids)
        return model.to(self.device)


class DocstringDataset(Dataset):
    """Custom Dataset for code-docstring pairs"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Prepare input (code without docstring)
        code_input = f"Generate docstring for Python function:\n{item['code']}"
        
        # Tokenize input
        model_inputs = self.tokenizer(
            code_input,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize target (docstring)
        labels = self.tokenizer(
            item['docstring'],
            max_length=150,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }


class CodeT5DocstringGenerator:
    """Main class for CodeT5-based docstring generation"""
    
    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        self.gpu_manager = GPUManager()
        self.model_name = model_name
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = None
        
    def remove_docstring(self, code: str) -> str:
        """Remove existing docstrings from code"""
        # Pattern for triple-quoted docstrings
        pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        code_without_docstring = re.sub(pattern, '', code)
        return code_without_docstring.strip()
    
    def prepare_dataset(self, dataset_name: str = "code_x_glue_ct_code_to_text"):
        """Load and prepare dataset for training"""
        print("\n📦 Loading dataset...")
        
        # Load CodeSearchNet dataset from CodeXGlue
        dataset = load_dataset(dataset_name, "python")
        
        # Prepare data
        train_data = []
        val_data = []
        
        print("🔧 Preprocessing training data...")
        for item in dataset['train']:
            if item['docstring'] and item['code']:
                clean_code = self.remove_docstring(item['code'])
                train_data.append({
                    'code': clean_code,
                    'docstring': item['docstring']
                })
        
        print("🔧 Preprocessing validation data...")
        for item in dataset['validation']:
            if item['docstring'] and item['code']:
                clean_code = self.remove_docstring(item['code'])
                val_data.append({
                    'code': clean_code,
                    'docstring': item['docstring']
                })
        
        print(f"   ├─ Training samples: {len(train_data)}")
        print(f"   └─ Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def setup_model(self):
        """Initialize CodeT5 model with GPU configuration"""
        print("\n🤖 Loading CodeT5 model...")
        
        # Load model with device map for multi-GPU
        device_map = self.gpu_manager.get_device_map()
        
        if device_map:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch.float16  # Mixed precision for RTX 3080
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name
            )
            self.model = self.gpu_manager.wrap_model_for_multi_gpu(self.model)
        
        print(f"   └─ Model loaded: {self.model_name}")
        
        # Print model size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"   └─ Total parameters: {param_count/1e6:.2f}M")
    
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              output_dir: str = "./codet5-docstring-model",
              num_epochs: int = 10,
              batch_size: int = 8):
        """Fine-tune CodeT5 model on docstring generation task"""
        
        # Create datasets
        train_dataset = DocstringDataset(train_data, self.tokenizer)
        val_dataset = DocstringDataset(val_data, self.tokenizer)
        
        # Data collator for dynamic padding
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Configure training arguments optimized for RTX 3080
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=500,
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="tensorboard",
            
            # Performance optimization for RTX 3080
            fp16=True,  # Mixed precision training
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            
            # Multi-GPU settings
            ddp_find_unused_parameters=False if self.gpu_manager.n_gpu > 1 else None,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("\n🚀 Starting training...")
        print(f"   ├─ Epochs: {num_epochs}")
        print(f"   ├─ Batch size per device: {batch_size}")
        print(f"   ├─ Effective batch size: {batch_size * self.gpu_manager.n_gpu * 2}")
        print(f"   └─ Total training steps: {len(train_dataset) // (batch_size * self.gpu_manager.n_gpu * 2) * num_epochs}")
        
        # Train the model
        trainer.train()
        
        # Save final model
        print("\n💾 Saving final model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"   └─ Model saved to {output_dir}")
        
        return trainer
    
    def generate_docstring(self, code_snippet: str, 
                          max_length: int = 150,
                          num_beams: int = 5,
                          temperature: float = 0.7) -> str:
        """Generate docstring for given code snippet"""
        
        # Remove existing docstring
        clean_code = self.remove_docstring(code_snippet)
        
        # Prepare input
        prompt = f"Generate docstring for Python function:\n{clean_code}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.gpu_manager.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode output
        docstring = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return docstring


def main():
    """Main execution flow"""
    
    print("=" * 70)
    print("CodeT5 Docstring Generator - Portfolio Implementation")
    print("=" * 70)
    
    # Initialize generator
    generator = CodeT5DocstringGenerator(model_name="Salesforce/codet5-base")
    
    # Prepare dataset
    train_data, val_data = generator.prepare_dataset()
    
    # Setup model
    generator.setup_model()
    
    # Train model
    trainer = generator.train(
        train_data=train_data[:10000],  # Use subset for faster training
        val_data=val_data[:1000],
        num_epochs=5,
        batch_size=8  # Optimized for RTX 3080 (10GB VRAM)
    )
    
    # Test inference
    print("\n" + "=" * 70)
    print("Testing Inference")
    print("=" * 70)
    
    test_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
    
    print("\n📝 Input Code:")
    print(test_code)
    
    generated_docstring = generator.generate_docstring(test_code)
    
    print("\n✨ Generated Docstring:")
    print(f'"""{generated_docstring}"""')
    
    print("\n" + "=" * 70)
    print("Training Complete! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    main()
