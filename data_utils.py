"""
Data Utilities for CodeT5 Docstring Generator
Helper functions for data preprocessing and analysis
"""

import re
import ast
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json


class CodeAnalyzer:
    """Analyze Python code structure and extract information"""
    
    @staticmethod
    def extract_function_info(code: str) -> Dict:
        """Extract function metadata from code"""
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'num_args': len(node.args.args),
                        'has_docstring': ast.get_docstring(node) is not None,
                        'docstring': ast.get_docstring(node) or "",
                        'num_lines': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                    }
                    functions.append(func_info)
            
            return {
                'num_functions': len(functions),
                'functions': functions
            }
        except:
            return {'num_functions': 0, 'functions': []}
    
    @staticmethod
    def remove_docstring(code: str) -> str:
        """Remove docstrings from code"""
        # Pattern for triple-quoted docstrings
        pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        return re.sub(pattern, '', code).strip()
    
    @staticmethod
    def extract_docstring(code: str) -> Optional[str]:
        """Extract docstring from code"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return ast.get_docstring(node)
            return None
        except:
            return None
    
    @staticmethod
    def is_valid_python(code: str) -> bool:
        """Check if code is valid Python"""
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    @staticmethod
    def count_code_lines(code: str) -> int:
        """Count non-empty, non-comment lines"""
        lines = code.split('\n')
        code_lines = [
            line.strip() 
            for line in lines 
            if line.strip() and not line.strip().startswith('#')
        ]
        return len(code_lines)


class DatasetStatistics:
    """Calculate statistics for docstring dataset"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    def analyze_dataset(self, data: List[Dict]) -> Dict:
        """Compute comprehensive dataset statistics"""
        
        stats = {
            'total_samples': len(data),
            'code_lengths': [],
            'docstring_lengths': [],
            'num_functions': [],
            'has_valid_python': 0,
            'has_docstring': 0,
            'function_names': Counter(),
            'num_args_distribution': Counter()
        }
        
        for item in data:
            code = item.get('code', '')
            docstring = item.get('docstring', '')
            
            # Code length
            code_length = len(code.split())
            stats['code_lengths'].append(code_length)
            
            # Docstring length
            docstring_length = len(docstring.split())
            stats['docstring_lengths'].append(docstring_length)
            
            # Function analysis
            func_info = self.analyzer.extract_function_info(code)
            stats['num_functions'].append(func_info['num_functions'])
            
            for func in func_info['functions']:
                stats['function_names'][func['name']] += 1
                stats['num_args_distribution'][func['num_args']] += 1
            
            # Validity checks
            if self.analyzer.is_valid_python(code):
                stats['has_valid_python'] += 1
            
            if docstring:
                stats['has_docstring'] += 1
        
        # Compute summary statistics
        import numpy as np
        
        stats['code_length_stats'] = {
            'min': int(np.min(stats['code_lengths'])),
            'max': int(np.max(stats['code_lengths'])),
            'mean': float(np.mean(stats['code_lengths'])),
            'median': float(np.median(stats['code_lengths'])),
            'std': float(np.std(stats['code_lengths']))
        }
        
        stats['docstring_length_stats'] = {
            'min': int(np.min(stats['docstring_lengths'])),
            'max': int(np.max(stats['docstring_lengths'])),
            'mean': float(np.mean(stats['docstring_lengths'])),
            'median': float(np.median(stats['docstring_lengths'])),
            'std': float(np.std(stats['docstring_lengths']))
        }
        
        stats['valid_python_ratio'] = stats['has_valid_python'] / stats['total_samples']
        stats['has_docstring_ratio'] = stats['has_docstring'] / stats['total_samples']
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Pretty print dataset statistics"""
        
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        
        print(f"\n📊 General Statistics:")
        print(f"   ├─ Total Samples: {stats['total_samples']:,}")
        print(f"   ├─ Valid Python: {stats['has_valid_python']:,} ({stats['valid_python_ratio']:.1%})")
        print(f"   └─ Has Docstring: {stats['has_docstring']:,} ({stats['has_docstring_ratio']:.1%})")
        
        print(f"\n📏 Code Length Statistics (words):")
        code_stats = stats['code_length_stats']
        print(f"   ├─ Min: {code_stats['min']}")
        print(f"   ├─ Max: {code_stats['max']}")
        print(f"   ├─ Mean: {code_stats['mean']:.1f}")
        print(f"   ├─ Median: {code_stats['median']:.1f}")
        print(f"   └─ Std Dev: {code_stats['std']:.1f}")
        
        print(f"\n📝 Docstring Length Statistics (words):")
        doc_stats = stats['docstring_length_stats']
        print(f"   ├─ Min: {doc_stats['min']}")
        print(f"   ├─ Max: {doc_stats['max']}")
        print(f"   ├─ Mean: {doc_stats['mean']:.1f}")
        print(f"   ├─ Median: {doc_stats['median']:.1f}")
        print(f"   └─ Std Dev: {doc_stats['std']:.1f}")
        
        print(f"\n🔧 Function Statistics:")
        print(f"   ├─ Most Common Functions:")
        for func_name, count in stats['function_names'].most_common(5):
            print(f"   │  ├─ {func_name}: {count}")
        
        print(f"   └─ Argument Distribution:")
        for num_args, count in sorted(stats['num_args_distribution'].items()):
            print(f"      ├─ {num_args} args: {count}")
        
        print("\n" + "=" * 70)


class DataCleaner:
    """Clean and preprocess code-docstring pairs"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    def clean_code(self, code: str) -> str:
        """Clean code by removing extra whitespace and comments"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')]
            
            # Remove trailing whitespace
            line = line.rstrip()
            
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_docstring(self, docstring: str) -> str:
        """Clean docstring by normalizing whitespace"""
        # Remove extra whitespace
        docstring = ' '.join(docstring.split())
        
        # Remove common prefixes
        prefixes = ['def ', 'function ', 'method ']
        for prefix in prefixes:
            if docstring.lower().startswith(prefix):
                docstring = docstring[len(prefix):]
        
        return docstring.strip()
    
    def is_valid_pair(self, code: str, docstring: str,
                     min_code_length: int = 5,
                     max_code_length: int = 500,
                     min_docstring_length: int = 3,
                     max_docstring_length: int = 200) -> bool:
        """Check if code-docstring pair is valid"""
        
        # Check Python validity
        if not self.analyzer.is_valid_python(code):
            return False
        
        # Check lengths
        code_words = len(code.split())
        doc_words = len(docstring.split())
        
        if code_words < min_code_length or code_words > max_code_length:
            return False
        
        if doc_words < min_docstring_length or doc_words > max_docstring_length:
            return False
        
        return True
    
    def clean_dataset(self, data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Clean entire dataset"""
        
        cleaned_data = []
        stats = {
            'original_count': len(data),
            'cleaned_count': 0,
            'removed_invalid_python': 0,
            'removed_length_issues': 0
        }
        
        for item in data:
            code = item.get('code', '')
            docstring = item.get('docstring', '')
            
            # Clean
            cleaned_code = self.clean_code(code)
            cleaned_docstring = self.clean_docstring(docstring)
            
            # Validate
            if not self.analyzer.is_valid_python(cleaned_code):
                stats['removed_invalid_python'] += 1
                continue
            
            if not self.is_valid_pair(cleaned_code, cleaned_docstring):
                stats['removed_length_issues'] += 1
                continue
            
            # Add to cleaned dataset
            cleaned_data.append({
                'code': cleaned_code,
                'docstring': cleaned_docstring
            })
            stats['cleaned_count'] += 1
        
        return cleaned_data, stats


def save_dataset(data: List[Dict], filename: str):
    """Save dataset to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {len(data)} samples to {filename}")


def load_dataset(filename: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"📦 Loaded {len(data)} samples from {filename}")
    return data


def main():
    """Example usage of data utilities"""
    
    print("\n" + "=" * 70)
    print("Data Utilities Demo")
    print("=" * 70)
    
    # Example data
    sample_data = [
        {
            'code': '''
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''',
            'docstring': 'Calculate nth Fibonacci number using recursion'
        },
        {
            'code': '''
def add(a, b):
    """Add two numbers"""
    return a + b
''',
            'docstring': 'Add two numbers and return the result'
        }
    ]
    
    # Analyze dataset
    analyzer = DatasetStatistics()
    stats = analyzer.analyze_dataset(sample_data)
    analyzer.print_statistics(stats)
    
    # Clean dataset
    cleaner = DataCleaner()
    cleaned_data, clean_stats = cleaner.clean_dataset(sample_data)
    
    print("\n📊 Cleaning Statistics:")
    print(f"   ├─ Original: {clean_stats['original_count']}")
    print(f"   ├─ Cleaned: {clean_stats['cleaned_count']}")
    print(f"   ├─ Removed (invalid Python): {clean_stats['removed_invalid_python']}")
    print(f"   └─ Removed (length issues): {clean_stats['removed_length_issues']}")


if __name__ == "__main__":
    main()
