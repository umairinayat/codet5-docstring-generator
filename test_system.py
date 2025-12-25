"""
System Test Script for CodeT5 Docstring Generator
Verify installation, dependencies, and GPU availability
"""

import sys
import subprocess
from typing import Dict, List, Tuple


class SystemTester:
    """Test system setup and dependencies"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def test_python_version(self) -> Tuple[bool, str]:
        """Test Python version"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 8:
            return True, f"[PASS] Python {version_str} (OK)"
        else:
            return False, f"[FAIL] Python {version_str} (Requires >= 3.8)"
    
    def test_package_import(self, package_name: str, import_name: str = None) -> Tuple[bool, str]:
        """Test if package can be imported"""
        if import_name is None:
            import_name = package_name
        
        try:
            __import__(import_name)
            return True, f"[PASS] {package_name}"
        except ImportError as e:
            return False, f"[FAIL] {package_name} - {str(e)}"
    
    def test_torch_cuda(self) -> Tuple[bool, str]:
        """Test PyTorch CUDA availability"""
        try:
            import torch
            
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
                
                result = f"[PASS] CUDA Available\n"
                result += f"   ├─ PyTorch version: {torch.__version__}\n"
                result += f"   ├─ CUDA version: {torch.version.cuda}\n"
                result += f"   ├─ Number of GPUs: {n_gpu}\n"
                
                for i, name in enumerate(gpu_names):
                    memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    result += f"   ├─ GPU {i}: {name} ({memory:.1f} GB)\n"
                
                result += f"   └─ Current device: {torch.cuda.current_device()}"
                
                return True, result
            else:
                return False, "[WARN] CUDA not available (will use CPU)"
                
        except Exception as e:
            return False, f"[FAIL] Error checking CUDA: {str(e)}"
    
    def test_transformers_models(self) -> Tuple[bool, str]:
        """Test if CodeT5 models are accessible"""
        try:
            from transformers import AutoTokenizer
            
            # Try to load CodeT5 tokenizer (doesn't download full model)
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
            
            return True, "[PASS] HuggingFace models accessible"
        except Exception as e:
            return False, f"[FAIL] Cannot access HuggingFace models: {str(e)}"
    
    def test_disk_space(self) -> Tuple[bool, str]:
        """Test available disk space"""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            
            if free_gb > 20:  # Need at least 20GB for models and data
                return True, f"[PASS] Disk space: {free_gb:.1f} GB free"
            else:
                return False, f"[WARN] Low disk space: {free_gb:.1f} GB free (recommend > 20GB)"
                
        except Exception as e:
            return False, f"[FAIL] Error checking disk space: {str(e)}"
    
    def test_memory(self) -> Tuple[bool, str]:
        """Test system memory"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024 ** 3)
            available_gb = memory.available / (1024 ** 3)
            
            result = f"RAM: {available_gb:.1f} GB / {total_gb:.1f} GB available"
            
            if available_gb > 8:
                return True, f"[PASS] {result}"
            else:
                return False, f"[WARN] {result} (recommend > 8GB)"
                
        except ImportError:
            return True, "[WARN] psutil not installed (cannot check memory)"
        except Exception as e:
            return False, f"[FAIL] Error checking memory: {str(e)}"
    
    def run_all_tests(self):
        """Run all system tests"""
        
        print("\n" + "=" * 70)
        print("SYSTEM TEST - CodeT5 Docstring Generator")
        print("=" * 70)
        
        # Python version
        print("\n[Python] Testing Python Environment...")
        success, msg = self.test_python_version()
        print(f"   {msg}")
        self.results['python'] = success
        
        # Core packages
        print("\n[Packages] Testing Core Dependencies...")
        
        core_packages = [
            ('torch', 'torch'),
            ('transformers', 'transformers'),
            ('datasets', 'datasets'),
            ('numpy', 'numpy'),
            ('tqdm', 'tqdm')
        ]
        
        for pkg_name, import_name in core_packages:
            success, msg = self.test_package_import(pkg_name, import_name)
            print(f"   {msg}")
            self.results[pkg_name] = success
        
        # CUDA
        print("\n[CUDA] Testing GPU/CUDA...")
        success, msg = self.test_torch_cuda()
        print(f"   {msg}")
        self.results['cuda'] = success
        
        # Evaluation packages
        print("\n[Evaluation] Testing Evaluation Dependencies...")
        
        eval_packages = [
            ('sacrebleu', 'sacrebleu'),
            ('rouge-score', 'rouge_score')
        ]
        
        for pkg_name, import_name in eval_packages:
            success, msg = self.test_package_import(pkg_name, import_name)
            print(f"   {msg}")
            self.results[pkg_name] = success
        
        # API packages
        print("\n[API] Testing API Dependencies...")
        
        api_packages = [
            ('flask', 'flask'),
            ('flask-cors', 'flask_cors')
        ]
        
        for pkg_name, import_name in api_packages:
            success, msg = self.test_package_import(pkg_name, import_name)
            print(f"   {msg}")
            self.results[pkg_name] = success
        
        # HuggingFace access
        print("\n[HuggingFace] Testing HuggingFace Access...")
        success, msg = self.test_transformers_models()
        print(f"   {msg}")
        self.results['huggingface'] = success
        
        # System resources
        print("\n[Resources] Testing System Resources...")
        
        success, msg = self.test_disk_space()
        print(f"   {msg}")
        self.results['disk'] = success
        
        success, msg = self.test_memory()
        print(f"   {msg}")
        self.results['memory'] = success
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for v in self.results.values() if v)
        failed_tests = total_tests - passed_tests
        
        print(f"\n[Results]")
        print(f"   ├─ Total Tests: {total_tests}")
        print(f"   ├─ Passed: {passed_tests}")
        print(f"   └─ Failed/Warnings: {failed_tests}")
        
        if failed_tests == 0:
            print("\n[PASS] All tests passed! System is ready.")
            print("\n[Next Steps] You can now run:")
            print("   1. python codet5_docstring_generator.py  # Train model")
            print("   2. python demo.py                         # Test inference")
            print("   3. python api_server.py                   # Start API server")
        else:
            print("\n[WARN] Some tests failed. Please install missing packages:")
            print("\n   pip install -r requirements.txt")
            
            # List failed packages
            print("\n[Failed/Missing]")
            for pkg, success in self.results.items():
                if not success:
                    print(f"   [FAIL] {pkg}")
        
        print("\n" + "=" * 70 + "\n")


def quick_torch_test():
    """Quick PyTorch CUDA test"""
    
    print("\n" + "=" * 70)
    print("QUICK TORCH CUDA TEST")
    print("=" * 70)
    
    try:
        import torch
        
        print(f"\n[PASS] PyTorch successfully imported")
        print(f"   └─ Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"\n[PASS] CUDA is available")
            print(f"   ├─ CUDA version: {torch.version.cuda}")
            print(f"   ├─ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"   ├─ GPU {i}: {torch.cuda.get_device_name(i)}")
                
                # Test tensor operation
                x = torch.randn(1000, 1000).cuda(i)
                y = torch.randn(1000, 1000).cuda(i)
                z = torch.mm(x, y)
                
                print(f"   │  └─ Test tensor operation: [PASS] Success")
            
            print(f"   └─ Current device: cuda:{torch.cuda.current_device()}")
            
        else:
            print(f"\n[WARN] CUDA is not available")
            print(f"   └─ Training will use CPU (much slower)")
        
        print("\n" + "=" * 70)
        
    except ImportError:
        print(f"\n[FAIL] PyTorch is not installed")
        print(f"   └─ Install with: pip install torch")
        print("\n" + "=" * 70)


def install_missing_packages():
    """Attempt to install missing packages"""
    
    print("\n" + "=" * 70)
    print("ATTEMPTING TO INSTALL MISSING PACKAGES")
    print("=" * 70)
    
    try:
        print("\n[Installing] Running: pip install -r requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n[PASS] Installation completed successfully")
    except Exception as e:
        print(f"\n[FAIL] Installation failed: {str(e)}")
    
    print("\n" + "=" * 70)


def main():
    """Main test execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CodeT5 environment setup")
    parser.add_argument('--quick', action='store_true', help='Quick CUDA test only')
    parser.add_argument('--install', action='store_true', help='Install missing packages')
    
    args = parser.parse_args()
    
    if args.install:
        install_missing_packages()
        return
    
    if args.quick:
        quick_torch_test()
        return
    
    # Run full test suite
    tester = SystemTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
