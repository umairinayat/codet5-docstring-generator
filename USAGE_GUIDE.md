# Quick Start Guide - CodeT5 Docstring Generator

## 📋 Table of Contents
1. [Installation](#installation)
2. [Testing Your Setup](#testing-your-setup)
3. [Training the Model](#training-the-model)
4. [Using the Model](#using-the-model)
5. [API Deployment](#api-deployment)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Step 1: Clone/Download the Project
```bash
cd your-project-directory
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Testing Your Setup

### Quick System Test
```bash
python test_system.py
```

This will check:
- ✅ Python version
- ✅ All dependencies
- ✅ GPU/CUDA availability
- ✅ System resources
- ✅ HuggingFace access

### Quick CUDA Test Only
```bash
python test_system.py --quick
```

### Auto-Install Missing Packages
```bash
python test_system.py --install
```

---

## 🎯 Training the Model

### Option 1: Default Training (Recommended for RTX 3080)
```bash
python codet5_docstring_generator.py
```

**Expected Output:**
```
🔥 GPU Detection:
   ├─ Number of GPUs available: 1
   ├─ GPU 0: NVIDIA GeForce RTX 3080 (10.00 GB)
   └─ Single GPU training on NVIDIA GeForce RTX 3080

📦 Loading dataset...
🔧 Preprocessing training data...
   ├─ Training samples: 251,820
   └─ Validation samples: 13,914

🤖 Loading CodeT5 model...
   └─ Model loaded: Salesforce/codet5-base
   └─ Total parameters: 220.00M

🚀 Starting training...
   ├─ Epochs: 10
   ├─ Batch size per device: 8
   ├─ Effective batch size: 16
   └─ Total training steps: 157,387
```

### Option 2: Custom Training Configuration

#### Quick Test (for debugging)
```python
from config import get_config, print_config
from codet5_docstring_generator import CodeT5DocstringGenerator

# Load quick test config
config = get_config("quick_test")
print_config(config)

# Train with config
generator = CodeT5DocstringGenerator(
    model_name=config.model.model_name
)
# ... continue with custom parameters
```

#### Multi-GPU Training
```python
# Automatically detected if you have multiple GPUs
# No code changes needed!
```

---

## 💡 Using the Model

### Method 1: Python Script
```python
from codet5_docstring_generator import CodeT5DocstringGenerator

# Initialize
generator = CodeT5DocstringGenerator()
generator.setup_model()

# Your code
code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""

# Generate docstring
docstring = generator.generate_docstring(code)
print(f'"""{docstring}"""')
```

### Method 2: Interactive Demo
```bash
python demo.py
```

**Menu Options:**
1. Run Example Generations
2. Interactive Mode (Enter Custom Code)
3. Exit

**Interactive Mode Example:**
```
>>> def my_function(x, y):
...     return x + y
... END

✨ Generated Docstring:
"""Add two values and return their sum."""
```

---

## 🌐 API Deployment

### Start the API Server
```bash
python api_server.py
```

**Server Info:**
```
🚀 Server starting...
   ├─ Model: ./codet5-docstring-model
   ├─ Device: cuda
   └─ Endpoints:
      ├─ GET  /health   - Health check
      ├─ POST /generate - Generate docstring
      ├─ POST /batch    - Batch generation
      └─ GET  /example  - Usage example

📡 Server will run on http://0.0.0.0:5000
```

### API Usage Examples

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Generate Docstring
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a + b",
    "max_length": 150,
    "num_beams": 5,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
    "success": true,
    "docstring": "Add two numbers and return the result.",
    "inference_time": "0.245s",
    "model": "./codet5-docstring-model"
}
```

#### Batch Generation
```bash
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "codes": [
      "def func1():\n    pass",
      "def func2():\n    pass"
    ]
  }'
```

---

## 📊 Evaluation

### Evaluate Trained Model
```bash
python evaluate_model.py
```

**Expected Output:**
```
📊 BLEU Scores:
   ├─ BLEU: 42.5000
   ├─ BLEU-1: 58.3000
   ├─ BLEU-2: 45.2000
   ├─ BLEU-3: 38.7000
   └─ BLEU-4: 32.1000

📊 ROUGE Scores:
   ├─ ROUGE-1: 0.5830
   ├─ ROUGE-2: 0.3870
   └─ ROUGE-L: 0.5420

📊 Additional Metrics:
   ├─ Exact Match: 15.80%
   └─ Samples Evaluated: 1000
```

---

## 🔧 Troubleshooting

### Problem: CUDA Out of Memory

**Solution 1: Reduce Batch Size**
```python
# In codet5_docstring_generator.py
batch_size=4  # Instead of 8
```

**Solution 2: Enable Gradient Checkpointing** (Already enabled)
```python
gradient_checkpointing=True
```

**Solution 3: Reduce Sequence Length**
```python
max_length=256  # Instead of 512
```

### Problem: Slow Training

**Check GPU Usage:**
```bash
nvidia-smi
# Should show ~90%+ GPU utilization
```

**Solutions:**
- ✅ Mixed precision is already enabled (FP16)
- ✅ Increase batch size if GPU has headroom
- ✅ Ensure dataloader workers are set correctly

### Problem: Import Errors

**Solution:**
```bash
# Reinstall transformers
pip uninstall transformers
pip install transformers==4.35.0

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

### Problem: Model Not Found

**Solution:**
```bash
# Make sure you've trained the model first
python codet5_docstring_generator.py

# Or check the model path
ls -la ./codet5-docstring-model/
```

---

## 📈 Performance Tips

### For RTX 3080 (10GB VRAM)
- ✅ Batch size: 8
- ✅ FP16 enabled
- ✅ Gradient accumulation: 2
- ✅ Gradient checkpointing: True

### For Multi-GPU Setup
- ✅ Automatic detection
- ✅ DataParallel wrapping
- ✅ Effective batch size multiplied

### For CPU Training
- ⚠️ Disable FP16
- ⚠️ Reduce batch size to 2-4
- ⚠️ Expect 10-20x slower training

---

## 📚 File Structure

```
codet5-docstring-generator/
├── codet5_docstring_generator.py  # Main training script ⭐
├── evaluate_model.py               # Evaluation metrics
├── api_server.py                   # REST API server
├── demo.py                         # Interactive demo
├── config.py                       # Configuration management
├── data_utils.py                   # Data preprocessing
├── test_system.py                  # System verification
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
└── USAGE_GUIDE.md                 # This file
```

---

## 🎓 Next Steps

1. ✅ **Test Your Setup**: `python test_system.py`
2. ✅ **Train the Model**: `python codet5_docstring_generator.py`
3. ✅ **Try the Demo**: `python demo.py`
4. ✅ **Evaluate Results**: `python evaluate_model.py`
5. ✅ **Deploy API**: `python api_server.py`

---

## 💡 Tips for Portfolio

- 📸 Take screenshots of training progress
- 📊 Include evaluation metrics in your presentation
- 🎥 Record a demo video showing inference
- 📝 Document any modifications you made
- 🔬 Add your own experiments/improvements

---

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_system.py` for diagnostics
3. Review the full README.md
4. Check HuggingFace documentation

---

**Good luck with your portfolio project! 🚀**
