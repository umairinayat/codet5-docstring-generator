"""
CodeT5 Docstring Generator REST API
Flask-based inference endpoint for production deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import re
import logging
from typing import Dict, Optional
import time


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests


class DocstringAPI:
    """API handler for docstring generation"""
    
    def __init__(self, model_path: str = "./codet5-docstring-model"):
        self.model_path = model_path
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _setup_device(self) -> torch.device:
        """Setup GPU/CPU device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🔥 Using GPU: {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.info("⚠️  Using CPU")
        return device
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"📦 Loading model from {self.model_path}")
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def remove_docstring(self, code: str) -> str:
        """Remove existing docstrings from code"""
        pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        return re.sub(pattern, '', code).strip()
    
    def generate(self, code: str, 
                max_length: int = 150,
                num_beams: int = 5,
                temperature: float = 0.7) -> Dict:
        """Generate docstring for given code"""
        
        try:
            start_time = time.time()
            
            # Clean code
            clean_code = self.remove_docstring(code)
            
            # Prepare input
            prompt = f"Generate docstring for Python function:\n{clean_code}"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
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
            
            # Decode
            docstring = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            
            return {
                'success': True,
                'docstring': docstring,
                'inference_time': f"{inference_time:.3f}s",
                'model': self.model_path
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# Initialize API
api = DocstringAPI()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': api.model_path,
        'device': str(api.device)
    })


@app.route('/generate', methods=['POST'])
def generate_docstring():
    """
    Generate docstring endpoint
    
    Request body:
    {
        "code": "def function():\n    pass",
        "max_length": 150,      # optional
        "num_beams": 5,         # optional
        "temperature": 0.7      # optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'code' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "code" field in request'
            }), 400
        
        code = data['code']
        max_length = data.get('max_length', 150)
        num_beams = data.get('num_beams', 5)
        temperature = data.get('temperature', 0.7)
        
        # Validate parameters
        if not isinstance(code, str) or not code.strip():
            return jsonify({
                'success': False,
                'error': 'Code must be a non-empty string'
            }), 400
        
        # Generate docstring
        result = api.generate(
            code=code,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def batch_generate():
    """
    Batch generation endpoint
    
    Request body:
    {
        "codes": ["def func1():\n    pass", "def func2():\n    pass"],
        "max_length": 150,
        "num_beams": 5,
        "temperature": 0.7
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'codes' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "codes" field in request'
            }), 400
        
        codes = data['codes']
        max_length = data.get('max_length', 150)
        num_beams = data.get('num_beams', 5)
        temperature = data.get('temperature', 0.7)
        
        if not isinstance(codes, list):
            return jsonify({
                'success': False,
                'error': 'codes must be a list'
            }), 400
        
        # Generate for each code snippet
        results = []
        for code in codes:
            result = api.generate(
                code=code,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature
            )
            results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch request error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/example', methods=['GET'])
def get_example():
    """Get example usage"""
    return jsonify({
        'endpoint': '/generate',
        'method': 'POST',
        'example_request': {
            'code': '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
            'max_length': 150,
            'num_beams': 5,
            'temperature': 0.7
        },
        'example_response': {
            'success': True,
            'docstring': 'Calculate the nth Fibonacci number using recursion.',
            'inference_time': '0.245s',
            'model': './codet5-docstring-model'
        }
    })


def main():
    """Run the Flask server"""
    print("\n" + "=" * 70)
    print("CodeT5 Docstring Generator API Server")
    print("=" * 70)
    print(f"\n🚀 Server starting...")
    print(f"   ├─ Model: {api.model_path}")
    print(f"   ├─ Device: {api.device}")
    print(f"   └─ Endpoints:")
    print(f"      ├─ GET  /health   - Health check")
    print(f"      ├─ POST /generate - Generate docstring")
    print(f"      ├─ POST /batch    - Batch generation")
    print(f"      └─ GET  /example  - Usage example")
    print(f"\n📡 Server will run on http://0.0.0.0:5000")
    print("=" * 70 + "\n")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
