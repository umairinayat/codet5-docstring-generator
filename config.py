"""
Configuration file for CodeT5 Docstring Generator
Centralized parameter management for training and inference
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Model selection
    model_name: str = "Salesforce/codet5-base"
    # Options: "Salesforce/codet5-small", "Salesforce/codet5-base", "Salesforce/codet5-large"
    
    # Model variants info:
    # - codet5-small: 60M params, ~2GB VRAM
    # - codet5-base: 220M params, ~4GB VRAM  (recommended for RTX 3080)
    # - codet5-large: 770M params, ~10GB VRAM
    
    # Tokenization
    max_input_length: int = 512
    max_output_length: int = 150
    
    # Generation parameters
    num_beams: int = 5
    temperature: float = 0.7
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Dataset
    dataset_name: str = "code_x_glue_ct_code_to_text"
    dataset_language: str = "python"
    max_train_samples: Optional[int] = None  # None = use all
    max_val_samples: Optional[int] = None
    
    # Training hyperparameters
    num_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Gradient optimization
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Mixed precision (for RTX 3080)
    fp16: bool = True  # Enable for faster training on RTX 3080
    fp16_opt_level: str = "O1"
    
    # Evaluation and checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging
    logging_dir: str = "./logs"
    logging_steps: int = 100
    report_to: str = "tensorboard"
    
    # Output
    output_dir: str = "./codet5-docstring-model"
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class GPUConfig:
    """GPU configuration parameters"""
    
    # Multi-GPU settings
    use_multi_gpu: bool = True  # Auto-detect and use multiple GPUs
    device_map: str = "auto"  # "auto" for multi-GPU, None for single GPU
    
    # Memory optimization
    max_memory_per_gpu: Optional[str] = None  # e.g., "9GiB" for RTX 3080
    offload_folder: Optional[str] = None  # For model offloading if needed
    
    # DDP settings (for multi-GPU)
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    
    # Metrics
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_exact_match: bool = True
    
    # Evaluation settings
    num_eval_samples: int = 1000
    eval_batch_size: int = 16
    
    # Output
    save_predictions: bool = True
    predictions_file: str = "predictions.json"
    results_file: str = "evaluation_results.json"


@dataclass
class APIConfig:
    """API server configuration parameters"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Model settings
    model_path: str = "./codet5-docstring-model"
    
    # CORS settings
    enable_cors: bool = True
    cors_origins: str = "*"
    
    # Rate limiting (optional)
    enable_rate_limit: bool = False
    rate_limit_per_minute: int = 60


@dataclass
class Config:
    """Master configuration combining all sub-configs"""
    
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    gpu: GPUConfig = GPUConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: APIConfig = APIConfig()


# Default configuration instance
default_config = Config()


# Predefined configurations for different scenarios

# RTX 3080 Optimized (10GB VRAM)
rtx_3080_config = Config(
    model=ModelConfig(
        model_name="Salesforce/codet5-base"
    ),
    training=TrainingConfig(
        num_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=True
    ),
    gpu=GPUConfig(
        max_memory_per_gpu="9GiB"
    )
)

# Multi-GPU Configuration (2x RTX 3080)
multi_gpu_config = Config(
    model=ModelConfig(
        model_name="Salesforce/codet5-base"
    ),
    training=TrainingConfig(
        num_epochs=10,
        per_device_train_batch_size=12,  # Higher batch size with 2 GPUs
        gradient_accumulation_steps=1,
        fp16=True
    ),
    gpu=GPUConfig(
        use_multi_gpu=True,
        device_map="auto"
    )
)

# Quick Test Configuration (for debugging)
quick_test_config = Config(
    model=ModelConfig(
        model_name="Salesforce/codet5-small"
    ),
    training=TrainingConfig(
        num_epochs=2,
        per_device_train_batch_size=4,
        max_train_samples=1000,
        max_val_samples=100,
        eval_steps=50,
        save_steps=50
    )
)

# High Quality Configuration (for best results)
high_quality_config = Config(
    model=ModelConfig(
        model_name="Salesforce/codet5-large",
        num_beams=10
    ),
    training=TrainingConfig(
        num_epochs=15,
        per_device_train_batch_size=4,  # Smaller batch for large model
        learning_rate=3e-5,
        warmup_steps=1000
    )
)


def get_config(config_name: str = "default") -> Config:
    """
    Get configuration by name
    
    Args:
        config_name: Name of the configuration
            Options: "default", "rtx_3080", "multi_gpu", "quick_test", "high_quality"
    
    Returns:
        Config object
    """
    
    configs = {
        "default": default_config,
        "rtx_3080": rtx_3080_config,
        "multi_gpu": multi_gpu_config,
        "quick_test": quick_test_config,
        "high_quality": high_quality_config
    }
    
    if config_name not in configs:
        print(f"⚠️  Warning: Config '{config_name}' not found. Using default.")
        return default_config
    
    return configs[config_name]


def print_config(config: Config):
    """Print configuration in a readable format"""
    
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    
    print("\n📋 Model Configuration:")
    print(f"   ├─ Model: {config.model.model_name}")
    print(f"   ├─ Max Input Length: {config.model.max_input_length}")
    print(f"   ├─ Max Output Length: {config.model.max_output_length}")
    print(f"   ├─ Num Beams: {config.model.num_beams}")
    print(f"   └─ Temperature: {config.model.temperature}")
    
    print("\n🎯 Training Configuration:")
    print(f"   ├─ Epochs: {config.training.num_epochs}")
    print(f"   ├─ Batch Size (per device): {config.training.per_device_train_batch_size}")
    print(f"   ├─ Learning Rate: {config.training.learning_rate}")
    print(f"   ├─ FP16: {config.training.fp16}")
    print(f"   ├─ Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"   └─ Output Dir: {config.training.output_dir}")
    
    print("\n🖥️  GPU Configuration:")
    print(f"   ├─ Multi-GPU: {config.gpu.use_multi_gpu}")
    print(f"   ├─ Device Map: {config.gpu.device_map}")
    print(f"   └─ Max Memory: {config.gpu.max_memory_per_gpu}")
    
    print("\n📊 Evaluation Configuration:")
    print(f"   ├─ Compute BLEU: {config.evaluation.compute_bleu}")
    print(f"   ├─ Compute ROUGE: {config.evaluation.compute_rouge}")
    print(f"   └─ Num Samples: {config.evaluation.num_eval_samples}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Available configurations:")
    print("1. default")
    print("2. rtx_3080")
    print("3. multi_gpu")
    print("4. quick_test")
    print("5. high_quality")
    
    # Load and print RTX 3080 config
    config = get_config("rtx_3080")
    print_config(config)
