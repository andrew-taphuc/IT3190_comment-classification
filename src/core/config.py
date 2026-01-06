"""
File cấu hình cho project.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Cấu hình cho model training."""
    # TF-IDF parameters
    word_ngram_range: Tuple[int, int] = (1, 2)
    char_ngram_range: Tuple[int, int] = (3, 5)
    min_df: int = 2
    max_df: float = 0.95
    max_features: int = 50000
    sublinear_tf: bool = True
    
    # Model parameters
    svm_C: float = 2.0
    svm_class_weight: str = "balanced"
    calibration_cv: int = 3
    calibration_method: str = "sigmoid"
    
    # Threshold
    default_threshold: float = 0.70
    
    # Random state
    random_state: int = 42


@dataclass
class DataConfig:
    """Cấu hình cho data paths."""
    data_dir: str = "data/processed"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    text_col: str = "text"
    label_col: str = "label"


@dataclass
class OutputConfig:
    """Cấu hình cho output paths."""
    model_out: str = "outputs/toxicity_pipeline.joblib"
    meta_out: str = "outputs/toxicity_meta.json"
    comparison_out: str = "outputs/model_comparison.csv"


# Default configs
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()

