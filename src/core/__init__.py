"""
Core modules cho comment classification project.
"""
from .text_cleaner import clean_text, normalize_unicode, normalize_repeated_chars, normalize_teencode
from .config import ModelConfig, DataConfig, OutputConfig, DEFAULT_MODEL_CONFIG, DEFAULT_DATA_CONFIG, DEFAULT_OUTPUT_CONFIG
from .evaluation import evaluate_model, print_evaluation_report
from .feature_extractor import extract_text_features, combine_features
from .utils import load_data_split, get_label_distribution, print_label_distribution, ensure_dir

__all__ = [
    'clean_text',
    'normalize_unicode',
    'normalize_repeated_chars',
    'normalize_teencode',
    'ModelConfig',
    'DataConfig',
    'OutputConfig',
    'DEFAULT_MODEL_CONFIG',
    'DEFAULT_DATA_CONFIG',
    'DEFAULT_OUTPUT_CONFIG',
    'evaluate_model',
    'print_evaluation_report',
    'extract_text_features',
    'combine_features',
    'load_data_split',
    'get_label_distribution',
    'print_label_distribution',
    'ensure_dir',
]

