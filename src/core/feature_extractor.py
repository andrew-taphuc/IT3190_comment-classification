"""
Module chứa các hàm trích xuất features từ text.
"""
import re
import numpy as np
from typing import List, Dict


def extract_text_features(texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Trích xuất các features từ danh sách text.
    
    Returns:
        Dict với keys: 'emoji_count', 'exclamation_count', 'question_count',
        'uppercase_ratio', 'punctuation_count', 'word_count', 'char_count'
    """
    features = {
        'emoji_count': [],
        'exclamation_count': [],
        'question_count': [],
        'uppercase_ratio': [],
        'punctuation_count': [],
        'word_count': [],
        'char_count': [],
    }
    
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Emoji count
        emoji_matches = emoji_pattern.findall(text)
        features['emoji_count'].append(len(emoji_matches))
        
        # Punctuation counts
        features['exclamation_count'].append(text.count('!'))
        features['question_count'].append(text.count('?'))
        features['punctuation_count'].append(
            len(re.findall(r'[!?.]+', text))
        )
        
        # Uppercase ratio
        if len(text) > 0:
            uppercase_count = sum(1 for c in text if c.isupper())
            features['uppercase_ratio'].append(uppercase_count / len(text))
        else:
            features['uppercase_ratio'].append(0.0)
        
        # Word and char counts
        words = text.split()
        features['word_count'].append(len(words))
        features['char_count'].append(len(text))
    
    # Convert to numpy arrays
    return {k: np.array(v).reshape(-1, 1) for k, v in features.items()}


def combine_features(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Kết hợp tất cả features thành một matrix.
    """
    return np.hstack([v for v in feature_dict.values()])

