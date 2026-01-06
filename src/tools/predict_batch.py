"""
Script để predict nhiều text cùng lúc (batch prediction).
"""
import json
import argparse
import sys
import pandas as pd
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import clean_text  # Import để đảm bảo module có sẵn khi load model


def predict_batch(pipe, texts, threshold=0.70):
    """
    Predict nhiều text cùng lúc.
    
    Args:
        pipe: Trained pipeline
        texts: List of texts hoặc pandas Series
        threshold: Threshold để phân loại toxic
    
    Returns:
        List of dicts với predictions
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Predict probabilities
    proba = pipe.predict_proba(texts)
    classes = list(pipe.classes_)
    
    # Get toxic index
    if 'toxic' in classes:
        toxic_idx = classes.index('toxic')
    else:
        toxic_idx = -1
    
    results = []
    for i, text in enumerate(texts):
        toxic_score = float(proba[i, toxic_idx])
        label = "toxic" if toxic_score >= threshold else "non_toxic"
        
        results.append({
            "text": text,
            "label": label,
            "toxic_score": toxic_score,
            "threshold": threshold,
        })
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Batch prediction")
    ap.add_argument("--model", default="outputs/toxicity_pipeline.joblib")
    ap.add_argument("--input", default=None, help="File CSV hoặc text file (một text mỗi dòng)")
    ap.add_argument("--text_col", default="text", help="Tên cột text nếu input là CSV")
    ap.add_argument("--output", default=None, help="File output (JSON hoặc CSV)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--format", choices=["json", "csv"], default="json")
    
    args = ap.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    pipe = joblib.load(args.model)
    
    # Load threshold từ meta nếu có
    threshold = args.threshold
    if threshold is None:
        try:
            # Try to find meta file in same directory as model
            import os
            model_dir = os.path.dirname(args.model) if os.path.dirname(args.model) else "outputs"
            model_name = os.path.basename(args.model).replace('.joblib', '_meta.json')
            meta_path = os.path.join(model_dir, model_name)
            # Also try without _meta suffix (for backward compatibility)
            if not os.path.exists(meta_path):
                meta_path = args.model.replace('.joblib', '.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                threshold = meta.get('threshold_toxic', 0.70)
        except:
            threshold = 0.70
    
    # Load texts
    if args.input:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            if args.text_col not in df.columns:
                raise ValueError(f"CSV phải có cột '{args.text_col}'")
            texts = df[args.text_col].astype(str).tolist()
        else:
            # Text file, một dòng một text
            with open(args.input, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        # Đọc từ stdin
        texts = [line.strip() for line in sys.stdin if line.strip()]
    
    if not texts:
        print("❌ Không có text nào để predict!", file=sys.stderr)
        return
    
    print(f"Predicting {len(texts)} texts...")
    results = predict_batch(pipe, texts, threshold)
    
    # Output
    if args.output:
        if args.format == "json":
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            # CSV
            df_out = pd.DataFrame(results)
            df_out.to_csv(args.output, index=False)
        print(f"✅ Saved results to: {args.output}")
    else:
        # Print to stdout
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

