"""
Script để tìm threshold tối ưu cho classification.
"""
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_data_split, evaluate_model


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Tìm threshold tối ưu dựa trên metric.
    
    Args:
        y_true: True labels (binary: 0 hoặc 1)
        y_proba: Predicted probabilities cho class 1
        metric: 'f1', 'precision', 'recall', hoặc 'balanced' (F1 với recall >= 0.7)
    
    Returns:
        best_threshold, best_score
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true phải là binary labels")
    
    # Convert labels to binary
    y_binary = (np.array(y_true) == 'toxic').astype(int) if isinstance(y_true[0], str) else y_true
    
    # Get probabilities for toxic class
    if y_proba.ndim > 1:
        toxic_proba = y_proba[:, 1]  # Assume toxic is class 1
    else:
        toxic_proba = y_proba
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.95, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (toxic_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_binary, y_pred, zero_division=0)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(y_binary, y_pred, zero_division=0)
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            score = recall_score(y_binary, y_pred, zero_division=0)
        elif metric == 'balanced':
            from sklearn.metrics import precision_score, recall_score
            prec = precision_score(y_binary, y_pred, zero_division=0)
            rec = recall_score(y_binary, y_pred, zero_division=0)
            if rec >= 0.7:
                score = f1_score(y_binary, y_pred, zero_division=0)
            else:
                score = 0  # Penalize low recall
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return float(best_threshold), float(best_score)


def main():
    ap = argparse.ArgumentParser(description="Tìm threshold tối ưu")
    ap.add_argument("--val_csv", default="data/processed/val.csv")
    ap.add_argument("--model", default="outputs/toxicity_pipeline.joblib")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--metric", choices=['f1', 'precision', 'recall', 'balanced'],
                   default='f1', help="Metric để optimize")
    ap.add_argument("--output", default="outputs/optimal_threshold.json")
    
    args = ap.parse_args()
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "outputs", exist_ok=True)
    
    import joblib
    from core import clean_text  # Import để đảm bảo module có sẵn khi load model
    
    print(f"Loading model: {args.model}")
    pipe = joblib.load(args.model)
    
    print(f"Loading validation data: {args.val_csv}")
    X_val, y_val = load_data_split(args.val_csv, args.text_col, args.label_col)
    
    print(f"Predicting probabilities...")
    y_proba = pipe.predict_proba(X_val)
    labels = sorted(list(set(y_val)))
    
    # Find optimal threshold
    print(f"\n🔍 Finding optimal threshold (metric={args.metric})...")
    best_threshold, best_score = find_optimal_threshold(y_val, y_proba, args.metric)
    
    print(f"\n✅ Optimal threshold: {best_threshold:.4f}")
    print(f"✅ Best {args.metric} score: {best_score:.4f}")
    
    # Evaluate với threshold mới
    toxic_idx = labels.index('toxic') if 'toxic' in labels else 1
    y_pred_new = (y_proba[:, toxic_idx] >= best_threshold).astype(int)
    y_pred_new_labels = [labels[toxic_idx] if p == 1 else labels[1-toxic_idx] for p in y_pred_new]
    
    metrics = evaluate_model(y_val, y_pred_new_labels, y_proba, labels)
    
    print(f"\n📊 Metrics với threshold {best_threshold:.4f}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    if 'roc_auc' in metrics and metrics['roc_auc']:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save results
    results = {
        "optimal_threshold": best_threshold,
        "best_score": best_score,
        "metric": args.metric,
        "metrics_with_threshold": {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved results to: {args.output}")


if __name__ == "__main__":
    main()

