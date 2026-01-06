"""
Module chứa các hàm đánh giá model.
"""
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)


def evaluate_model(y_true, y_pred, y_proba=None, labels=None):
    """
    Đánh giá model với nhiều metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        labels: List of class labels
    
    Returns:
        Dict chứa các metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    if labels is not None:
        for label, f1 in zip(labels, f1_per_class):
            metrics[f'f1_{label}'] = float(f1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC-AUC và PR-AUC (nếu có probabilities)
    if y_proba is not None and labels is not None:
        # Tìm index của class "toxic"
        if 'toxic' in labels:
            toxic_idx = list(labels).index('toxic')
            y_proba_toxic = y_proba[:, toxic_idx] if y_proba.ndim > 1 else y_proba
            y_binary = (np.array(y_true) == 'toxic').astype(int)
            
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_binary, y_proba_toxic))
            except ValueError:
                metrics['roc_auc'] = None
            
            try:
                metrics['pr_auc'] = float(average_precision_score(y_binary, y_proba_toxic))
            except ValueError:
                metrics['pr_auc'] = None
    
    return metrics


def print_evaluation_report(y_true, y_pred, y_proba=None, labels=None, split_name=""):
    """
    In báo cáo đánh giá chi tiết.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {split_name}")
    print(f"{'='*60}")
    
    metrics = evaluate_model(y_true, y_pred, y_proba, labels)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['weighted_f1']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics and metrics['pr_auc'] is not None:
        print(f"  PR-AUC        : {metrics['pr_auc']:.4f}")
    
    print(f"\n📈 Per-class F1:")
    for key, value in metrics.items():
        if key.startswith('f1_'):
            print(f"  {key:15s}: {value:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))
    
    print(f"\n🔢 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    if labels is not None:
        print(f"      {'Predicted':>20}")
        print(f"      {labels[0]:>10} {labels[1]:>10}")
        print(f"True {labels[0]:>5} {cm[0][0]:>10} {cm[0][1]:>10}")
        print(f"     {labels[1]:>5} {cm[1][0]:>10} {cm[1][1]:>10}")
    else:
        print(np.array(cm))
    
    return metrics

