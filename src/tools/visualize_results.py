"""
Script để visualize kết quả model: ROC curve, PR curve, confusion matrix.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_data_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_roc_curve(y_true, y_proba, labels, save_path=None):
    """Vẽ ROC curve."""
    toxic_idx = labels.index('toxic') if 'toxic' in labels else 1
    y_binary = (np.array(y_true) == 'toxic').astype(int)
    y_proba_toxic = y_proba[:, toxic_idx] if y_proba.ndim > 1 else y_proba
    
    fpr, tpr, thresholds = roc_curve(y_binary, y_proba_toxic)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curve to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pr_curve(y_true, y_proba, labels, save_path=None):
    """Vẽ Precision-Recall curve."""
    toxic_idx = labels.index('toxic') if 'toxic' in labels else 1
    y_binary = (np.array(y_true) == 'toxic').astype(int)
    y_proba_toxic = y_proba[:, toxic_idx] if y_proba.ndim > 1 else y_proba
    
    precision, recall, thresholds = precision_recall_curve(y_binary, y_proba_toxic)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='PR curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PR curve to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Vẽ confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_model_comparison(csv_path, save_path=None):
    """Vẽ so sánh các models."""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot macro F1
    axes[0].barh(df['model'], df['test_macro_f1'], color='steelblue')
    axes[0].set_xlabel('Macro F1 Score', fontsize=12)
    axes[0].set_title('Model Comparison - Macro F1', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot accuracy
    axes[1].barh(df['model'], df['test_acc'], color='coral')
    axes[1].set_xlabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved model comparison to: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize model results")
    ap.add_argument("--model", default="outputs/toxicity_pipeline.joblib")
    ap.add_argument("--val_csv", default="data/processed/val.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--output_dir", default="outputs/plots")
    ap.add_argument("--comparison_csv", default="outputs/model_comparison.csv")
    
    args = ap.parse_args()
    
    import os
    import joblib
    from core import clean_text  # Import để đảm bảo module có sẵn khi load model
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    print(f"Loading model: {args.model}")
    pipe = joblib.load(args.model)
    
    print(f"Loading validation data: {args.val_csv}")
    X_val, y_val = load_data_split(args.val_csv, args.text_col, args.label_col)
    
    # Predict
    print("Predicting...")
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)
    labels = sorted(list(set(y_val)))
    
    # Create plots
    print("\n📊 Creating visualizations...")
    
    plot_roc_curve(y_val, y_proba, labels, 
                   save_path=f"{args.output_dir}/roc_curve.png")
    
    plot_pr_curve(y_val, y_proba, labels,
                  save_path=f"{args.output_dir}/pr_curve.png")
    
    plot_confusion_matrix(y_val, y_pred, labels,
                         save_path=f"{args.output_dir}/confusion_matrix.png")
    
    # Model comparison
    if os.path.exists(args.comparison_csv):
        plot_model_comparison(args.comparison_csv,
                             save_path=f"{args.output_dir}/model_comparison.png")
    
    print(f"\n✅ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

