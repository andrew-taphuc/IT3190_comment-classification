"""
Script để visualize kết quả Linear SVM model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Arial'

# Import core modules
from core import clean_text, load_data_split

def plot_all_visualizations():
    """Tạo tất cả các biểu đồ cho SVM model."""
    
    # Create output directory
    output_dir = "outputs/plots_svm"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("📊 Loading Linear SVM model...")
    model_path = "outputs/toxicity_pipeline.joblib"
    pipe = joblib.load(model_path)
    
    # Load validation data
    print("📊 Loading validation data...")
    X_val, y_val = load_data_split("data/processed/val.csv", "text", "label")
    
    # Load test data
    print("📊 Loading test data...")
    X_test, y_test = load_data_split("data/processed/test.csv", "text", "label")
    
    # Predictions
    print("📊 Making predictions...")
    y_val_pred = pipe.predict(X_val)
    y_val_proba = pipe.predict_proba(X_val)
    
    y_test_pred = pipe.predict(X_test)
    y_test_proba = pipe.predict_proba(X_test)
    
    labels = ['non_toxic', 'toxic']
    
    # 1. ROC Curve
    print("\n📈 Creating ROC Curve...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Val ROC
    toxic_idx = 1
    y_val_binary = (np.array(y_val) == 'toxic').astype(int)
    fpr_val, tpr_val, _ = roc_curve(y_val_binary, y_val_proba[:, toxic_idx])
    roc_auc_val = roc_auc_score(y_val_binary, y_val_proba[:, toxic_idx])
    
    axes[0].plot(fpr_val, tpr_val, linewidth=2.5, color='#2196F3', label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC Curve - Validation Set', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Test ROC
    y_test_binary = (np.array(y_test) == 'toxic').astype(int)
    fpr_test, tpr_test, _ = roc_curve(y_test_binary, y_test_proba[:, toxic_idx])
    roc_auc_test = roc_auc_score(y_test_binary, y_test_proba[:, toxic_idx])
    
    axes[1].plot(fpr_test, tpr_test, linewidth=2.5, color='#4CAF50', label=f'ROC curve (AUC = {roc_auc_test:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve - Test Set', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    roc_path = f"{output_dir}/roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {roc_path}")
    plt.close()
    
    # 2. Precision-Recall Curve
    print("📈 Creating Precision-Recall Curve...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Val PR
    precision_val, recall_val, _ = precision_recall_curve(y_val_binary, y_val_proba[:, toxic_idx])
    pr_auc_val = average_precision_score(y_val_binary, y_val_proba[:, toxic_idx])
    
    axes[0].plot(recall_val, precision_val, linewidth=2.5, color='#FF9800', label=f'PR curve (AUC = {pr_auc_val:.4f})')
    axes[0].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[0].set_title('Precision-Recall Curve - Validation Set', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower left', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Test PR
    precision_test, recall_test, _ = precision_recall_curve(y_test_binary, y_test_proba[:, toxic_idx])
    pr_auc_test = average_precision_score(y_test_binary, y_test_proba[:, toxic_idx])
    
    axes[1].plot(recall_test, precision_test, linewidth=2.5, color='#9C27B0', label=f'PR curve (AUC = {pr_auc_test:.4f})')
    axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision-Recall Curve - Test Set', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    pr_path = f"{output_dir}/pr_curve.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {pr_path}")
    plt.close()
    
    # 3. Confusion Matrix
    print("📈 Creating Confusion Matrix...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Val confusion matrix
    cm_val = confusion_matrix(y_val, y_val_pred, labels=labels)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=axes[0],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix - Validation Set', fontsize=13, fontweight='bold')
    
    # Test confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred, labels=labels)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=axes[1],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix - Test Set', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {cm_path}")
    plt.close()
    
    # 4. Model Comparison
    print("📈 Creating Model Comparison...")
    comparison_csv = "outputs/model_comparison.csv"
    if os.path.exists(comparison_csv):
        df = pd.read_csv(comparison_csv)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort by macro F1
        df_sorted = df.sort_values(by='test_macro_f1', ascending=True)
        
        # Plot macro F1
        colors = ['#FF5252' if model == 'LinearSVM' else '#78909C' for model in df_sorted['model']]
        axes[0].barh(df_sorted['model'], df_sorted['test_macro_f1'], color=colors, edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Comparison - Test Macro F1', fontsize=13, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            axes[0].text(row['test_macro_f1'] + 0.01, i, f"{row['test_macro_f1']:.4f}", 
                        va='center', fontweight='bold', fontsize=10)
        
        # Plot accuracy
        colors = ['#4CAF50' if model == 'LinearSVM' else '#78909C' for model in df_sorted['model']]
        axes[1].barh(df_sorted['model'], df_sorted['test_acc'], color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Comparison - Test Accuracy', fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            axes[1].text(row['test_acc'] + 0.005, i, f"{row['test_acc']:.4f}", 
                        va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        comp_path = f"{output_dir}/model_comparison.png"
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {comp_path}")
        plt.close()
    
    # 5. Performance Metrics Bar Chart
    print("📈 Creating Performance Metrics Chart...")
    metrics_data = {
        'Validation': {
            'Accuracy': 0.8774,
            'Macro F1': 0.8045,
            'F1 (non_toxic)': 0.9239,
            'F1 (toxic)': 0.6850
        },
        'Test': {
            'Accuracy': 0.8748,
            'Macro F1': 0.7900,
            'F1 (non_toxic)': 0.9235,
            'F1 (toxic)': 0.6565
        }
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = list(metrics_data['Validation'].keys())
    val_scores = list(metrics_data['Validation'].values())
    test_scores = list(metrics_data['Test'].values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_scores, width, label='Validation', color='#2196F3', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test', color='#4CAF50', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Linear SVM Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    perf_path = f"{output_dir}/performance_metrics.png"
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {perf_path}")
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print(f"   - roc_curve.png")
    print(f"   - pr_curve.png")
    print(f"   - confusion_matrix.png")
    print(f"   - model_comparison.png")
    print(f"   - performance_metrics.png")

if __name__ == "__main__":
    plot_all_visualizations()
