# 04_compare_all_models.py
"""
Script tổng hợp để train tất cả các models và so sánh kết quả.
Có thể chạy từng model riêng hoặc chạy tất cả cùng lúc.
"""
import pandas as pd
import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import các module
sys.path.insert(0, str(Path(__file__).parent))

# Import với importlib vì tên file bắt đầu bằng số
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import module từ file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import các modules
base_path = Path(__file__).parent
train_nb_module = import_module_from_file("train_multinomial_nb", base_path / "04a_train_multinomial_nb.py")
train_lr_module = import_module_from_file("train_logistic_regression", base_path / "04b_train_logistic_regression.py")
train_svm_module = import_module_from_file("train_linear_svm", base_path / "04c_train_linear_svm.py")
train_rf_module = import_module_from_file("train_random_forest", base_path / "04d_train_random_forest.py")

train_nb = train_nb_module.main
train_lr = train_lr_module.main
train_svm = train_svm_module.main
train_rf = train_rf_module.main

# Set style cho seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def plot_model_comparison(res_df, output_dir="outputs/plots"):
    """Vẽ các biểu đồ so sánh models bằng seaborn."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu cho plotting
    df_plot = res_df.copy()
    df_plot = df_plot.sort_values(by="val_macro_f1", ascending=True)  # Sắp xếp để vẽ từ dưới lên
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart: Validation Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    colors = sns.color_palette("husl", len(df_plot))
    bars1 = ax1.barh(df_plot['model'], df_plot['val_acc'], color=colors)
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(axis='x', alpha=0.3)
    # Thêm giá trị trên mỗi bar
    for i, (bar, val) in enumerate(zip(bars1, df_plot['val_acc'])):
        ax1.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')
    
    # 2. Bar chart: Test Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(df_plot['model'], df_plot['test_acc'], color=colors)
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    # Thêm giá trị trên mỗi bar
    for i, (bar, val) in enumerate(zip(bars2, df_plot['test_acc'])):
        ax2.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')
    
    # 3. Bar chart: Validation Macro F1
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.barh(df_plot['model'], df_plot['val_macro_f1'], color=colors)
    ax3.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('Validation Macro F1 Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 1])
    ax3.grid(axis='x', alpha=0.3)
    # Thêm giá trị trên mỗi bar
    for i, (bar, val) in enumerate(zip(bars3, df_plot['val_macro_f1'])):
        ax3.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')
    
    # 4. Bar chart: Test Macro F1
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.barh(df_plot['model'], df_plot['test_macro_f1'], color=colors)
    ax4.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax4.set_title('Test Macro F1 Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.grid(axis='x', alpha=0.3)
    # Thêm giá trị trên mỗi bar
    for i, (bar, val) in enumerate(zip(bars4, df_plot['test_macro_f1'])):
        ax4.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')
    
    # 5. ROC-AUC comparison (nếu có)
    ax5 = fig.add_subplot(gs[2, 0])
    df_roc = df_plot[df_plot['val_roc_auc'].notna()].copy()
    if len(df_roc) > 0:
        bars5 = ax5.barh(df_roc['model'], df_roc['val_roc_auc'], 
                         color=[colors[list(df_plot['model']).index(m)] for m in df_roc['model']])
        ax5.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        ax5.set_title('Validation ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax5.set_xlim([0, 1])
        ax5.grid(axis='x', alpha=0.3)
        # Thêm giá trị trên mỗi bar
        for i, (bar, val) in enumerate(zip(bars5, df_roc['val_roc_auc'])):
            ax5.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'ROC-AUC data not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Validation ROC-AUC Comparison', fontsize=14, fontweight='bold')
    
    # 6. Heatmap: Tổng hợp tất cả metrics
    ax6 = fig.add_subplot(gs[2, 1])
    # Chuẩn bị dữ liệu cho heatmap
    heatmap_data = df_plot[['val_acc', 'test_acc', 'val_macro_f1', 'test_macro_f1']].T
    heatmap_data.columns = df_plot['model']
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, ax=ax6, vmin=0, vmax=1)
    ax6.set_title('Metrics Heatmap', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Models', fontsize=12, fontweight='bold')
    
    plt.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    # Lưu biểu đồ
    output_path = os.path.join(output_dir, "model_comparison_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved comparison dashboard -> {output_path}")
    plt.close()
    
    # Vẽ thêm biểu đồ so sánh Validation vs Test
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy: Validation vs Test
    x = range(len(df_plot))
    width = 0.35
    axes[0].bar([i - width/2 for i in x], df_plot['val_acc'], width, 
                label='Validation', color='steelblue', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], df_plot['test_acc'], width, 
                label='Test', color='coral', alpha=0.8)
    axes[0].set_xlabel('Models', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy: Validation vs Test', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_plot['model'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Macro F1: Validation vs Test
    axes[1].bar([i - width/2 for i in x], df_plot['val_macro_f1'], width, 
                label='Validation', color='steelblue', alpha=0.8)
    axes[1].bar([i + width/2 for i in x], df_plot['test_macro_f1'], width, 
                label='Test', color='coral', alpha=0.8)
    axes[1].set_xlabel('Models', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Macro F1: Validation vs Test', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_plot['model'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, "model_comparison_validation_vs_test.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"📊 Saved validation vs test comparison -> {output_path2}")
    plt.close()
    
    # 3. Radar/Spider chart để so sánh tổng thể
    try:
        from math import pi
        
        # Chuẩn bị dữ liệu cho radar chart
        metrics = ['val_acc', 'test_acc', 'val_macro_f1', 'test_macro_f1']
        if df_plot['val_roc_auc'].notna().any():
            metrics.append('val_roc_auc')
        
        # Normalize data (0-1 scale)
        df_radar = df_plot[['model'] + metrics].copy()
        for col in metrics:
            if col in df_radar.columns:
                df_radar[col] = df_radar[col].fillna(0)
        
        # Tạo radar chart
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Số lượng metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Đóng vòng tròn
        
        colors_radar = sns.color_palette("husl", len(df_radar))
        
        for idx, row in df_radar.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]  # Đóng vòng tròn
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors_radar[idx])
            ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        
        output_path3 = os.path.join(output_dir, "model_comparison_radar.png")
        plt.savefig(output_path3, dpi=300, bbox_inches='tight')
        print(f"📊 Saved radar chart -> {output_path3}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Không thể tạo radar chart: {e}")
    
    # 4. Correlation matrix giữa các metrics
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_metrics = ['val_acc', 'test_acc', 'val_macro_f1', 'test_macro_f1']
    if df_plot['val_roc_auc'].notna().any():
        corr_metrics.append('val_roc_auc')
    
    corr_data = df_plot[corr_metrics].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix Between Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    output_path4 = os.path.join(output_dir, "metrics_correlation.png")
    plt.savefig(output_path4, dpi=300, bbox_inches='tight')
    print(f"📊 Saved correlation matrix -> {output_path4}")
    plt.close()
    
    # 5. Ranking chart - Xếp hạng models theo từng metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    ranking_metrics = [
        ('val_acc', 'Validation Accuracy Ranking'),
        ('test_acc', 'Test Accuracy Ranking'),
        ('val_macro_f1', 'Validation Macro F1 Ranking'),
        ('test_macro_f1', 'Test Macro F1 Ranking')
    ]
    
    for idx, (metric, title) in enumerate(ranking_metrics):
        ax = axes[idx]
        df_rank = df_plot.sort_values(by=metric, ascending=True)
        df_rank['rank'] = range(1, len(df_rank) + 1)
        
        bars = ax.barh(df_rank['model'], df_rank[metric], 
                       color=[colors[list(df_plot['model']).index(m)] for m in df_rank['model']])
        ax.set_xlabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Thêm rank number
        for i, (bar, val, rank) in enumerate(zip(bars, df_rank[metric], df_rank['rank'])):
            ax.text(val + 0.01, i, f'#{rank} ({val:.4f})', va='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('Model Rankings by Different Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path5 = os.path.join(output_dir, "model_rankings.png")
    plt.savefig(output_path5, dpi=300, bbox_inches='tight')
    print(f"📊 Saved ranking chart -> {output_path5}")
    plt.close()
    
    # 6. Performance gap chart (Validation - Test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_gap = df_plot.copy()
    df_gap['acc_gap'] = df_gap['val_acc'] - df_gap['test_acc']
    df_gap['f1_gap'] = df_gap['val_macro_f1'] - df_gap['test_macro_f1']
    df_gap = df_gap.sort_values(by='acc_gap', ascending=True)
    
    # Accuracy gap
    bars1 = axes[0].barh(df_gap['model'], df_gap['acc_gap'], 
                         color=['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' 
                                for x in df_gap['acc_gap']], alpha=0.7)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Gap (Validation - Test)', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Gap: Validation vs Test', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, df_gap['acc_gap'])):
        axes[0].text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.4f}', 
                    va='center', ha='left' if val >= 0 else 'right', fontweight='bold')
    
    # F1 gap
    bars2 = axes[1].barh(df_gap['model'], df_gap['f1_gap'],
                         color=['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' 
                                for x in df_gap['f1_gap']], alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Gap (Validation - Test)', fontsize=12, fontweight='bold')
    axes[1].set_title('Macro F1 Gap: Validation vs Test', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, df_gap['f1_gap'])):
        axes[1].text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.4f}', 
                    va='center', ha='left' if val >= 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    output_path6 = os.path.join(output_dir, "performance_gap.png")
    plt.savefig(output_path6, dpi=300, bbox_inches='tight')
    print(f"📊 Saved performance gap chart -> {output_path6}")
    plt.close()
    
    # 7. Scatter plot: Validation vs Test
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy scatter
    for idx, row in df_plot.iterrows():
        axes[0].scatter(row['val_acc'], row['test_acc'], 
                       s=200, alpha=0.7, label=row['model'], color=colors[idx])
        axes[0].text(row['val_acc'] + 0.005, row['test_acc'] + 0.005, 
                    row['model'], fontsize=9, fontweight='bold')
    
    # Đường y=x (perfect correlation)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect correlation')
    axes[0].set_xlabel('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Validation vs Test Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlim([0.7, 1.0])
    axes[0].set_ylim([0.7, 1.0])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right', fontsize=8)
    
    # Macro F1 scatter
    for idx, row in df_plot.iterrows():
        axes[1].scatter(row['val_macro_f1'], row['test_macro_f1'], 
                       s=200, alpha=0.7, label=row['model'], color=colors[idx])
        axes[1].text(row['val_macro_f1'] + 0.005, row['test_macro_f1'] + 0.005, 
                    row['model'], fontsize=9, fontweight='bold')
    
    # Đường y=x
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect correlation')
    axes[1].set_xlabel('Validation Macro F1', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Test Macro F1', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation vs Test Macro F1', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0.6, 1.0])
    axes[1].set_ylim([0.6, 1.0])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    output_path7 = os.path.join(output_dir, "validation_vs_test_scatter.png")
    plt.savefig(output_path7, dpi=300, bbox_inches='tight')
    print(f"📊 Saved scatter plot -> {output_path7}")
    plt.close()
    
    # 8. Grouped bar chart - Tất cả metrics cùng một biểu đồ
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df_plot))
    width = 0.2
    
    # Vẽ các bars
    bars1 = ax.bar(x - 1.5*width, df_plot['val_acc'], width, label='Val Accuracy', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, df_plot['test_acc'], width, label='Test Accuracy', 
                   color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, df_plot['val_macro_f1'], width, label='Val Macro F1', 
                   color='#2ecc71', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, df_plot['test_macro_f1'], width, label='Test Macro F1', 
                   color='#f39c12', alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Metrics Comparison (Grouped)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['model'], rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path8 = os.path.join(output_dir, "all_metrics_grouped.png")
    plt.savefig(output_path8, dpi=300, bbox_inches='tight')
    print(f"📊 Saved grouped metrics chart -> {output_path8}")
    plt.close()
    
    # 9. Summary table visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Chuẩn bị dữ liệu cho table
    table_data = []
    for _, row in df_plot.iterrows():
        table_data.append([
            row['model'],
            f"{row['val_acc']:.4f}",
            f"{row['test_acc']:.4f}",
            f"{row['val_macro_f1']:.4f}",
            f"{row['test_macro_f1']:.4f}",
            f"{row['val_roc_auc']:.4f}" if pd.notna(row['val_roc_auc']) else "N/A"
        ])
    
    columns = ['Model', 'Val Acc', 'Test Acc', 'Val Macro F1', 'Test Macro F1', 'Val ROC-AUC']
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best scores
    best_val_acc_model = df_plot.loc[df_plot['val_acc'].idxmax(), 'model']
    best_test_acc_model = df_plot.loc[df_plot['test_acc'].idxmax(), 'model']
    best_val_f1_model = df_plot.loc[df_plot['val_macro_f1'].idxmax(), 'model']
    best_test_f1_model = df_plot.loc[df_plot['test_macro_f1'].idxmax(), 'model']
    
    for idx, (_, row) in enumerate(df_plot.iterrows(), 1):
        model_name = row['model']
        if model_name == best_val_acc_model:
            table[(idx, 1)].set_facecolor('#2ecc71')
        if model_name == best_test_acc_model:
            table[(idx, 2)].set_facecolor('#2ecc71')
        if model_name == best_val_f1_model:
            table[(idx, 3)].set_facecolor('#2ecc71')
        if model_name == best_test_f1_model:
            table[(idx, 4)].set_facecolor('#2ecc71')
    
    plt.title('Model Comparison Summary Table', fontsize=16, fontweight='bold', pad=20)
    output_path9 = os.path.join(output_dir, "model_comparison_table.png")
    plt.savefig(output_path9, dpi=300, bbox_inches='tight')
    print(f"📊 Saved summary table -> {output_path9}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train và so sánh tất cả các models")
    parser.add_argument("--models", nargs="+", 
                       choices=["nb", "lr", "svm", "rf", "all"],
                       default=["all"],
                       help="Models để train: nb (Naive Bayes), lr (Logistic Regression), svm (SVM), rf (Random Forest), all (tất cả)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("COMPARING ALL MODELS")
    print("="*60)
    
    results = []
    
    # Map model names
    model_map = {
        "nb": ("MultinomialNB", train_nb),
        "lr": ("LogisticRegression", train_lr),
        "svm": ("LinearSVM", train_svm),
        "rf": ("RandomForest", train_rf),
    }
    
    # Determine which models to train
    if "all" in args.models:
        models_to_train = ["nb", "lr", "svm", "rf"]
    else:
        models_to_train = args.models
    
    # Train và thu thập kết quả từ mỗi model
    for model_key in models_to_train:
        model_name, train_func = model_map[model_key]
        try:
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}")
            result = train_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Lỗi khi train {model_name}: {e}")
            continue
    
    if not results:
        print("\n❌ Không có kết quả nào để so sánh!")
        return
    
    # Tạo DataFrame và sắp xếp
    res_df = pd.DataFrame(results).sort_values(by="val_macro_f1", ascending=False)
    out_path = "outputs/model_comparison.csv"
    res_df.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY (sorted by val_macro_f1)")
    print("="*60)
    print(res_df.to_string(index=False))
    print(f"\n💾 Saved comparison table -> {out_path}")
    
    # In thêm thống kê
    print("\n" + "="*60)
    print("THỐNG KÊ")
    print("="*60)
    print(f"Model tốt nhất (val_macro_f1): {res_df.iloc[0]['model']} ({res_df.iloc[0]['val_macro_f1']:.4f})")
    print(f"Model tốt nhất (test_macro_f1): {res_df.loc[res_df['test_macro_f1'].idxmax(), 'model']} ({res_df['test_macro_f1'].max():.4f})")
    print(f"Model tốt nhất (val_acc): {res_df.loc[res_df['val_acc'].idxmax(), 'model']} ({res_df['val_acc'].max():.4f})")
    print(f"Model tốt nhất (test_acc): {res_df.loc[res_df['test_acc'].idxmax(), 'model']} ({res_df['test_acc'].max():.4f})")
    
    # Vẽ biểu đồ so sánh
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ SO SÁNH")
    print("="*60)
    try:
        plot_model_comparison(res_df)
        print("✅ Đã tạo biểu đồ so sánh thành công!")
    except Exception as e:
        print(f"⚠️  Lỗi khi tạo biểu đồ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

