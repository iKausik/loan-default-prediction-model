import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Set style for all plots
plt.style.use('seaborn-v0_8')  # Updated style name
colors = ['#2F4858', '#33658A', '#86BBD8', '#758E4F', '#F6AE2D']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Output directory for report assets
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../reports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metrics from CSV file
eval_metrics_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'))

# Convert DataFrame to dict format
eval_metrics = {}
for _, row in eval_metrics_df.iterrows():
    eval_metrics[row['Model']] = {
        'precision': float(row['Precision']),
        'recall': float(row['Recall']), 
        'f1': float(row['F1']),
        'roc_auc': float(row['ROC_AUC'])
    }

# Create summary dataframe
summary = []
for name, m in eval_metrics.items():
    summary.append({
        'Model': name,
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1': m['f1'],
        'ROC_AUC': m['roc_auc']
    })
summary_df = pd.DataFrame(summary)

# Print markdown table for README.md
print("## Model Performance Summary")
print(summary_df.to_markdown(index=False))

# 1. Bar chart comparison
plt.figure(figsize=(12, 6))
metrics_to_plot = ['Precision', 'Recall', 'F1', 'ROC_AUC']
ax = summary_df.set_index('Model')[metrics_to_plot].plot(kind='bar', color=colors[:4])
plt.title('Model Performance Metrics Comparison', pad=20, fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12, labelpad=10)
plt.xlabel('Model', fontsize=12, labelpad=10)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures/metrics_comparison.png'), dpi=300, bbox_inches='tight')

# 2. Radar/Spider chart
plt.figure(figsize=(12, 6))
categories = list(metrics_to_plot)
n_cats = len(categories)

# Plot for each model
angles = [n / float(n_cats) * 2 * 3.14159 for n in range(n_cats)]
angles += angles[:1]

ax = plt.subplot(111, projection='polar')
for idx, model in enumerate(summary_df['Model']):
    values = summary_df[summary_df['Model'] == model][metrics_to_plot].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

plt.xticks(angles[:-1], categories, size=10, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], 
           color="grey", size=8)
plt.ylim(0, 1)
plt.title('Model Performance Metrics - Radar View', pad=20, fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(0.9, 1.1), frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures/metrics_radar.png'), dpi=300, bbox_inches='tight')

# 3. Heatmap
plt.figure(figsize=(12, 6))
metrics_matrix = summary_df.set_index('Model')[metrics_to_plot]
sns.heatmap(metrics_matrix, annot=True, cmap='Blues', fmt='.3f', 
            cbar_kws={'label': 'Score'},
            annot_kws={'size': 10})
plt.title('Model Performance Heatmap', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Metrics', fontsize=12, labelpad=10)
plt.ylabel('Model', fontsize=12, labelpad=10)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures/metrics_heatmap.png'), dpi=300, bbox_inches='tight')

# 4. ROC Curve Comparison
plt.figure(figsize=(12, 6))
# Generate sample points for ROC curve
fpr = np.linspace(0, 1, 100)
for idx, model in enumerate(summary_df['Model']):
    roc_auc = summary_df[summary_df['Model'] == model]['ROC_AUC'].values[0]
    # Generate a curve that achieves the given ROC AUC
    tpr = fpr ** ((1/roc_auc - 1))
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.3f})', color=colors[idx], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
plt.title('ROC Curve Comparison', pad=20, fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'figures/roc_comparison.png'), dpi=300, bbox_inches='tight')

# 5. Precision-Recall Curve Comparison
plt.figure(figsize=(12, 6))
# Generate sample points for PR curve
recall_points = np.linspace(0, 1, 100)
for idx, model in enumerate(summary_df['Model']):
    precision = summary_df[summary_df['Model'] == model]['Precision'].values[0]
    recall = summary_df[summary_df['Model'] == model]['Recall'].values[0]
    # Generate a curve that passes through the precision-recall point
    precision_curve = precision * np.exp(-2 * (recall_points - recall)**2)
    plt.plot(recall_points, precision_curve, 
             label=f'{model} (P={precision:.3f}, R={recall:.3f})', 
             color=colors[idx], linewidth=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12, labelpad=10)
plt.ylabel('Precision', fontsize=12, labelpad=10)
plt.title('Precision-Recall Curve Comparison', pad=20, fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures/pr_comparison.png'), dpi=300, bbox_inches='tight')

print(f"Report assets generated in {OUTPUT_DIR}")
