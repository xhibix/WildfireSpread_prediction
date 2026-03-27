import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_excel('d:/Files/Imperial_MSc/sem2/Climate/final_project/results.xlsx')

# Parse the data from the messy Excel structure
data_rows = df.iloc[2:8].copy()
models = data_rows['Unnamed: 2'].tolist()
ap_mean = data_rows['Unnamed: 3'].astype(float).tolist()
ap_std = data_rows['Unnamed: 4'].astype(float).tolist()
fold_0 = data_rows['Unnamed: 5'].astype(float).tolist()
fold_1 = data_rows['Unnamed: 6'].astype(float).tolist()
fold_2 = data_rows['Unnamed: 7'].astype(float).tolist()

# Shorten model names for readability
short_names = [m.strip() for m in models]

colors = plt.cm.tab10.colors[:len(models)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance (Average Precision)', fontsize=14, fontweight='bold')

# --- Left: Bar chart of mean AP with std error bars ---
x = np.arange(len(models))
bars = ax1.bar(x, ap_mean, yerr=ap_std, color=colors, capsize=5, width=0.6,
               error_kw=dict(elinewidth=1.5, ecolor='black'))
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('AP')
ax1.set_title('Mean AP per Model (±std)')
ax1.set_ylim(0, max(ap_mean) * 1.4)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- Right: Line plot of AP per fold for each model ---
folds = [0, 1, 2]
for i, model in enumerate(short_names):
    ap_per_fold = [fold_0[i], fold_1[i], fold_2[i]]
    ax2.plot(folds, ap_per_fold, marker='o', color=colors[i], label=model, linewidth=2)

ax2.set_xticks(folds)
ax2.set_xticklabels(['Fold 0', 'Fold 1', 'Fold 2'])
ax2.set_ylabel('AP')
ax2.set_title('AP per Fold per Model')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(linestyle='--', alpha=0.5)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

plt.tight_layout()
plt.savefig('d:/Files/Imperial_MSc/sem2/Climate/final_project/results_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to results_plot.png")
