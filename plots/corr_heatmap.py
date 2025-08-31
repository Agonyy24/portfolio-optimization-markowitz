import sys
import os

plots_dir = os.path.dirname(os.path.abspath(__file__)) #zapisz sobie notatke co to sys path os path po co to

project_root = os.path.dirname(os.path.dirname(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")

import matplotlib.pyplot as plt
import seaborn as sns
from src.stock_stat import stock_stat as stat 

returns_df, mean_returns, cov_matrix = stat()
corr_matrix = returns_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Return correlation matrix", fontsize=14)

os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=300)

plt.show()