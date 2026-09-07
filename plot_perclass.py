"""
plot_per_class.py
-----------------
Reads results_tables/per_class_f1.csv (written by collect_results.py) and plots
each model's per-class F1:
  per_class_heatmap.png   -- classes x models, color = F1 (cleanest overview)
  per_class_bars.png      -- grouped bars, one group per class, one bar per model

Run:  python plot_per_class.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = os.path.join("results_tables", "per_class_f1.csv")
OUT = "results_tables"

df = pd.read_csv(CSV, index_col=0)
macro = df.loc["MACRO"] if "MACRO" in df.index else None
classes = df.drop(index="MACRO", errors="ignore")        # 15 classes x models
models = list(classes.columns)

# order models by their MACRO (best first) if we have it
if macro is not None:
    models = list(macro.sort_values(ascending=False).index)
    classes = classes[models]

# ---- 1) heatmap ---------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(1.3 * len(models) + 3, 0.5 * len(classes) + 2))
data = classes.values.astype(float)
im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes.index)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=8)
cbar = fig.colorbar(im, ax=ax, fraction=0.025); cbar.set_label("F1")
ax.set_title("Per-class F1 by model (non-hidden)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_class_heatmap.png"), dpi=130)
print("saved per_class_heatmap.png")

# ---- 2) grouped bars ----------------------------------------------------- #
fig, ax = plt.subplots(figsize=(16, 6))
n_cls, n_mod = len(classes), len(models)
x = np.arange(n_cls)
w = 0.8 / n_mod
cmap = plt.cm.viridis(np.linspace(0, 0.9, n_mod))
for m, (model, col) in enumerate(zip(models, cmap)):
    ax.bar(x + m * w - 0.4 + w / 2, classes[model].values, w, label=model, color=col)
ax.set_xticks(x); ax.set_xticklabels(classes.index, rotation=40, ha="right")
ax.set_ylabel("F1"); ax.set_ylim(0, 1)
ax.set_title("Per-class F1 by model (non-hidden)")
ax.legend(ncol=min(n_mod, 4), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_class_bars.png"), dpi=130)
print("saved per_class_bars.png")