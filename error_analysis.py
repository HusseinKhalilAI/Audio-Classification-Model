"""
error_analysis.py  --  Phase 2(d): Qualitative Error Analysis (Task 2d)
-----------------------------------------------------------------------
Visualise, for 2-3 files, the log-mel spectrogram together with ground-truth
and predicted sound events over time. Use the NON-HIDDEN test files (validation
recordings that still carry annotations in their .npz) so ground truth exists.

The colour panels make success / failure cases easy to read:
  green  GT cell active
  blue   prediction active
  a class row with green-but-no-blue = a MISS; blue-but-no-green = false alarm.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data import build_feature_matrix, aggregate_labels, CLASS_NAMES
from sed_pipeline import make_sklearn_score_fn  # noqa: F401 (handy for callers)

LABEL_CFG = {"overlap_thresh": 0.0, "vote": "majority"}


def per_second(filepath, score_fn, thresholds, label_cfg=LABEL_CFG):
    """Return whole-second times, GT (T,15), pred (T,15), scores (T,15), mel (128,N_all), start_all."""
    d = dict(np.load(filepath, allow_pickle=True))
    start_all = d["start_time"]
    whole = np.isclose(start_all % 1.0, 0.0)

    X_all = build_feature_matrix(d)
    scores_all = np.asarray(score_fn(X_all), np.float32)
    thr = np.asarray(thresholds, np.float32)

    scores = scores_all[whole]
    preds = (scores >= thr).astype(int)
    gt = (aggregate_labels(d["annotations"], **label_cfg)[whole]
          if "annotations" in d else np.zeros_like(preds))
    return start_all[whole], gt, preds, scores, d["melspect_mean"].T, start_all


def plot_file(filepath, score_fn, thresholds, label_cfg=LABEL_CFG, save_to=None):
    times, gt, pred, scores, mel, start_all = per_second(
        filepath, score_fn, thresholds, label_cfg)
    fname = os.path.basename(filepath).replace(".npz", ".wav")
    C = len(CLASS_NAMES)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.4, 1.4]})

    # 1) log-mel spectrogram over all (0.5 s) segments
    ax0 = axes[0]
    extent = [start_all[0], start_all[-1] + 1, 0, mel.shape[0]]
    ax0.imshow(np.log1p(mel - mel.min() + 1e-6), aspect="auto", origin="lower",
               extent=extent, cmap="magma")
    ax0.set_ylabel("Mel bin")
    ax0.set_title(f"{fname} — log-mel spectrogram")

    # 2) ground truth, 3) predictions  (binary heatmaps on the 1 s grid)
    green = ListedColormap(["white", "#2e7d32"])
    blue  = ListedColormap(["white", "#1565c0"])
    t_edges = np.append(times, times[-1] + 1)
    c_edges = np.arange(C + 1)

    for ax, mat, cmap, title in [
        (axes[1], gt,   green, "Ground truth"),
        (axes[2], pred, blue,  "Predictions"),
    ]:
        ax.pcolormesh(t_edges, c_edges, mat.T, cmap=cmap, vmin=0, vmax=1,
                      edgecolors="lightgray", linewidth=0.3)
        ax.set_yticks(np.arange(C) + 0.5)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_ylabel(title)
        ax.set_ylim(C, 0)

    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=120)
        print(f"saved {save_to}")
    return fig


def summarise_errors(filepath, score_fn, thresholds, label_cfg=LABEL_CFG):
    """Quick per-file TP/FP/FN counts per class -- helps pick interesting files."""
    _, gt, pred, _, _, _ = per_second(filepath, score_fn, thresholds, label_cfg)
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    return {"file": os.path.basename(filepath), "TP": tp, "FP": fp, "FN": fn}


if __name__ == "__main__":
    # Demo with a dummy random score_fn on whatever annotated sample is present.
    import glob
    rng = np.random.default_rng(0)
    sf = lambda X: rng.random((X.shape[0], 15)).astype(np.float32)
    f = sorted(glob.glob("0000[67]*.npz"))[0]
    print(summarise_errors(f, sf, 0.5))
    plot_file(f, sf, 0.5, save_to="error_analysis_demo.png")