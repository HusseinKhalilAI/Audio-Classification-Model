"""
post_processing.py  --  Phase 3: Temporal Refinement (Report Task 3, 8 pts)
---------------------------------------------------------------------------
Median filtering on the per-second prediction sequence. Independent 1-second
classifications are temporally noisy: a class flickers on/off across seconds.
A median filter over a short temporal window removes isolated false activations
(and fills single-second gaps) while preserving longer events.

Investigated parameter: the filter window size W (odd: 1 = off, 3, 5, 7, ...).
We compare Segment-based Macro F1 with vs. without post-processing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from challenge_data import CLASS_NAMES, PATH_VAL
from sed_pipeline import (
    score_file, predictions_to_intervals, evaluate_sed,
)


def median_filter_preds(binary: np.ndarray, window: int) -> np.ndarray:
    """Per-class temporal median filter on a (T, C) binary matrix.

    window must be odd. window=1 returns the input unchanged (no filtering).
    Edges use 'nearest' so boundary seconds aren't dragged to zero.
    """
    if window <= 1:
        return binary
    if window % 2 == 0:
        raise ValueError("window must be odd")
    out = np.empty_like(binary)
    for c in range(binary.shape[1]):
        out[:, c] = median_filter(binary[:, c], size=window, mode="nearest")
    return out


def generate_predictions_pp(file_list, score_fn, thresholds, window=1) -> pd.DataFrame:
    """SED inference with an optional median-filter step before interval merge."""
    thr = np.asarray(thresholds, np.float32)
    rows = []
    for fp in file_list:
        scores, times, fname = score_file(fp, score_fn)
        binary = (scores >= thr).astype(int)
        binary = median_filter_preds(binary, window)
        rows.extend(predictions_to_intervals(binary, times, fname))
    if not rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(rows)


def evaluate_pp(file_list, score_fn, thresholds, ann_df, window=1):
    """Macro F1 + per-class table for a given filter window."""
    pred = generate_predictions_pp(file_list, score_fn, thresholds, window)
    return evaluate_sed(pred, file_list, ann_df)


def sweep_window(files, score_fn, thresholds, ann_df,
                 windows=(1, 3, 5, 7, 9), eval_split="local_val"):
    """Compare Macro F1 across filter windows. window=1 is the no-PP baseline."""
    ann_df = ann_df if ann_df is not None else pd.read_csv(
        os.path.join(PATH_VAL, "annotations.csv"))
    rows = []
    per_class = {}
    for w in windows:
        f1, table = evaluate_pp(files[eval_split], score_fn, thresholds, ann_df, w)
        tag = "no PP" if w == 1 else f"W={w}"
        print(f"  median window {w:>2} ({tag:6s}) -> Macro F1 = {f1:.4f}")
        rows.append({"window": w, "macro_f1": f1})
        per_class[w] = table.set_index("annotation")["f1"]

    df = pd.DataFrame(rows)
    best_w = int(df.loc[df["macro_f1"].idxmax(), "window"])
    print(f"\nBest window: {best_w}  "
          f"(no-PP {df.loc[df.window==1,'macro_f1'].values[0]:.4f} -> "
          f"{df['macro_f1'].max():.4f})")

    # Plot Macro F1 vs window
    plt.figure(figsize=(7, 4))
    plt.plot(df["window"], df["macro_f1"], marker="o")
    plt.axhline(df.loc[df.window == 1, "macro_f1"].values[0], color="gray",
                ls="--", lw=1, label="no post-processing")
    plt.xlabel("Median filter window (seconds)")
    plt.ylabel("Macro F1")
    plt.title("Phase 3 — median filtering")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig("median_filter_sweep.png", dpi=120)
    print("saved median_filter_sweep.png")

    # Which classes benefit most (best window vs no-PP)
    delta = (per_class[best_w] - per_class[1]).sort_values(ascending=False)
    return {"results": df, "best_window": best_w,
            "per_class_delta": delta, "per_class_f1": per_class}


# =========================================================================== #
# Fair variant: temporal SMOOTHING on the score sequence + threshold RE-TUNING
# =========================================================================== #
from scipy.ndimage import uniform_filter1d
from challenge_data import build_feature_matrix, aggregate_labels
from sed_pipeline import score_file as _score_file, tune_thresholds


def smooth_scores(scores: np.ndarray, window: int, mode: str = "mean") -> np.ndarray:
    """Per-class temporal smoothing of the (T, C) SCORE sequence (floats).

    mode='mean'   -> moving average (uniform_filter1d)
    mode='median' -> rolling median on the probabilities
    Unlike filtering binary predictions, this softens confidence rather than
    hard-deleting isolated detections.
    """
    if window <= 1:
        return scores
    out = np.empty_like(scores)
    for c in range(scores.shape[1]):
        if mode == "mean":
            out[:, c] = uniform_filter1d(scores[:, c], size=window, mode="nearest")
        elif mode == "median":
            out[:, c] = median_filter(scores[:, c], size=window, mode="nearest")
        else:
            raise ValueError(f"unknown mode: {mode}")
    return out


def _collect_smoothed(file_list, score_fn, window, mode, label_cfg):
    """Whole-second (smoothed scores, labels) stacked across files, for tuning.
    Smoothing is applied PER FILE so it never crosses recording boundaries."""
    S, Y = [], []
    for fp in file_list:
        d = dict(np.load(fp, allow_pickle=True))
        whole = np.isclose(d["start_time"] % 1.0, 0.0)
        sc = np.asarray(score_fn(build_feature_matrix(d)), np.float32)[whole]
        S.append(smooth_scores(sc, window, mode))
        Y.append(aggregate_labels(d["annotations"], **label_cfg)[whole])
    return np.vstack(S), np.vstack(Y)


def generate_predictions_smooth(file_list, score_fn, thresholds, window, mode="mean"):
    """SED inference with per-file score smoothing before thresholding."""
    thr = np.asarray(thresholds, np.float32)
    rows = []
    for fp in file_list:
        scores, times, fname = _score_file(fp, score_fn)
        scores = smooth_scores(scores, window, mode)
        binary = (scores >= thr).astype(int)
        rows.extend(predictions_to_intervals(binary, times, fname))
    if not rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(rows)


def sweep_smoothing(files, score_fn, ann_df, label_cfg,
                    windows=(1, 3, 5, 7), mode="mean", eval_split="local_val"):
    """For each window: RE-TUNE thresholds on smoothed local-val scores, then
    score. window=1 reproduces the no-post-processing result. This is the fair
    comparison the median-on-binary sweep lacked."""
    rows, per_class, thr_by_w = [], {}, {}
    for w in windows:
        S, Y = _collect_smoothed(files["local_val"], score_fn, w, mode, label_cfg)
        thr_w = tune_thresholds(S, Y)
        pred = generate_predictions_smooth(files[eval_split], score_fn, thr_w, w, mode)
        f1, table = evaluate_sed(pred, files[eval_split], ann_df)
        tag = "no PP" if w == 1 else f"W={w}"
        print(f"  smooth({mode}) window {w:>2} ({tag:6s}) -> Macro F1 = {f1:.4f}  (re-tuned)")
        rows.append({"window": w, "macro_f1": f1})
        per_class[w] = table.set_index("annotation")["f1"]
        thr_by_w[w] = thr_w

    df = pd.DataFrame(rows)
    best_w = int(df.loc[df["macro_f1"].idxmax(), "window"])
    base = df.loc[df.window == 1, "macro_f1"].values[0]
    print(f"\nBest window: {best_w}  (no-PP {base:.4f} -> {df['macro_f1'].max():.4f}, "
          f"delta {df['macro_f1'].max() - base:+.4f})")

    plt.figure(figsize=(7, 4))
    plt.plot(df["window"], df["macro_f1"], marker="o")
    plt.axhline(base, color="gray", ls="--", lw=1, label="no post-processing")
    plt.xlabel(f"{mode} smoothing window (seconds)")
    plt.ylabel("Macro F1"); plt.title(f"Phase 3 — score smoothing ({mode}) + re-tuned thresholds")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"smoothing_{mode}_sweep.png", dpi=120)
    print(f"saved smoothing_{mode}_sweep.png")

    delta = (per_class[best_w] - per_class[1]).sort_values(ascending=False)
    return {"results": df, "best_window": best_w, "mode": mode,
            "thresholds": thr_by_w[best_w], "per_class_delta": delta}


if __name__ == "__main__":
    # Demo with a dummy score_fn on sample files (plumbing check).
    import glob
    rng = np.random.default_rng(0)
    sf = lambda X: rng.random((X.shape[0], 15)).astype(np.float32)
    b = (np.asarray(sf(np.zeros((10, 960)))) >= 0.5).astype(int)
    print("median_filter_preds shape ok:", median_filter_preds(b, 3).shape)
    print("smooth_scores shape ok:",
          smooth_scores(sf(np.zeros((10, 960))), 3, "mean").shape)