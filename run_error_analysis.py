"""
run_error_analysis.py  --  Phase 2(d): Qualitative Error Analysis (Task 2d)
---------------------------------------------------------------------------
Load the saved best model (no retraining), rank non-hidden test files by their
error profile, and plot spectrogram + ground-truth vs predictions for a few
interesting cases: one clean success, one with misses, one with false alarms.

Uses the SAME post-processing the submission uses (if it helped), so the plots
reflect the final system. Run:  python run_error_analysis.py
"""

import joblib
import numpy as np
import pandas as pd

from RandomForest import load_training_data, LABEL_CFG
from sed_pipeline import make_sklearn_score_fn, score_file
from post_processing import smooth_scores
from data import aggregate_labels, build_feature_matrix, CLASS_NAMES
from error_analysis import plot_file, per_second
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

N_PLOTS = 3
 
 
def build_scorer(saved):
    """Return a score_fn that already includes the chosen post-processing,
    so error plots match the final submission system."""
    base = make_sklearn_score_fn(saved["model"])
    pp_mode = saved.get("pp_mode")
    pp_window = saved.get("pp_window", 1)
    if pp_mode and pp_window > 1:
        def score_fn(X_all):
            return smooth_scores(np.asarray(base(X_all), np.float32), pp_window, pp_mode)
        thr = saved.get("pp_thresholds", saved["thresholds"])
        print(f"scorer: {saved.get('which','best')} + {pp_mode} smoothing W={pp_window}")
        return score_fn, thr
    print(f"scorer: {saved.get('which','best')} (no post-processing)")
    return base, saved["thresholds"]
 
 
def rank_files(file_list, score_fn, thresholds):
    """Per-file TP/FP/FN + F1 over the whole-second grid, for picking examples."""
    rows = []
    for fp in file_list:
        d = dict(np.load(fp, allow_pickle=True))
        whole = np.isclose(d["start_time"] % 1.0, 0.0)
        sc = np.asarray(score_fn(build_feature_matrix(d)), np.float32)[whole]
        pred = (sc >= np.asarray(thresholds, np.float32)).astype(int)
        gt = aggregate_labels(d["annotations"], **LABEL_CFG)[whole]
        n_tp = int(((gt == 1) & (pred == 1)).sum())
        n_fp = int(((gt == 0) & (pred == 1)).sum())
        n_fn = int(((gt == 1) & (pred == 0)).sum())
        n_active = int(gt.sum())
        rows.append({"path": fp, "file": os.path.basename(fp),
                     "TP": n_tp, "FP": n_fp, "FN": n_fn, "gt_active": n_active})
    df = pd.DataFrame(rows)
    df["precision"] = df.TP / (df.TP + df.FP).replace(0, np.nan)
    df["recall"]    = df.TP / (df.TP + df.FN).replace(0, np.nan)
    return df
 
 
if __name__ == "__main__":
    saved = joblib.load("best_model.joblib")
    score_fn, thr = build_scorer(saved)
    _, _, files, _ = load_training_data()
 
    print("Ranking non-hidden files by error profile...")
    df = rank_files(files["non_hidden"], score_fn, thr)
    df = df[df["gt_active"] >= 5]            # skip near-empty recordings
 
    # Pick three contrasting cases for the report
    success   = df.sort_values("precision", ascending=False).iloc[0]
    misses    = df.sort_values("FN", ascending=False).iloc[0]
    false_pos = df.sort_values("FP", ascending=False).iloc[0]
    picks = {"success": success, "misses": misses, "false_alarms": false_pos}
 
    print("\nChosen example files:")
    for label, row in picks.items():
        print(f"  {label:12s}: {row['file']}  "
              f"TP={row['TP']} FP={row['FP']} FN={row['FN']} "
              f"P={row['precision']:.2f} R={row['recall']:.2f}")
 
    for label, row in picks.items():
        out = f"errplot_{label}_{row['file'].replace('.wav','')}.png"
        plot_file(row["path"], score_fn, thr, label_cfg=LABEL_CFG, save_to=out)
 
