"""
random_forest.py  --  Phase 2: Simple Classifiers (Report Task 2, 10 pts)
--------------------------------------------------------------------------
Replace the decision-tree baseline with a Random Forest and evaluate it on the
official Segment-based Macro F1. Mirrors the previous project's RF()/sweep_rf()
idiom, but every config tunes per-class thresholds and is scored by Macro F1.

class_weight='balanced' is ON by default — it reweights each binary tree toward
the rare class, which directly helps the macro (per-class-equal) F1.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier

from challenge_data import (
    list_split_files, construct_Xy, PATH_VAL, CLASS_NAMES,
)
from sed_pipeline import evaluate_model

LABEL_CFG = {"overlap_thresh": 0.0, "vote": "majority"}   # eval-aligned
MAX_TRAIN_SEGMENTS = 50_000


# --------------------------------------------------------------------------- #
# Shared data load (once per session)
# --------------------------------------------------------------------------- #
def load_training_data():
    files = list_split_files()
    rng = np.random.default_rng(42)
    X, Y = construct_Xy(files["train"], **LABEL_CFG, verbose=True)
    if MAX_TRAIN_SEGMENTS and X.shape[0] > MAX_TRAIN_SEGMENTS:
        idx = rng.choice(X.shape[0], MAX_TRAIN_SEGMENTS, replace=False)
        X, Y = X[idx], Y[idx]
    ann_df = pd.read_csv(os.path.join(PATH_VAL, "annotations.csv"))
    return X, Y, files, ann_df


# --------------------------------------------------------------------------- #
# Train + evaluate one RF config
# --------------------------------------------------------------------------- #
def RF(X_train, y_train, files, ann_df,
       n_estimators=100, max_depth=None, min_samples_leaf=1,
       class_weight="balanced", random_state=42, eval_split="local_val"):
    print(f"Training RF (n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_leaf={min_samples_leaf}, class_weight={class_weight})...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight=class_weight,
        n_jobs=-1, random_state=random_state,
    )
    rf.fit(X_train, y_train)
    print(f"  train {time.time() - t0:.1f}s -- tuning thresholds + scoring F1...")

    res = evaluate_model(rf, files, ann_df, LABEL_CFG, eval_split=eval_split)
    res["model"] = rf
    res["hyperparams"] = {"n_estimators": n_estimators, "max_depth": max_depth,
                          "min_samples_leaf": min_samples_leaf,
                          "class_weight": class_weight}
    print(f"  RF Macro F1 ({eval_split}): {res['macro_f1']:.4f}")
    return res


# --------------------------------------------------------------------------- #
# Two-hyperparameter sweep (Task 2b): n_estimators then max_depth
# --------------------------------------------------------------------------- #
def sweep_rf(X_train, y_train, files, ann_df):
    # 1) n_estimators (max_depth=None)
    print("=== RF sweep 1: n_estimators (max_depth=None) ===")
    n_grid = [50, 100, 200, 400]
    sweep1 = [RF(X_train, y_train, files, ann_df, n_estimators=n, max_depth=None)
              for n in n_grid]
    best_n = max(sweep1, key=lambda r: r["macro_f1"])["hyperparams"]["n_estimators"]
    print(f"\nBest n_estimators: {best_n}")

    # 2) max_depth (at best n_estimators)
    print(f"\n=== RF sweep 2: max_depth (n_estimators={best_n}) ===")
    d_grid = [10, 20, 30, 50, None]
    sweep2 = [RF(X_train, y_train, files, ann_df, n_estimators=best_n, max_depth=d)
              for d in d_grid]
    best_d = max(sweep2, key=lambda r: r["macro_f1"])["hyperparams"]["max_depth"]
    print(f"\nBest max_depth: {best_d}")

    # Plot both sweeps (Macro F1 on local validation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r["hyperparams"]["n_estimators"] for r in sweep1],
                 [r["macro_f1"] for r in sweep1], marker="o")
    axes[0].set_xlabel("n_estimators"); axes[0].set_title("RF sweep 1 (max_depth=None)")
    axes[1].plot([str(r["hyperparams"]["max_depth"]) for r in sweep2],
                 [r["macro_f1"] for r in sweep2], marker="o")
    axes[1].set_xlabel("max_depth"); axes[1].set_title(f"RF sweep 2 (n_est={best_n})")
    for ax in axes:
        ax.set_ylabel("Val Macro F1"); ax.grid(True)
    plt.tight_layout(); plt.savefig("rf_hp_search.png", dpi=120)
    print("saved rf_hp_search.png")

    joblib.dump({"sweep1_n_estimators": [_slim(r) for r in sweep1],
                 "sweep2_max_depth":    [_slim(r) for r in sweep2]},
                "rf_hp_search.joblib")
    return {"best_n_estimators": best_n, "best_max_depth": best_d}


def _slim(res):
    """Drop the heavy model before pickling sweep results."""
    return {k: v for k, v in res.items() if k not in ("model", "score_fn")}


if __name__ == "__main__":
    X, Y, files, ann_df = load_training_data()
    best = sweep_rf(X, Y, files, ann_df)
    print("\nBest RF config:", best)




""" 
---------------------------------------------------Results---------------------------------------------------
=== RF sweep 1: n_estimators (max_depth=None) ===
Training RF (n_estimators=50, max_depth=None, min_samples_leaf=1, class_weight=balanced)...
  train 41.2s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4539
Training RF (n_estimators=100, max_depth=None, min_samples_leaf=1, class_weight=balanced)...
  train 94.1s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4701
Training RF (n_estimators=200, max_depth=None, min_samples_leaf=1, class_weight=balanced)...
  train 155.9s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4790
Training RF (n_estimators=400, max_depth=None, min_samples_leaf=1, class_weight=balanced)...
  train 347.6s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4811

Best n_estimators: 400

=== RF sweep 2: max_depth (n_estimators=400) ===
Training RF (n_estimators=400, max_depth=10, min_samples_leaf=1, class_weight=balanced)...
  train 214.3s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.3897
Training RF (n_estimators=400, max_depth=20, min_samples_leaf=1, class_weight=balanced)...
  train 264.2s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4548
Training RF (n_estimators=400, max_depth=30, min_samples_leaf=1, class_weight=balanced)...
  train 397.5s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4784
Training RF (n_estimators=400, max_depth=50, min_samples_leaf=1, class_weight=balanced)...
  train 363.5s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4806
Training RF (n_estimators=400, max_depth=None, min_samples_leaf=1, class_weight=balanced)...
  train 435.0s -- tuning thresholds + scoring F1...
  RF Macro F1 (local_val): 0.4811

Best max_depth: None
saved rf_hp_search.png

Best RF config: {'best_n_estimators': 400, 'best_max_depth': None}
---------------------------------------------------------------------------------------------------------------
"""