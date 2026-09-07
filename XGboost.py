"""
xgboost_clf.py  --  Phase 2: Simple Classifiers (Report Task 2, 10 pts)
-----------------------------------------------------------------------
Second classical model: gradient-boosted trees (XGBoost), one binary booster
per class via MultiOutputClassifier. Same F1-with-threshold-tuning loop as RF.

Note on imbalance: XGBoost has no class_weight. We leave scale_pos_weight at
its default and rely on per-class threshold tuning to set the operating point,
which is exactly what the Macro F1 metric rewards. tree_method='hist' keeps
training fast on the 50k x 960 matrix.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

from sed_pipeline import evaluate_model
from RandomForest import load_training_data, LABEL_CFG, _slim


def XGB(X_train, y_train, files, ann_df,
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8, random_state=42,
        eval_split="local_val"):
    print(f"Training XGB (n_estimators={n_estimators}, max_depth={max_depth}, "
          f"lr={learning_rate})...")
    t0 = time.time()
    base = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=random_state,
    )
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"  train {time.time() - t0:.1f}s -- tuning thresholds + scoring F1...")

    res = evaluate_model(clf, files, ann_df, LABEL_CFG, eval_split=eval_split)
    res["model"] = clf
    res["hyperparams"] = {"n_estimators": n_estimators, "max_depth": max_depth,
                          "learning_rate": learning_rate}
    print(f"  XGB Macro F1 ({eval_split}): {res['macro_f1']:.4f}")
    return res


def sweep_xgb(X_train, y_train, files, ann_df):
    # 1) max_depth (lr=0.1, n_estimators=300)
    print("=== XGB sweep 1: max_depth (lr=0.1, n_estimators=300) ===")
    depth_grid = [4, 6, 8, 10]
    sweep1 = [XGB(X_train, y_train, files, ann_df, max_depth=d) for d in depth_grid]
    best_d = max(sweep1, key=lambda r: r["macro_f1"])["hyperparams"]["max_depth"]
    print(f"\nBest max_depth: {best_d}")

    # 2) learning_rate (at best max_depth)
    print(f"\n=== XGB sweep 2: learning_rate (max_depth={best_d}) ===")
    lr_grid = [0.03, 0.1, 0.2, 0.3]
    sweep2 = [XGB(X_train, y_train, files, ann_df, max_depth=best_d, learning_rate=lr)
              for lr in lr_grid]
    best_lr = max(sweep2, key=lambda r: r["macro_f1"])["hyperparams"]["learning_rate"]
    print(f"\nBest learning_rate: {best_lr}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r["hyperparams"]["max_depth"] for r in sweep1],
                 [r["macro_f1"] for r in sweep1], marker="o")
    axes[0].set_xlabel("max_depth"); axes[0].set_title("XGB sweep 1 (lr=0.1)")
    axes[1].plot([r["hyperparams"]["learning_rate"] for r in sweep2],
                 [r["macro_f1"] for r in sweep2], marker="o")
    axes[1].set_xlabel("learning_rate"); axes[1].set_title(f"XGB sweep 2 (depth={best_d})")
    for ax in axes:
        ax.set_ylabel("Val Macro F1"); ax.grid(True)
    plt.tight_layout(); plt.savefig("xgb_hp_search.png", dpi=120)
    print("saved xgb_hp_search.png")

    joblib.dump({"sweep1_max_depth":     [_slim(r) for r in sweep1],
                 "sweep2_learning_rate": [_slim(r) for r in sweep2]},
                "xgb_hp_search.joblib")
    return {"best_max_depth": best_d, "best_learning_rate": best_lr}


if __name__ == "__main__":
    X, Y, files, ann_df = load_training_data()
    best = sweep_xgb(X, Y, files, ann_df)
    print("\nBest XGB config:", best)



"""
--------------------------------------------------------Results---------------------------------------------------------------
=== XGB sweep 1: max_depth (lr=0.1, n_estimators=300) ===
Training XGB (n_estimators=300, max_depth=4, lr=0.1)...
  train 305.9s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5704
Training XGB (n_estimators=300, max_depth=6, lr=0.1)...
  train 635.7s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5731
Training XGB (n_estimators=300, max_depth=8, lr=0.1)...
  train 958.0s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5748
Training XGB (n_estimators=300, max_depth=10, lr=0.1)...
  train 1332.6s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5720

Best max_depth: 8

=== XGB sweep 2: learning_rate (max_depth=8) ===
Training XGB (n_estimators=300, max_depth=8, lr=0.03)...
  train 1061.6s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5713
Training XGB (n_estimators=300, max_depth=8, lr=0.1)...
  train 692.8s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5748
Training XGB (n_estimators=300, max_depth=8, lr=0.2)...
  train 567.5s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5649
Training XGB (n_estimators=300, max_depth=8, lr=0.3)...
  train 527.3s -- tuning thresholds + scoring F1...
  XGB Macro F1 (local_val): 0.5563

Best learning_rate: 0.1
saved xgb_hp_search.png

Best XGB config: {'best_max_depth': 8, 'best_learning_rate': 0.1}

------------------------------------------------------------------------------------------------------------------------------
"""