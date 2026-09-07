"""
run_best_model.py
-----------------
Lock the best classical systems end to end:

  1. Retrain best RF and best XGB on train.
  2. Tune per-class thresholds on local-val; report HONEST Macro F1 on the
     non-hidden test set for BOTH (the unbiased final estimate).
  3. Plot per-class F1 on the test set: baseline vs RF vs XGB (Task 2c figure).
  4. Persist both models (best_rf.joblib, best_xgb.joblib) and the overall
     winner (best_model.joblib) so later phases reuse them without retraining.
  5. Run the Phase 3 median-filter window sweep on the winner.

Run:  python run_best_model.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import CLASS_NAMES
from RandomForest import load_training_data, LABEL_CFG, RF
from XGboost import XGB
from sed_pipeline import make_sklearn_score_fn
from post_processing import sweep_window, evaluate_pp

BEST_RF  = dict(n_estimators=400, max_depth=None)              # class_weight='balanced' default
BEST_XGB = dict(n_estimators=300, max_depth=8, learning_rate=0.1)

# Baseline per-class F1 on the non-hidden test set (decision-tree, untuned).
# From the provided notebook run; used only as a reference series in the plot.
BASELINE_F1 = {
    "bell_ringing": 0.288, "coffee_machine": 0.362, "cutlery_dishes": 0.287,
    "door_open_close": 0.177, "footsteps": 0.333, "keyboard_typing": 0.359,
    "keychain": 0.321, "light_switch": 0.234, "microwave": 0.390,
    "phone_ringing": 0.485, "running_water": 0.578, "toilet_flushing": 0.290,
    "vacuum_cleaner": 0.470, "wardrobe_drawer_open_close": 0.120,
    "window_open_close": 0.061,
}
BASELINE_MACRO = 0.317


def f1_series(res):
    """Per-class F1 aligned to CLASS_NAMES order, from an evaluate_model result."""
    s = res["per_class"].set_index("annotation")["f1"]
    return s.reindex(CLASS_NAMES)


def plot_comparison(rf_res, xgb_res, save_to="model_comparison_test.png"):
    base = pd.Series(BASELINE_F1).reindex(CLASS_NAMES)
    rf   = f1_series(rf_res)
    xgb  = f1_series(xgb_res)

    x = np.arange(len(CLASS_NAMES)); w = 0.27
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, base, w, label=f"Baseline (macro {BASELINE_MACRO:.3f})", color="#bdbdbd")
    ax.bar(x,     rf,   w, label=f"RF (macro {rf_res['macro_f1']:.3f})",   color="#1565c0")
    ax.bar(x + w, xgb,  w, label=f"XGB (macro {xgb_res['macro_f1']:.3f})", color="#2e7d32")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Segment F1 (non-hidden test)"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-class F1 on the non-hidden test set — baseline vs RF vs XGB")
    ax.legend()
    plt.tight_layout(); plt.savefig(save_to, dpi=120)
    print(f"saved {save_to}")


if __name__ == "__main__":
    X, Y, files, ann_df = load_training_data()

    # 1+2: retrain both, score on the NON-HIDDEN test set (honest estimate)
    print("\n>>> Retraining best RF on non-hidden test...")
    rf_res = RF(X, Y, files, ann_df, eval_split="non_hidden", **BEST_RF)
    print("\n>>> Retraining best XGB on non-hidden test...")
    xgb_res = XGB(X, Y, files, ann_df, eval_split="non_hidden", **BEST_XGB)

    print("\n=== Non-hidden test Macro F1 ===")
    print(f"  RF : {rf_res['macro_f1']:.4f}")
    print(f"  XGB: {xgb_res['macro_f1']:.4f}")

    # 3: comparison figure (Task 2c)
    plot_comparison(rf_res, xgb_res)

    # 4: save both, and the overall winner as the submission model
    for tag, res, cfg in [("rf", rf_res, BEST_RF), ("xgb", xgb_res, BEST_XGB)]:
        joblib.dump({"model": res["model"], "thresholds": res["thresholds"],
                     "hyperparams": res["hyperparams"], "label_cfg": LABEL_CFG,
                     "macro_f1_non_hidden": res["macro_f1"]}, f"best_{tag}.joblib")
        print(f"saved best_{tag}.joblib")

    winner = xgb_res if xgb_res["macro_f1"] >= rf_res["macro_f1"] else rf_res
    wtag = "xgb" if winner is xgb_res else "rf"
    joblib.dump({"model": winner["model"], "thresholds": winner["thresholds"],
                 "hyperparams": winner["hyperparams"], "label_cfg": LABEL_CFG,
                 "which": wtag, "macro_f1_non_hidden": winner["macro_f1"]},
                "best_model.joblib")
    print(f"\nWinner: {wtag.upper()} ({winner['macro_f1']:.4f}) -> saved best_model.joblib")

    # 5: Phase 3 median-filter sweep on the winner
    score_fn = make_sklearn_score_fn(winner["model"])
    thr = winner["thresholds"]
    print("\n--- Phase 3: median-filter window sweep (local validation) ---")
    pp = sweep_window(files, score_fn, thr, ann_df,
                      windows=(1, 3, 5, 7, 9), eval_split="local_val")
    best_w = pp["best_window"]
    print("\nPer-class F1 change at best window vs. no post-processing:")
    print(pp["per_class_delta"].round(3).to_string())

    f1_nopp, _ = evaluate_pp(files["non_hidden"], score_fn, thr, ann_df, window=1)
    f1_best, _ = evaluate_pp(files["non_hidden"], score_fn, thr, ann_df, window=best_w)
    print(f"\n=== Non-hidden test: no-PP {f1_nopp:.4f} -> "
          f"W={best_w} {f1_best:.4f}  (delta {f1_best - f1_nopp:+.4f}) ===")

    saved = joblib.load("best_model.joblib")
    saved["best_window"] = int(best_w)
    saved["macro_f1_pp_non_hidden"] = float(f1_best)
    joblib.dump(saved, "best_model.joblib")
    print("updated best_model.joblib with best_window =", best_w)

"""
>>> Retraining best RF on non-hidden test...
Training RF (n_estimators=400, max_depth=None, min_samples_leaf=1, class_weight=balanced)...
  train 311.1s -- tuning thresholds + scoring F1...
  RF Macro F1 (non_hidden): 0.4829

>>> Retraining best XGB on non-hidden test...
Training XGB (n_estimators=300, max_depth=8, lr=0.1)...
  train 593.2s -- tuning thresholds + scoring F1...
  XGB Macro F1 (non_hidden): 0.5656

=== Non-hidden test Macro F1 ===
  RF : 0.4829
  XGB: 0.5656
saved model_comparison_test.png
saved best_rf.joblib
saved best_xgb.joblib

Winner: XGB (0.5656) -> saved best_model.joblib

--- Phase 3: median-filter window sweep (local validation) ---
  median window  1 (no PP ) -> Macro F1 = 0.5748
  median window  3 (W=3   ) -> Macro F1 = 0.5653
  median window  5 (W=5   ) -> Macro F1 = 0.5339
  median window  7 (W=7   ) -> Macro F1 = 0.5070
  median window  9 (W=9   ) -> Macro F1 = 0.4802

Best window: 1  (no-PP 0.5748 -> 0.5748)
saved median_filter_sweep.png

Per-class F1 change at best window vs. no post-processing:
annotation
bell_ringing                  0.0
coffee_machine                0.0
cutlery_dishes                0.0
door_open_close               0.0
footsteps                     0.0
keyboard_typing               0.0
keychain                      0.0
light_switch                  0.0
microwave                     0.0
phone_ringing                 0.0
running_water                 0.0
toilet_flushing               0.0
vacuum_cleaner                0.0
wardrobe_drawer_open_close    0.0
window_open_close             0.0

=== Non-hidden test: no-PP 0.5656 -> W=1 0.5656  (delta +0.0000) ===
updated best_model.joblib with best_window = 1
"""