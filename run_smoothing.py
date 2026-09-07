"""
run_postproc.py
---------------
Phase 3, the FAIR version: load the saved best model (no retraining), and sweep
temporal SCORE smoothing with threshold RE-TUNING per window. Tries both mean
and median smoothing. window=1 reproduces the no-post-processing number.

Run:  python run_postproc.py
"""

import joblib
import numpy as np

from RandomForest import load_training_data, LABEL_CFG
from sed_pipeline import make_sklearn_score_fn
from Post_processing import sweep_smoothing, generate_predictions_smooth
from sed_pipeline import evaluate_sed

if __name__ == "__main__":
    saved = joblib.load("best_model.joblib")
    model = saved["model"]
    score_fn = make_sklearn_score_fn(model)
    print(f"Loaded {saved.get('which', 'best')} model "
          f"(non-hidden F1 = {saved['macro_f1_non_hidden']:.4f})")

    # We need the file lists + ann_df; load_training_data also returns X,Y (ignored)
    _, _, files, ann_df = load_training_data()

    best_overall = None
    for mode in ("mean", "median"):
        print(f"\n--- score smoothing: {mode} (re-tuned thresholds, local val) ---")
        res = sweep_smoothing(files, score_fn, ann_df, LABEL_CFG,
                              windows=(1, 3, 5, 7), mode=mode, eval_split="local_val")
        if best_overall is None or res["results"]["macro_f1"].max() > \
                best_overall["results"]["macro_f1"].max():
            best_overall = res

    # Confirm the best (mode, window) on the non-hidden test set
    w = best_overall["best_window"]; mode = best_overall["mode"]
    thr = best_overall["thresholds"]
    pred = generate_predictions_smooth(files["non_hidden"], score_fn, thr, w, mode)
    f1_pp, _ = evaluate_sed(pred, files["non_hidden"], ann_df)
    print(f"\n=== Non-hidden test: best PP = {mode} W={w} -> Macro F1 {f1_pp:.4f} "
          f"(no-PP was {saved['macro_f1_non_hidden']:.4f}) ===")

    if f1_pp > saved["macro_f1_non_hidden"]:
        saved.update({"pp_mode": mode, "pp_window": int(w),
                      "pp_thresholds": thr, "macro_f1_pp_non_hidden": float(f1_pp)})
        joblib.dump(saved, "best_model.joblib")
        print("post-processing helped -> saved to best_model.joblib")
    else:
        print("post-processing did not help on non-hidden -> keeping raw model")