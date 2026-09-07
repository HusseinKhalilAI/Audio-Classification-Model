"""
make_submission.py
------------------
Generate the hidden-test prediction CSV with the locked best system
(XGB + mean smoothing W=3 + re-tuned thresholds = 0.602 on non-hidden test).

Loads best_model.joblib, runs SED inference on every hidden test recording,
applies the SAME post-processing used in evaluation, then validates the CSV
format against evaluate.py's rules before writing.

Run:  python make_submission.py
Output: submission.csv  (columns: filename,annotation,onset,offset)
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd

from challenge_data import PATH_TEST, CLASS_NAMES
from sed_pipeline import make_sklearn_score_fn
from Post_processing import generate_predictions_smooth, generate_predictions_pp

OUTPUT_CSV = "submission.csv"


def load_system():
    saved = joblib.load("best_model.joblib")
    base = make_sklearn_score_fn(saved["model"])
    mode = saved.get("pp_mode")
    window = saved.get("pp_window", 1)
    # post-processing path uses the thresholds that were re-tuned WITH smoothing
    thr = saved.get("pp_thresholds", saved["thresholds"])
    desc = (f"{saved.get('which','best')} + {mode} smoothing W={window}"
            if mode and window > 1 else f"{saved.get('which','best')} (no PP)")
    print(f"Loaded system: {desc}")
    print(f"  non-hidden F1 (raw)  = {saved.get('macro_f1_non_hidden'):.4f}")
    if saved.get("macro_f1_pp_non_hidden"):
        print(f"  non-hidden F1 (+PP)  = {saved['macro_f1_pp_non_hidden']:.4f}")
    return base, thr, mode, window


def validate(df: pd.DataFrame, hidden_files):
    """Mirror evaluate.py's input checks so submission can't be rejected."""
    required = {"filename", "annotation", "onset", "offset"}
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"
    assert (df["offset"] >= df["onset"]).all(), "found offset < onset"
    assert (df["onset"] >= 0).all() and (df["offset"] >= 0).all(), "negative times"
    bad = set(df["annotation"].unique()) - set(CLASS_NAMES)
    assert not bad, f"unexpected classes: {bad}"

    # every hidden file should appear (files with no events are allowed to be
    # absent, but warn so we notice if a whole split silently dropped out)
    pred_files = set(df["filename"].unique())
    all_files = {os.path.basename(f).replace(".npz", ".wav") for f in hidden_files}
    missing = all_files - pred_files
    if missing:
        print(f"  note: {len(missing)} files have no predicted events "
              f"(allowed; omitted from CSV)")
    print("  format validation passed.")


if __name__ == "__main__":
    score_fn, thr, mode, window = load_system()

    hidden_files = sorted(glob.glob(os.path.join(PATH_TEST, "audio_features", "*.npz")))
    print(f"\nGenerating predictions for {len(hidden_files)} hidden test recordings...")

    if mode and window > 1:
        pred = generate_predictions_smooth(hidden_files, score_fn, thr, window, mode)
    else:
        pred = generate_predictions_pp(hidden_files, score_fn, thr)

    pred = pred.sort_values(["filename", "annotation", "onset"]).reset_index(drop=True)

    print(f"  {len(pred)} event intervals across "
          f"{pred['filename'].nunique()} files.")
    print("\nValidating submission format...")
    validate(pred, hidden_files)

    pred.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved {OUTPUT_CSV}  ({len(pred)} rows)")
    print("\nPer-class predicted interval counts:")
    print(pred["annotation"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    print("\nFirst rows:")
    print(pred.head(8).to_string(index=False))







"""
Per-class predicted interval counts:
annotation
bell_ringing                    62
coffee_machine                  78
cutlery_dishes                 471
door_open_close                583
footsteps                     1073
keyboard_typing                353
keychain                       301
light_switch                   164
microwave                      194
phone_ringing                  317
running_water                  431
toilet_flushing                121
vacuum_cleaner                 164
wardrobe_drawer_open_close     290
window_open_close              134

First rows:
  filename      annotation  onset  offset
000020.wav  cutlery_dishes    0.0     5.0
000020.wav door_open_close    0.0     3.0
000020.wav door_open_close    6.0     9.0
000020.wav door_open_close   21.0    22.0
000020.wav door_open_close   27.0    29.0
000020.wav       footsteps    5.0    12.0
000020.wav       footsteps   13.0    21.0
000020.wav       footsteps   27.0    31.0

"""