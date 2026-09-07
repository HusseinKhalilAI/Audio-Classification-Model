"""
sed_pipeline.py
---------------
The shared SED backbone for the challenge. Every model (RF, XGBoost, and the
bonus LSTM/CRNN) plugs into this. It is the piece the previous project did not
have: it turns per-segment SCORES into the challenge's onset/offset intervals
and computes the official Segment-based Macro F1 via the provided evaluate.py.

Flow:  score_fn -> per-second scores -> per-class threshold -> binary
        -> merge into intervals -> CSV  ->  evaluate.py (Macro F1)

A `score_fn` maps a file's full feature matrix X_all (N_all, 960) to per-segment
class scores (N_all, 15) in [0, 1]. RF/XGB wrap predict_proba; the sequence
models wrap a forward pass. This keeps inference uniform across all models.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Tuple

from sklearn.metrics import f1_score

from challenge_data import build_feature_matrix, CLASS_NAMES, SEGMENT_LENGTH
# Reuse the OFFICIAL evaluation helpers so our local score == the challenge score.
from evaluate import (
    aggregate_ground_truth_annotations,
    build_segment_frame_from_intervals,
    calculate_f1_score,
)

ScoreFn = Callable[[np.ndarray], np.ndarray]   # (N_all, 960) -> (N_all, 15)


# --------------------------------------------------------------------------- #
# 1. Per-file scoring (whole-second segments only, like the baseline)
# --------------------------------------------------------------------------- #
def score_file(filepath: str, score_fn: ScoreFn) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (scores[T_ws, 15], start_times[T_ws], filename) for one recording.

    Scores are produced for every segment, then we keep only the whole-second
    ones (t = 0,1,2,...) to match the 1-second evaluation grid.
    """
    data = dict(np.load(filepath, allow_pickle=True))
    start_all = data["start_time"]
    X_all = build_feature_matrix(data)
    scores_all = np.asarray(score_fn(X_all), dtype=np.float32)   # (N_all, 15)

    whole = np.isclose(start_all % 1.0, 0.0)
    fname = os.path.basename(filepath).replace(".npz", ".wav")
    return scores_all[whole], start_all[whole], fname


# --------------------------------------------------------------------------- #
# 2. Scores -> binary -> intervals
# --------------------------------------------------------------------------- #
def predictions_to_intervals(preds: np.ndarray, start_times: np.ndarray,
                             filename: str) -> List[Dict]:
    """Merge consecutive active whole-second segments per class into intervals."""
    rows = []
    for c, cls in enumerate(CLASS_NAMES):
        col = preds[:, c]
        in_event, onset = False, None
        for t, p in zip(start_times, col):
            if p == 1 and not in_event:
                onset, in_event = float(t), True
            elif p == 0 and in_event:
                rows.append({"filename": filename, "annotation": cls,
                             "onset": onset, "offset": float(t)})
                in_event = False
        if in_event:
            rows.append({"filename": filename, "annotation": cls,
                         "onset": onset, "offset": float(start_times[-1]) + SEGMENT_LENGTH})
    return rows


def generate_predictions(file_list: List[str], score_fn: ScoreFn,
                         thresholds) -> pd.DataFrame:
    """Full SED inference over many files -> prediction DataFrame.

    `thresholds` is a scalar or a (15,) per-class vector applied to the scores.
    """
    thr = np.asarray(thresholds, dtype=np.float32)
    rows = []
    for fp in file_list:
        scores, times, fname = score_file(fp, score_fn)
        binary = (scores >= thr).astype(int)
        rows.extend(predictions_to_intervals(binary, times, fname))
    if not rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Official Segment-based Macro F1 (wraps evaluate.py)
# --------------------------------------------------------------------------- #
def evaluate_sed(pred_df: pd.DataFrame, file_list: List[str],
                 ann_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """Macro F1 + per-class table for predictions on a labelled split.

    Replicates the baseline's evaluate_split: filter annotations to this split,
    majority-vote aggregate the ground truth, expand both to 1-second segments,
    then score. `ann_df` is the split's annotations.csv loaded as a DataFrame.
    """
    split_names = {os.path.basename(f).replace(".npz", ".wav") for f in file_list}
    ann_split = ann_df[ann_df["filename"].isin(split_names)].copy()

    gt = aggregate_ground_truth_annotations(ann_split)
    gt_seg   = build_segment_frame_from_intervals(gt,      name="ground_truth")
    pred_seg = build_segment_frame_from_intervals(pred_df, name="predictions")
    if len(pred_seg) > 0:
        keep = pred_seg.index.get_level_values("filename").isin(split_names)
        pred_seg = pred_seg[keep]
    return calculate_f1_score(gt_seg, pred_seg)


# --------------------------------------------------------------------------- #
# 4. Per-class threshold tuning  ---  the key new step for Macro F1
# --------------------------------------------------------------------------- #
def collect_whole_second(file_list: List[str], score_fn: ScoreFn,
                         overlap_thresh: float = 0.0, vote: str = "majority"
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Stack whole-second (scores, labels) across files for fast thresholding."""
    from challenge_data import aggregate_labels
    S, Y = [], []
    for fp in file_list:
        data = dict(np.load(fp, allow_pickle=True))
        start_all = data["start_time"]
        whole = np.isclose(start_all % 1.0, 0.0)
        X_all = build_feature_matrix(data)
        S.append(np.asarray(score_fn(X_all), np.float32)[whole])
        Y.append(aggregate_labels(data["annotations"], overlap_thresh, vote)[whole])
    return np.vstack(S), np.vstack(Y)


def tune_thresholds(scores: np.ndarray, labels: np.ndarray,
                    grid: np.ndarray = None) -> np.ndarray:
    """Pick the per-class threshold maximising that class's segment F1.

    This is directly aligned with the metric: the challenge F1 is segment-based,
    so tuning each class's threshold on validation segments optimises the thing
    we are graded on. Returns a (15,) threshold vector.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    n_classes = labels.shape[1]
    best = np.full(n_classes, 0.5, dtype=np.float32)
    for c in range(n_classes):
        if labels[:, c].sum() == 0:        # class absent in val -> keep default
            continue
        f1s = [f1_score(labels[:, c], (scores[:, c] >= t).astype(int),
                        zero_division=0.0) for t in grid]
        best[c] = float(grid[int(np.argmax(f1s))])
    return best


# --------------------------------------------------------------------------- #
# 5. score_fn factory for any sklearn multi-output classifier (RF, XGB, DT)
# --------------------------------------------------------------------------- #
def make_sklearn_score_fn(model) -> ScoreFn:
    """Wrap a fitted multi-output sklearn model into a score_fn.

    predict_proba returns a list of (N, n_classes_c) arrays — one per output.
    We take the P(class=1) column for each, guarding classes the model only
    ever saw as negative (single-column proba -> score 0).
    """
    def score_fn(X_all: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X_all)
        cols = []
        for p in proba:
            cols.append(p[:, 1] if p.shape[1] == 2 else np.zeros(p.shape[0], np.float32))
        return np.stack(cols, axis=1).astype(np.float32)
    return score_fn


# --------------------------------------------------------------------------- #
# 6. One-call F1 evaluation for a fitted model (tune on local-val, score split)
# --------------------------------------------------------------------------- #
def evaluate_model(model, files: Dict[str, List[str]], ann_df: pd.DataFrame,
                   label_cfg: Dict, eval_split: str = "local_val",
                   grid: np.ndarray = None) -> Dict:
    """Wrap fitted model -> tune per-class thresholds on local-val -> Macro F1.

    Thresholds are ALWAYS tuned on local_val (never the eval split), so this is
    safe to call with eval_split='non_hidden' for the final estimate too.
    """
    score_fn = make_sklearn_score_fn(model)
    S, Y = collect_whole_second(files["local_val"], score_fn, **label_cfg)
    thr = tune_thresholds(S, Y, grid)
    pred = generate_predictions(files[eval_split], score_fn, thr)
    f1, per_class = evaluate_sed(pred, files[eval_split], ann_df)
    return {"thresholds": thr, "macro_f1": f1, "per_class": per_class,
            "score_fn": score_fn}