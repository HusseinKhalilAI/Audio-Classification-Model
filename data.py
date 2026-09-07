import os
import glob
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Callable

from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
PATH_TO_DATASET = os.environ.get("SED_DATA_ROOT", "data/features")
PATH_TRAIN = os.path.join(PATH_TO_DATASET, "train")
PATH_VAL   = os.path.join(PATH_TO_DATASET, "validation")
PATH_TEST  = os.path.join(PATH_TO_DATASET, "test")

# --------------------------------------------------------------------------- #
# Feature list  ---  kept in YOUR ordering (MFCC family first, then mel, ...)
# 13 feature groups -> 960 columns total.
# --------------------------------------------------------------------------- #
training_features = [
    # MFCC family (12 keys x 32 dims = 384 cols)
    "mfcc_mean",     "mfcc_std",     "mfcc_min",     "mfcc_max",
    "mfcc_d_mean",   "mfcc_d_std",   "mfcc_d_min",   "mfcc_d_max",
    "mfcc_d2_mean",  "mfcc_d2_std",  "mfcc_d2_min",  "mfcc_d2_max",
    # Mel spectrogram (4 keys x 128 dims = 512 cols)
    "melspect_mean", "melspect_std", "melspect_min", "melspect_max",
    # Zero crossing rate (4 cols)
    "zcr_mean",      "zcr_std",      "zcr_min",      "zcr_max",
    # Spectral flux (4 cols)
    "flux_mean",     "flux_std",     "flux_min",     "flux_max",
    # Spectral flatness (4 cols)
    "flatness_mean", "flatness_std", "flatness_min", "flatness_max",
    # Spectral centroid (4 cols)
    "centroid_mean", "centroid_std", "centroid_min", "centroid_max",
    # Spectral bandwidth (4 cols)
    "bandwidth_mean","bandwidth_std","bandwidth_min","bandwidth_max",
    # Spectral contrast (4 keys x 7 dims = 28 cols)
    "contrast_mean", "contrast_std", "contrast_min", "contrast_max",
    # Spectral rolloff low (4 cols)
    "rolloff_low_mean",  "rolloff_low_std",  "rolloff_low_min",  "rolloff_low_max",
    # Spectral rolloff high (4 cols)
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    # Energy (4 cols)
    "energy_mean",   "energy_std",   "energy_min",   "energy_max",
    # Power (4 cols)
    "power_mean",    "power_std",    "power_min",    "power_max",
]

# 15 target classes, alphabetical -> matches the .npz `class_names` order and
# the second axis of the `annotations` array.
CLASS_NAMES = [
    "bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
    "footsteps", "keyboard_typing", "keychain", "light_switch", "microwave",
    "phone_ringing", "running_water", "toilet_flushing", "vacuum_cleaner",
    "wardrobe_drawer_open_close", "window_open_close",
]

SEGMENT_LENGTH = 1.0
HOP_SIZE       = 0.5


# --------------------------------------------------------------------------- #
# Labels  ---  ALIGNED TO THE EVALUATOR
# --------------------------------------------------------------------------- #
def aggregate_labels(
    annotations: np.ndarray,
    overlap_thresh: float = 0.0,
    vote: str = "majority",
) -> np.ndarray:
    """Reduce a (T, C, A) annotation array to binary (T, C) segment labels.

    Parameters
    ----------
    annotations : (T, C, A) float array
        Per-segment, per-class, per-annotator overlap fraction in [0, 1].
    overlap_thresh : float, default 0.0
        A segment "contains" the class for an annotator if overlap > thresh.
        0.0 (any overlap) mirrors the evaluator. (Set e.g. 0.5 for an ablation.)
    vote : {"majority", "any", "all"}, default "majority"
        How to combine annotators. "majority" -> votes >= ceil(A/2),
        which is exactly what evaluate.py's ground-truth aggregation uses.
        "any" = OR (your previous project's rule), "all" = AND.

    Returns
    -------
    (T, C) int array
    """
    if annotations.ndim != 3:
        raise ValueError(f"Expected 3D (T, C, A), got {annotations.shape}")

    binary = annotations > overlap_thresh           # (T, C, A)
    votes  = binary.sum(axis=-1)                     # (T, C)
    A = binary.shape[-1]

    if vote == "any":
        labels = votes >= 1
    elif vote == "all":
        labels = votes >= A
    elif vote == "majority":
        labels = votes >= math.ceil(A / 2)           # matches evaluate.py
    else:
        raise ValueError(f"unknown vote rule: {vote}")
    return labels.astype(np.int8)


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_feature_matrix(data: dict) -> np.ndarray:
    """Concatenate all `training_features` from a loaded .npz into (N, 960)."""
    arrays = []
    for k in training_features:
        feat = data[k]
        if feat.ndim == 1:
            feat = feat[:, np.newaxis]
        arrays.append(feat.astype(np.float32))
    return np.concatenate(arrays, axis=1)


def construct_Xy(
    file_list: List[str],
    overlap_thresh: float = 0.0,
    vote: str = "majority",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack ALL segments from all files into flat (N, 960) X and (N, 15) y.

    For training the temporal order is irrelevant, so we simply vstack.
    Use this for RF / XGBoost.
    """
    X_list, y_list = [], []
    for i, fp in enumerate(file_list):
        d = dict(np.load(fp, allow_pickle=True))
        X_list.append(build_feature_matrix(d))
        y_list.append(aggregate_labels(d["annotations"], overlap_thresh, vote))
        if verbose and (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{len(file_list)}")
    return np.vstack(X_list), np.vstack(y_list)


def build_sequences(
    file_list: List[str],
    scaler_mean: np.ndarray = None,
    scaler_scale: np.ndarray = None,
    overlap_thresh: float = 0.0,
    vote: str = "majority",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One (T_i, 960) feature array and (T_i, 15) label array per file.

    Used for the sequence models (LSTM / CRNN). If scaler stats are given,
    features are standardised with the TRAIN scaler.
    """
    X_seqs, y_seqs = [], []
    for fp in file_list:
        d = dict(np.load(fp, allow_pickle=True))
        feats = build_feature_matrix(d)
        if scaler_mean is not None:
            feats = (feats - scaler_mean) / scaler_scale
        labels = aggregate_labels(d["annotations"], overlap_thresh, vote).astype(np.float32)
        X_seqs.append(feats.astype(np.float32))
        y_seqs.append(labels)
    return X_seqs, y_seqs


# --------------------------------------------------------------------------- #
# Splits  ---  use the PROVIDED folders; split validation 500/499 (seed 42)
# --------------------------------------------------------------------------- #
def list_split_files() -> Dict[str, List[str]]:
    """Return .npz paths for train, local-val, non-hidden-test, hidden-test."""
    train_files = sorted(glob.glob(os.path.join(PATH_TRAIN, "audio_features", "*.npz")))
    val_files   = sorted(glob.glob(os.path.join(PATH_VAL,   "audio_features", "*.npz")))
    test_files  = sorted(glob.glob(os.path.join(PATH_TEST,  "audio_features", "*.npz")))

    rng = np.random.default_rng(seed=42)
    val_shuffled = rng.permutation(val_files).tolist()
    n_val = len(val_shuffled) // 2
    our_val_files  = val_shuffled[:n_val]  
    our_test_files = val_shuffled[n_val:]   

    return {
        "train":      train_files,
        "local_val":  our_val_files,
        "non_hidden": our_test_files,
        "hidden":     test_files,
    }


def generate_standard_splits(
    max_train_files: int = None,
    overlap_thresh: float = 0.0,
    vote: str = "majority",
) -> Dict:
    """Build standardised flat splits for the classical models.

    Returns flat X/y for train, local-val and non-hidden-test, the fitted
    StandardScaler, and the underlying file lists (needed for SED evaluation).
    """
    files = list_split_files()
    train_files = files["train"]
    if max_train_files is not None:
        rng = np.random.default_rng(seed=42)
        train_files = rng.choice(train_files,
                                 size=min(max_train_files, len(train_files)),
                                 replace=False).tolist()

    X_train, y_train = construct_Xy(train_files,           overlap_thresh, vote, verbose=True)
    X_val,   y_val   = construct_Xy(files["local_val"],    overlap_thresh, vote)
    X_nht,   y_nht   = construct_Xy(files["non_hidden"],   overlap_thresh, vote)

    scaler = StandardScaler().fit(X_train)
    return {
        "train":      (scaler.transform(X_train), y_train),
        "local_val":  (scaler.transform(X_val),   y_val),
        "non_hidden": (scaler.transform(X_nht),   y_nht),
        "scaler":     scaler,
        "files":      files,
        "label_cfg":  {"overlap_thresh": overlap_thresh, "vote": vote},
    }