from __future__ import annotations

import argparse
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

REQUIRED_COLUMNS = {"filename", "annotation", "onset", "offset"}
CLASS_NAMES = {'toilet_flushing', 'coffee_machine', 'running_water', 'cutlery_dishes', 'window_open_close', 'wardrobe_drawer_open_close', 'keychain', 'keyboard_typing', 'footsteps', 'vacuum_cleaner', 'bell_ringing', 'light_switch', 'door_open_close', 'microwave', 'phone_ringing'}

SEGMENT_SECONDS = 1.0


@dataclass(frozen=True)
class ClassEvaluation:
    annotation: str
    precision: float
    recall: float
    f1: float
    map: float

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate event predictions against ground-truth annotations with 0.5 s segments."
    )
    parser.add_argument("ground_truth_csv", type=Path, help="Path to the ground-truth CSV file.")
    parser.add_argument("prediction_csv", type=Path, help="Path to the prediction CSV file.")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Optional directory containing audio files used to validate prediction durations.",
    )
    args = parser.parse_args()

    macro_f1, results = evaluate_prediction_csvs(
        args.ground_truth_csv,
        args.prediction_csv,
        audio_dir=args.audio_dir,
    )

    if results.empty:
        print("No classes found in the provided files.")
    else:
        print(results.to_csv(index=False).strip())
    print(f"macro_f1,{macro_f1:.6f}")


def evaluate_prediction_csvs(
    ground_truth_csv: str | Path,
    prediction_csv: str | Path,
    *,
    audio_dir: str | Path | None = None,
) -> tuple[float, pd.DataFrame]:
    """Evaluate predictions against ground truth CSVs, optionally validating audio bounds."""

    print("Loading ground truth... ")
    ground_truth = load_annotation_csv(ground_truth_csv, ground_truth=True)

    print("Aggregating ground truth annotations in CSV via majority vote...")
    ground_truth = aggregate_ground_truth_annotations(ground_truth)

    print("Loading predictions... ")
    predictions = load_annotation_csv(prediction_csv, ground_truth=False, audio_dir=audio_dir)

    ground_truth_segments = build_segment_frame_from_intervals(ground_truth, name="ground_truth")
    prediction_segments = build_segment_frame_from_intervals(predictions, name="predictions")

    return calculate_f1_score(ground_truth_segments, prediction_segments)


def load_annotation_csv(path: str | Path, ground_truth=False, audio_dir=None) -> pd.DataFrame:
    """Load an annotation CSV and validate required columns, timestamps, and optional audio bounds."""
    csv_path = Path(path).expanduser().resolve()
    df = pd.read_csv(csv_path)
    name = str(csv_path)
    # check columns

    if ground_truth:
        req = REQUIRED_COLUMNS.union({"annotator_id"})
    else:
        req = REQUIRED_COLUMNS

    missing = req.difference(df.columns)

    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_list}")

    if (df["offset"] < df["onset"]).any():
        invalid_rows = df.loc[df["offset"] < df["onset"], ["filename", "annotation", "onset", "offset"]]
        raise ValueError(f"{name} contains rows where offset is smaller than onset:\n{invalid_rows}")

    prediction_classes = set(df["annotation"].dropna().unique().tolist())
    unexpected_classes = sorted(prediction_classes.difference(CLASS_NAMES))
    if unexpected_classes:
        unexpected_classes_list = ", ".join(unexpected_classes)
        raise ValueError(
            "predictions contains classes that do not appear in ground_truth: "
            f"{unexpected_classes_list}"
        )

    if audio_dir:
        audio_durations = get_audio_durations(audio_dir=audio_dir)
        for i, row in df.iterrows():
            if row['onset'] > audio_durations[row['filename']] or row['offset'] > audio_durations[row['filename']]:
                raise ValueError(
                    "prediction is outside of waveform support: "
                    f"{row['filename']}, {row['annotation']}, {row['onset']}, {row['offset']}"
                )
            if row['onset'] < 0 or row['offset'] < 0:
                raise ValueError(
                    "onset and offset must be positive"
                    f"{row['filename']}, {row['annotation']}, {row['onset']}, {row['offset']}"
                )

    return df

def calculate_f1_score(
    ground_truth_segments: pd.DataFrame,
    prediction_segments: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Return macro F1 and a per-class precision/recall/F1 table from segment labels."""
    combined_index = ground_truth_segments.index.union(prediction_segments.index)
    ground_truth_segments = ground_truth_segments.reindex(combined_index, fill_value=0)
    prediction_segments = prediction_segments.reindex(combined_index, fill_value=0)

    classes = sorted(
        set(ground_truth_segments.columns.tolist()).union(prediction_segments.columns.tolist())
    )

    if not classes:
        results = pd.DataFrame(columns=["annotation", "precision", "recall", "f1", "map"])
        return float("nan"), results

    per_class_results: list[ClassEvaluation] = []
    for annotation in classes:
        gt_values = ground_truth_segments.get(annotation, pd.Series(0, index=combined_index, dtype=int))
        pred_values = prediction_segments.get(annotation, pd.Series(0, index=combined_index, dtype=int))
        precision = float(precision_score(gt_values, pred_values, zero_division=0.0))
        recall = float(recall_score(gt_values, pred_values, zero_division=0.0))
        f1 = float(f1_score(gt_values, pred_values, zero_division=0.0))
        per_class_results.append(ClassEvaluation(annotation=annotation, precision=precision, recall=recall, f1=f1, map=None))

    results = pd.DataFrame([vars(result) for result in per_class_results]).sort_values(
        "annotation"
    ).reset_index(drop=True)
    macro_f1 = float(results["f1"].mean())

    return macro_f1, results


def calculate_map_score(
        ground_truth_segments: pd.DataFrame,
        prediction_segments: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Return macro average precision and a per-class metric table from segment labels."""
    combined_index = ground_truth_segments.index.union(prediction_segments.index)
    ground_truth_segments = ground_truth_segments.reindex(combined_index, fill_value=0)
    prediction_segments = prediction_segments.reindex(combined_index, fill_value=0)

    classes = sorted(
        set(ground_truth_segments.columns.tolist()).union(prediction_segments.columns.tolist())
    )

    if not classes:
        results = pd.DataFrame(columns=["annotation", "precision", "recall", "f1", "map"])
        return float("nan"), results

    per_class_results: list[ClassEvaluation] = []
    for annotation in classes:
        gt_values = ground_truth_segments.get(annotation, pd.Series(0, index=combined_index, dtype=int))
        pred_values = prediction_segments.get(annotation, pd.Series(0, index=combined_index, dtype=int))

        if int(gt_values.sum()) == 0:
            map = 0.0
        else:
            map = float(average_precision_score(gt_values, pred_values))
        pred_values = pred_values > 0.5

        precision = float(precision_score(gt_values, pred_values, zero_division=0.0))
        recall = float(recall_score(gt_values, pred_values, zero_division=0.0))
        f1 = float(f1_score(gt_values, pred_values, zero_division=0.0))

        per_class_results.append(ClassEvaluation(annotation=annotation, precision=precision, recall=recall, f1=f1, map=map))

    results = pd.DataFrame([vars(result) for result in per_class_results]).sort_values(
        "annotation"
    ).reset_index(drop=True)
    macro_map = float(results["map"].mean())
    return macro_map, results

def build_segment_frame_from_intervals(
    df: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Expand interval annotations into multi-hot segment frame."""

    def _iter_segments(onset: float, offset: float) -> Iterable[float]:
        if offset <= onset:
            return []

        segment_count = int(round((offset - onset) / SEGMENT_SECONDS))
        return [onset + (index * SEGMENT_SECONDS) for index in range(segment_count)]

    rows: list[dict[str, object]] = []
    for record in df.itertuples(index=False):
        # round to half seconds
        onset = math.floor(float(record.onset) * (1/SEGMENT_SECONDS)) / (1/SEGMENT_SECONDS)
        offset = math.ceil(float(record.offset) * (1/SEGMENT_SECONDS)) / (1/SEGMENT_SECONDS)

        for segment_start in _iter_segments(onset, offset):
            rows.append(
                {
                    "filename": record.filename,
                    "segment_start": segment_start,
                    "annotation": record.annotation,
                    "value": 1,
                }
            )

    if not rows:
        empty_index = pd.MultiIndex.from_tuples([], names=["filename", "segment_start"])
        return pd.DataFrame(index=empty_index)

    segment_df = pd.DataFrame(rows)
    pivoted = segment_df.pivot_table(
        index=["filename", "segment_start"],
        columns="annotation",
        values="value",
        aggfunc="max",
        fill_value=0,
    )
    pivoted = pivoted.astype(int)
    pivoted.columns.name = None
    return pivoted.sort_index()


def aggregate_ground_truth_annotations(df: pd.DataFrame, *, file_col: str = "filename") -> pd.DataFrame:
    """Collapse annotator intervals into majority-vote intervals per file and class."""

    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda interval: interval[0])
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            previous_start, previous_end = merged[-1]
            if current_start <= previous_end:
                merged[-1] = (previous_start, max(previous_end, current_end))
                continue
            merged.append((current_start, current_end))

        return merged

    def _aggregate_majority_intervals(
            intervals: list[tuple[float, float]],
            *,
            threshold: int,
    ) -> list[tuple[float, float]]:
        if not intervals:
            return []

        deltas_by_time: dict[float, int] = {}
        for onset, offset in intervals:
            deltas_by_time[onset] = deltas_by_time.get(onset, 0) + 1
            deltas_by_time[offset] = deltas_by_time.get(offset, 0) - 1

        active = 0
        start: float | None = None
        merged: list[tuple[float, float]] = []

        for time in sorted(deltas_by_time):
            previous_active = active
            active += deltas_by_time[time]

            if previous_active < threshold and active >= threshold:
                start = time
            elif previous_active >= threshold and active < threshold and start is not None:
                merged.append((start, time))
                start = None

        return merged


    if "annotator_id" not in df.columns:
        return df.copy()

    merged_rows: list[dict[str, object]] = []

    for file_id, file_df in df.groupby(file_col, sort=False):
        annotator_count = file_df["annotator_id"].dropna().nunique()
        if annotator_count == 0:
            continue
        majority_threshold = math.ceil(annotator_count / 2)

        for annotation, annotation_df in file_df.groupby("annotation", sort=False):
            merged_intervals_per_annotator: list[tuple[float, float]] = []

            for _, annotator_df in annotation_df.groupby("annotator_id", sort=False):
                intervals = list(zip(annotator_df["onset"], annotator_df["offset"]))
                merged_intervals_per_annotator.extend(_merge_intervals(intervals))

            majority_intervals = _aggregate_majority_intervals(
                merged_intervals_per_annotator,
                threshold=majority_threshold,
            )
            for onset, offset in majority_intervals:
                merged_rows.append(
                    {
                        file_col: file_id,
                        "annotation": annotation,
                        "onset": onset,
                        "offset": offset,
                    }
                )

    return pd.DataFrame(merged_rows, columns=["filename", "annotation", "onset", "offset"])


def get_audio_durations(audio_dir: str | Path) -> dict[str, float]:
    """Return basename-keyed WAV durations from metadata, rounded up to 0.5 s."""

    audio_root = Path(audio_dir).expanduser().resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_root}")

    durations: dict[str, float] = {}
    for audio_path in sorted(path for path in audio_root.rglob("*") if path.is_file()):
        if audio_path.name in durations:
            raise ValueError(
                f"Duplicate audio filename found under {audio_root}: {audio_path.name}. "
                "CSV evaluation matches audio files by basename."
            )
        if audio_path.suffix.lower() != ".wav":
            continue

        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()

        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate for audio file: {audio_path}")
        duration = frame_count / sample_rate
        durations[audio_path.name] = math.ceil(float(duration) * 2) / 2

    return durations

if __name__ == "__main__":
    main()
