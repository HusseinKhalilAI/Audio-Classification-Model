# Polyphonic Sound Event Detection in Domestic Environments

Detection of 15 overlapping household sound events in continuous recordings, scored by segment-based macro F1 on a 1-second grid.

The final system reaches **0.730 macro F1**, a **2.3x improvement** over a decision-tree baseline (0.317). The progression is deliberately incremental: every model in this repo plugs into one shared inference backbone, so each result differs from the previous one by exactly one change.

---

## Results

All numbers are macro F1 on a held-out split that is never used for threshold tuning or model selection.

| System | Macro F1 |
|---|---|
| Decision tree on precomputed features (baseline) | 0.317 |
| Random forest, 400 trees, balanced class weights | 0.483 |
| XGBoost, depth 8, lr 0.1 | 0.566 |
| XGBoost + mean score smoothing W=3, thresholds re-tuned | 0.602 |
| LSTM over mel envelopes, trained from scratch | 0.481 |
| CRNN over log-mel patches, trained from scratch | 0.563 |
| Frozen `frame_mn10` AudioSet embeddings + MLP head | 0.683 |
| Deep MLP head + class-balanced loss | 0.680 |
| BiGRU head over embedding sequences | 0.712 |
| BiGRU → deep MLP head | 0.726 |
| **Ensemble: BiGRU → deep MLP (0.6) + XGBoost (0.4)** | **0.730** |

Per-class F1, baseline versus final system:

| Class | Baseline | Final |
|---|---|---|
| `keyboard_typing` | 0.36 | 0.91 |
| `running_water` | 0.58 | 0.88 |
| `vacuum_cleaner` | 0.47 | 0.88 |
| `toilet_flushing` | 0.29 | 0.86 |
| `phone_ringing` | 0.49 | 0.81 |
| `microwave` | 0.39 | 0.80 |
| `cutlery_dishes` | 0.29 | 0.77 |
| `coffee_machine` | 0.36 | 0.75 |
| `keychain` | 0.32 | 0.74 |
| `footsteps` | 0.33 | 0.73 |
| `door_open_close` | 0.18 | 0.65 |
| `light_switch` | 0.23 | 0.61 |
| `bell_ringing` | 0.29 | 0.60 |
| `wardrobe_drawer_open_close` | 0.12 | 0.53 |
| `window_open_close` | 0.06 | 0.44 |

The four worst baseline classes are all short transients. They are where nearly all of the headroom was, and where the temporal model paid off.

---

## Data

**The dataset is not redistributable and is not included here.** This section documents its exact structure so the pipeline can be pointed at an equivalent corpus.

Two roots are used, set through environment variables:

```
SED_DATA_ROOT   precomputed features + annotations
SED_RAW_ROOT    raw waveforms (needed only by the deep models)
```

### Layout

```
$SED_DATA_ROOT/
  train/       audio_features/*.npz      3,704 recordings
  validation/  audio_features/*.npz        999 recordings   + annotations.csv
  test/        audio_features/*.npz      1,007 recordings   (labels withheld)

$SED_RAW_ROOT/
  train/audio/*.wav
  validation/audio/*.wav
  test/audio/*.wav
```

### Form 1 — precomputed feature `.npz`

One archive per recording. Each holds a set of keyed arrays on a segment grid of **1.0 s windows at 0.5 s hop**:

| Key | Shape | Meaning |
|---|---|---|
| `start_time` | `(N_seg,)` | segment onset in seconds |
| `annotations` | `(N_seg, 15, A)` | per-annotator overlap fraction in [0, 1] |
| 13 feature groups | see below | acoustic descriptors per segment |

The feature groups concatenate to a **960-dimensional** vector per segment:

| Group | Dims | Statistics |
|---|---|---|
| MFCC | 32 | mean, std, min, max |
| MFCC delta | 32 | mean, std, min, max |
| MFCC delta² | 32 | mean, std, min, max |
| Mel spectrogram | 128 | mean, std, min, max |
| Spectral contrast | 7 | mean, std, min, max |
| Zero crossing rate | 1 | mean, std, min, max |
| Spectral flux | 1 | mean, std, min, max |
| Spectral flatness | 1 | mean, std, min, max |
| Spectral centroid | 1 | mean, std, min, max |
| Spectral bandwidth | 1 | mean, std, min, max |
| Rolloff low | 1 | mean, std, min, max |
| Rolloff high | 1 | mean, std, min, max |
| Energy / Power | 1 each | mean, std, min, max |

`384 + 512 + 28 + 36 = 960`. These dimensions are heavily inter-correlated, which is the root cause of the baseline's precision problem.

### Form 2 — annotations

Every recording is labelled independently by several annotators, with no agreement enforced on boundaries. Two representations exist and they must be aggregated identically or local scores drift from the official ones:

- **Dense**, inside each `.npz`: an `(N_seg, 15, A)` overlap-fraction tensor. Reduced to binary `(N_seg, 15)` by thresholding at any overlap, then **majority vote** across annotators (`votes >= ceil(A/2)`).
- **Interval**, in `validation/annotations.csv`: rows of `filename, annotator_id, annotation, onset, offset`. Aggregated by sweeping interval start/end deltas and keeping the spans where at least `ceil(A/2)` annotators are simultaneously active.

`aggregate_labels` in `data.py` implements the first and mirrors the second exactly. Getting this wrong is the single easiest way to produce a validation number that does not transfer.

### Form 3 — raw waveforms

Consumed only by the from-scratch and pretrained-embedding models, at different rates:

| Consumer | Sample rate | Transform |
|---|---|---|
| CRNN | 32 kHz | log-mel, `n_fft` 1024, hop 320, 128 mels → `(128, 100)` patch per second |
| LSTM | 32 kHz | same patches, averaged over time → 128-dim envelope per second |
| Pretrained backbone | 16 kHz | 10-second chunks → 250 frames at 40 ms |

Both paths cache their intermediate arrays to disk on the first pass, so training epochs do no audio decoding.

### Form 4 — cached embeddings

Frozen `frame_mn10` (MobileNetV3, pretrained on AudioSet-Strong) output, mean-pooled from 25 frames/second down to the 1-second grid:

```
emb_frame_mn10/<split>/<id>.npz  →  feats (N_seg, 960) fp16, labels (N_seg, 15), start (N_seg,)
```

Training set: 170,508 segments. Local validation: 23,404 segments.

### Splits

Train and validation folders are used as provided. Test labels are withheld, so **validation is split 500/499 with a fixed seed** into:

- `local_val` — threshold tuning, hyperparameter selection, early stopping
- `non_hidden` — the honest final estimate, touched only for reporting

Every number in the results table is `non_hidden`. Nothing is ever tuned on it.

### Evaluation

Predictions are emitted as `filename, annotation, onset, offset` intervals. Both ground truth and predictions are expanded onto a 1-second grid, per-class F1 is computed over segments, and the 15 class scores are averaged unweighted. Unweighted averaging is why rare, short classes dominate the metric despite contributing almost no audio.

---

## What I did, step by step

### 1 — Built the inference backbone first

Before any model, `sed_pipeline.py` defines the contract every system implements:

```
score_fn: (N_seg, 960) → (N_seg, 15) scores in [0,1]
```

and the fixed path from there: keep whole-second segments → apply per-class thresholds → merge consecutive active seconds into intervals → score with the official evaluator. Random forests, gradient boosting, CRNNs, and sequence heads all reduce to one `score_fn`, which is why later comparisons are apples to apples and why post-processing and error analysis were written once rather than per model.

The local scorer imports the official evaluation functions directly instead of reimplementing them, so the local number equals the leaderboard number by construction.

### 2 — Baseline and error analysis

A per-class decision tree on the 960-dim features: 0.317. The point was diagnosis. Four failure modes:

1. **Duration bias.** Long, steady sounds score well under 1-second aggregation because one correct frame carries the whole segment. Transients have no such margin — `window_open_close` sits at 0.06.
2. **No temporal context.** Frames are classified independently, so predictions flicker within a single continuous event.
3. **No co-occurrence modelling.** Fifteen independent binary classifiers discard the fact that dishes and running water co-occur and vacuum and light switch do not.
4. **Axis-aligned splits on correlated features.** Single-feature thresholds carve poor boundaries across 960 mutually correlated dimensions, and precision suffers.

`error_analysis.py` renders log-mel spectrogram, ground truth, and predictions on a shared time axis so misses and false alarms are readable per class per second, and ranks files by error profile to pick informative examples automatically.

### 3 — Per-class threshold tuning

Because the metric is macro F1 over segments, the operating point is a free parameter that matters more than most hyperparameters. Each class gets its own threshold, chosen on a grid over `local_val` to maximise that class's segment F1. A single global threshold is indefensible when class prevalence spans orders of magnitude. Thresholds are always tuned on `local_val`, never on the evaluation split.

### 4 — Exhausting the classical pipeline

Random forest, swept over `n_estimators` then `max_depth`, with balanced class weights: **0.483**. Depth-limiting hurt monotonically; unrestricted trees won.

XGBoost, one binary booster per class, swept over `max_depth` then `learning_rate`: **0.566**. No class weighting — threshold tuning sets the operating point instead, which is what the metric rewards.

Training was capped at 50,000 segments; full XGBoost sweeps already cost ~10 minutes per configuration.

### 5 — Temporal post-processing, done twice

The first attempt applied a **median filter to binary predictions**. It made things worse at every window (0.575 → 0.565 → 0.534 → 0.507 → 0.480 for W = 1, 3, 5, 7, 9). The reason is that thresholds had been tuned on unfiltered predictions; filtering then removes detections the threshold was calibrated to produce, so the comparison was never fair.

The correct version smooths the **score sequence** with a moving average and **re-tunes thresholds per window**, so each window is evaluated at its own optimal operating point. Under that comparison mean smoothing at W=3 helps, lifting XGBoost from 0.566 to **0.602**. Smoothing is applied per recording so it never crosses file boundaries.

That gap — same idea, opposite conclusion depending on whether the operating point moves with it — is the most useful thing this stage produced.

### 6 — Deep models from scratch, and why they lost

A CRNN over per-second log-mel patches (conv stack → BiGRU) reached **0.563**; an LSTM over 128-dim mel envelopes reached **0.481**. Both underperformed tuned XGBoost on the same grid, and both plateaued while still improving slowly at 20 epochs.

The conclusion was not that recurrence is useless — step 8 shows it is decisive. It was that ~3,700 recordings are not enough to learn general acoustic representations from raw audio, so the capacity is spent relearning what a mel spectrogram already encodes.

### 7 — Replacing the representation

Swapped the input to frozen `frame_mn10` embeddings, pretrained on AudioSet-Strong, and kept the classifier deliberately trivial — a one-hidden-layer MLP. **0.683**, beating the fully tuned gradient-boosting pipeline and every from-scratch network, with a fraction of the engineering.

The backbone is never fine-tuned. Embeddings are extracted once and cached, which drops each subsequent experiment to seconds and is what made the next stage practical at all.

### 8 — Sequence heads on frozen embeddings

Three heads on identical cached inputs, so the comparison isolates architecture:

| Head | Macro F1 |
|---|---|
| Deep MLP, class-balanced loss, AdamW + cosine | 0.680 |
| BiGRU, 2 layers, 256 hidden | 0.712 |
| BiGRU → deep MLP | 0.726 |

Adding depth and class balancing to the frame-independent head bought nothing. Adding temporal context bought 0.03 immediately. Events defined by an envelope — a drawer sliding, a sash moving, a switch clicking — cannot be represented by a model that sees one second in isolation, no matter how good that second's features are.

Class-balanced BCE (`pos_weight` = neg/pos, capped at 20) is applied throughout to keep the rare classes from being ignored by a loss that macro F1 will punish.

### 9 — Ensembling

The sequence head and XGBoost fail differently: one has temporal context on learned features, the other has fine-grained hand-engineered spectral statistics with no memory. Weighted score fusion at **0.6 / 0.4**, with thresholds re-tuned on the fused scores, gives the final **0.730**.

Final predictions are validated against the evaluator's own input rules — required columns, ordered onsets, known class names, in-bounds timestamps — before writing, so a malformed submission fails locally rather than silently scoring zero.

---

## Repository structure

```
data.py                  paths, feature assembly, label aggregation, splits
sed_pipeline.py          score_fn contract, intervals, thresholds, official scoring
evaluate.py              reference evaluator (segment-based macro F1)

random_forest.py         RF training + hyperparameter sweep
xgboost_model.py         XGBoost training + hyperparameter sweep
run_classical.py         retrain both, lock the winner, save artifacts

post_processing.py       median filtering and score smoothing
run_smoothing.py         fair smoothing sweep with threshold re-tuning

crnn_lstm.py             from-scratch CRNN and LSTM on cached log-mel patches
pretrained_embeddings.py frozen frame_mn10 extraction, caching, MLP head
sequence_heads.py        deep MLP / BiGRU / BiGRU→MLP heads, ensembling

error_analysis.py        spectrogram vs GT vs prediction plots
run_error_analysis.py    pick and render informative failure cases
collect_results.py       scrape every artifact into result tables
plot_per_class.py        per-class heatmap and grouped bars
make_submission.py       hidden-set inference + format validation
```

Everything is modular `.py`. Notebooks were used for exploration only, never as the pipeline.

## Running it

```bash
conda create -n sed python=3.12
conda activate sed
pip install -r requirements.txt

export SED_DATA_ROOT=/path/to/features
export SED_RAW_ROOT=/path/to/audio

python random_forest.py          # RF sweep
python xgboost_model.py          # XGB sweep
python run_classical.py          # lock the best classical system
python run_smoothing.py          # score smoothing + threshold re-tuning
python pretrained_embeddings.py  # extract and cache embeddings, train MLP head
python sequence_heads.py         # BiGRU heads + ensemble, writes submission.csv
python collect_results.py        # regenerate all result tables
```

Add `--smoke` to the deep-model scripts for a fast end-to-end shape check on a small subset.

---

## Findings

**The bottleneck was the representation, not the classifier.** Weeks of classifier work moved the score from 0.317 to 0.602. Changing the input representation cleared that in one step with a simpler model. Diagnosing which side of the pipeline is limiting you is worth more than any amount of tuning on the wrong side.

**Post-processing must be compared at a re-tuned operating point.** The same smoothing idea looked harmful under a fixed threshold and helpful once thresholds moved with it. Any post-processing that shifts the score distribution invalidates the threshold it was tuned against.

**Temporal structure is not optional for transient events.** Depth and class balancing on a frame-independent head bought nothing; two BiGRU layers on identical inputs bought 0.03. The classes that were unusable at baseline are exactly the ones defined by a short envelope.

**Pretrained features beat from-scratch networks at this data scale.** Trained on the same 3,700 recordings, the CRNN lost to gradient boosting. The same recurrent idea placed on top of AudioSet embeddings won by a wide margin.

**Unweighted macro averaging changes what to optimise.** A light switch click carries the same weight as minutes of running water. Effort spent on classes already above 0.8 is nearly wasted.

## Limitations

- The backbone is frozen. Full fine-tuning has a higher ceiling and was scoped out on compute grounds.
- The 15 outputs remain independent heads; class co-occurrence is still not modelled explicitly.
- Per-class thresholds are tuned on ~23k validation segments, which carries overfitting risk for the rarest classes.
- Onsets and offsets are quantised to whole seconds by construction, so the system cannot express finer boundaries than the metric measures.

## License

`<FILL: MIT / Apache-2.0>`
