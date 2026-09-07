

import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import f1_score

# --- make the PretrainedSED repo importable (models/, config.py, helpers/, ...) ---
PSED_REPO = os.environ.get("PSED_REPO_PATH", "PretrainedSED"")
sys.path.insert(0, PSED_REPO)

from data import CLASS_NAMES, list_split_files, aggregate_labels, PATH_VAL
from sed_pipeline import predictions_to_intervals, evaluate_sed, tune_thresholds
from post_processing import smooth_scores

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")

PATH_RAW = os.environ.get("SED_RAW_ROOT", "data/raw")
EMB_DIR  = "emb_frame_mn10"           # cached per-second embeddings, per split

MODEL_NAME = "frame_mn10"
SR             = 16_000               # the pretrained models are trained on 16 kHz
CHUNK_SEC      = 10                    # ... on 10-second pieces
CHUNK_SAMPLES  = CHUNK_SEC * SR
FRAMES_PER_CHUNK = 250                # PredictionsWrapper seq_len (40 ms resolution)
FPS_FRAMES     = FRAMES_PER_CHUNK / CHUNK_SEC   # 25 embedding frames / second

LABEL_CFG = {"overlap_thresh": 0.0, "vote": "majority"}


def _jsonable(o):
    if isinstance(o, dict):  return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


# =========================================================================== #
# 1. PRETRAINED BACKBONE  (head_type=None -> returns frame EMBEDDINGS)
# =========================================================================== #
def load_backbone():
    """frame_mn10 pretrained on AudioSet-strong, as a frozen embedding extractor.
    head_type=None makes forward() return (B, 250, embed_dim) sequence embeddings."""
    from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
    from models.frame_mn.utils import NAME_TO_WIDTH
    from models.prediction_wrapper import PredictionsWrapper

    width = NAME_TO_WIDTH(MODEL_NAME)
    frame_mn = FrameMNWrapper(width)
    embed_dim = frame_mn.state_dict()['frame_mn.features.16.1.bias'].shape[0]
    # load the strong checkpoint, then drop the 447-class head: head_type=None
    model = PredictionsWrapper(frame_mn, checkpoint=f"{MODEL_NAME}_strong_1",
                               embed_dim=embed_dim, head_type=None)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    print(f"loaded {MODEL_NAME} (embed_dim={embed_dim}), frozen")
    return model, embed_dim


# =========================================================================== #
# 2. FEATURE EXTRACTION  ->  per-second embeddings cached to disk (once)
# =========================================================================== #
def _feature_npz_path(filename, split):
    from challenge_data import PATH_TRAIN, PATH_VAL as _PV, PATH_TEST
    root = {"train": PATH_TRAIN, "validation": _PV, "test": PATH_TEST}[split]
    return os.path.join(root, "audio_features", os.path.basename(filename))


@torch.no_grad()
def _embed_recording(model, wav):
    """Raw 16 kHz waveform -> (embed_dim, total_frames) by 10 s chunking."""
    wav = torch.from_numpy(wav[None, :]).to(DEVICE)
    n = wav.shape[1]
    n_chunks = n // CHUNK_SAMPLES + (n % CHUNK_SAMPLES != 0)
    outs = []
    for i in range(n_chunks):
        chunk = wav[:, i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        if chunk.shape[1] < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[1]))
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            mel = model.mel_forward(chunk)
            emb = model(mel)                       # (1, 250, embed_dim)
        outs.append(emb.float().cpu())
    emb = torch.cat(outs, dim=1)[0].numpy()        # (total_frames, embed_dim)
    return emb.T                                   # (embed_dim, total_frames)


def _pool_to_segments(emb_DT, start_times):
    """Average the 40 ms frames inside each 1-second segment -> (N_seg, embed_dim)."""
    D, T = emb_DT.shape
    out = np.zeros((len(start_times), D), np.float32)
    for i, t in enumerate(start_times):
        f0 = int(round(t * FPS_FRAMES)); f1 = f0 + int(round(FPS_FRAMES))
        seg = emb_DT[:, f0:min(f1, T)]
        out[i] = seg.mean(axis=1) if seg.shape[1] > 0 else 0.0
    return out


def build_embeddings(model, split, file_list, max_files=None):
    """Cache per-second embeddings (N_seg, embed_dim) + labels + start per file.
    Iterates the SAME file list the pipeline uses, so the cache always matches
    later lookups (important in --smoke mode, where only a subset is cached)."""
    out_dir = os.path.join(EMB_DIR, split); os.makedirs(out_dir, exist_ok=True)
    fl = file_list[:max_files] if max_files else file_list
    print(f"[{split}] embedding {len(fl)} files -> {out_dir}")
    for i, fp in enumerate(fl):
        fid = os.path.basename(fp).replace(".npz", "")
        outp = os.path.join(out_dir, fid + ".npz")
        if os.path.exists(outp):
            continue
        wav_path = os.path.join(PATH_RAW, split, "audio", fid + ".wav")
        wav, _ = librosa.load(wav_path, sr=SR, mono=True)
        emb = _embed_recording(model, wav)
        fz = dict(np.load(_feature_npz_path(fid + ".npz", split), allow_pickle=True))
        start = fz["start_time"]
        feats = _pool_to_segments(emb, start)
        labels = (aggregate_labels(fz["annotations"], **LABEL_CFG)
                  if "annotations" in fz else np.zeros((len(start), 15), np.int8))
        np.savez(outp, feats=feats.astype(np.float16),
                 labels=labels.astype(np.int8), start=start.astype(np.float32))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(fl)}")
    return out_dir


def load_emb(filename, split):
    d = np.load(os.path.join(EMB_DIR, split, os.path.basename(filename).replace(".npz", ".npz")))
    return d["feats"].astype(np.float32), d["labels"], d["start"]


# =========================================================================== #
# 3. HEAD  +  training (flat per-segment; frozen features => very fast)
# =========================================================================== #
class Head(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=15, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


class FlatDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def _stack(files, split):
    X, Y = [], []
    for f in files:
        feats, lab, _ = load_emb(f, split)
        X.append(feats); Y.append(lab)
    return np.vstack(X), np.vstack(Y)


def train_head(Xtr, Ytr, Xvl, Yvl, in_dim, hidden=256, dropout=0.3,
               lr=1e-3, epochs=60, patience=8):
    head = Head(in_dim, hidden, 15, dropout).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tr = DataLoader(FlatDataset(Xtr, Ytr), batch_size=512, shuffle=True)
    Xvl_t = torch.tensor(Xvl, dtype=torch.float32, device=DEVICE)
    best_f1, best_state, bad = -1, None, 0
    for ep in range(epochs):
        head.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(head(xb), yb); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            sv = torch.sigmoid(head(Xvl_t)).cpu().numpy()
        vf1 = f1_score(Yvl, (sv >= 0.5).astype(int), average="macro", zero_division=0.0)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
        if ep % max(1, epochs // 10) == 0:
            print(f"  epoch {ep:2d}: val macroF1@0.5 = {vf1:.4f}")
        if bad >= patience:
            print(f"  early stop at epoch {ep}"); break
    head.load_state_dict(best_state)
    print(f"  best val macroF1@0.5 = {best_f1:.4f}")
    return head, best_f1


# =========================================================================== #
# 4. GLUE -> SED backbone
# =========================================================================== #
def _whole(start):
    return np.isclose(start % 1.0, 0.0)


def head_file_scores(head, filename, split):
    feats, _, start = load_emb(filename, split)
    head.eval()
    with torch.no_grad():
        s = torch.sigmoid(head(torch.tensor(feats, device=DEVICE))).cpu().numpy()
    return s, start


def dl_generate_predictions(head, files, split, thresholds, smooth_window=1):
    thr = np.asarray(thresholds, np.float32); rows = []
    for f in files:
        s, start = head_file_scores(head, f, split)
        w = _whole(start); sc = s[w]
        if smooth_window > 1:
            sc = smooth_scores(sc, smooth_window, "mean")
        binary = (sc >= thr).astype(int)
        rows.extend(predictions_to_intervals(binary, start[w],
                    os.path.basename(f).replace(".npz", ".wav")))
    if not rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(rows)


def dl_evaluate(head, files, split, thresholds, ann_df, smooth_window=1):
    pred = dl_generate_predictions(head, files, split, thresholds, smooth_window)
    return evaluate_sed(pred, files, ann_df)


def whole_second_scores(head, files, split):
    S, Y = [], []
    for f in files:
        s, start = head_file_scores(head, f, split)
        w = _whole(start); S.append(s[w])
        _, lab, _ = load_emb(f, split); Y.append(lab[w])
    return np.vstack(S), np.vstack(Y)


# =========================================================================== #
# 5. RUN: embed -> sweep head -> threshold -> smoothing -> non-hidden -> submit
# =========================================================================== #
def run_bonus2(smoke=False):
    files = list_split_files()
    if smoke:
        for k in files:
            files[k] = files[k][:40]
    split_map = {"train": files["train"], "local_val": files["local_val"],
                 "non_hidden": files["non_hidden"], "hidden": files["hidden"]}
    ann_df = pd.read_csv(os.path.join(PATH_VAL, "annotations.csv"))

    model, embed_dim = load_backbone()
    # validation embeddings must cover BOTH local_val and non_hidden subsets
    val_files = list(dict.fromkeys(split_map["local_val"] + split_map["non_hidden"]))
    split_files = {"train": split_map["train"], "validation": val_files,
                   "test": split_map["hidden"]}
    for sp in ("train", "validation", "test"):
        build_embeddings(model, sp, split_files[sp])
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # flat feature matrices from cached embeddings
    Xtr, Ytr = _stack(split_map["train"], "train")
    Xvl, Yvl = _stack(split_map["local_val"], "validation")
    print(f"train {Xtr.shape}  val {Xvl.shape}")

    # sweep head hidden size (cheap: frozen features)
    epochs = 3 if smoke else 60
    grid = [256] if smoke else [128, 256, 512]
    results = []
    best = None
    for h in grid:
        print(f"\n[head] hidden={h}")
        head, _ = train_head(Xtr, Ytr, Xvl, Yvl, embed_dim, hidden=h, epochs=epochs)
        S, Y = whole_second_scores(head, split_map["local_val"], "validation")
        thr = tune_thresholds(S, Y)
        f1, _ = dl_evaluate(head, split_map["local_val"], "validation", thr, ann_df)
        print(f"  local-val Macro F1 = {f1:.4f}")
        results.append({"hidden": h, "f1": float(f1)})
        if best is None or f1 > best["f1"]:
            best = {"hidden": h, "head": head, "thr": thr, "f1": f1}

    # smoothing sweep on the best head
    pp_best = {"window": 1, "thr": best["thr"], "f1": best["f1"]}
    smooth_curve = []
    for w in ((1, 3) if smoke else (1, 3, 5, 7)):
        if w == 1:
            f1w, thrw = best["f1"], best["thr"]
        else:
            Sw, Yw = [], []
            for f in split_map["local_val"]:
                s, start = head_file_scores(best["head"], f, "validation")
                ww = _whole(start)
                Sw.append(smooth_scores(s[ww], w, "mean"))
                Yw.append(load_emb(f, "validation")[1][ww])
            thrw = tune_thresholds(np.vstack(Sw), np.vstack(Yw))
            f1w, _ = dl_evaluate(best["head"], split_map["local_val"], "validation",
                                 thrw, ann_df, smooth_window=w)
        smooth_curve.append({"window": w, "macro_f1": float(f1w)})
        print(f"  smooth W={w}: local-val Macro F1 = {f1w:.4f}")
        if f1w > pp_best["f1"]:
            pp_best = {"window": w, "thr": thrw, "f1": f1w}

    # honest non-hidden number
    f1_nh, per_class = dl_evaluate(best["head"], split_map["non_hidden"], "validation",
                                   pp_best["thr"], ann_df, smooth_window=pp_best["window"])
    print(f"\n=== Bonus2 ({MODEL_NAME}) non-hidden Macro F1 = {f1_nh:.4f} "
          f"(hidden={best['hidden']}, smooth W={pp_best['window']}) ===")
    print(per_class.round(3).to_string(index=False))

    # compare to prior systems
    try:
        bm = joblib.load("best_model.joblib")
        xgb_f1 = bm.get("macro_f1_pp_non_hidden") or bm["macro_f1_non_hidden"]
    except Exception:
        xgb_f1 = None
    crnn_f1 = 0.5633
    summary = {"xgb": xgb_f1, "crnn": crnn_f1, MODEL_NAME: f1_nh}
    print("\n=== Non-hidden Macro F1 ===")
    for k, v in summary.items():
        print(f"  {k:7s}: {v:.4f}" if v else f"  {k:7s}: n/a")

    keys = [k for k in summary if summary[k] is not None]
    plt.figure(figsize=(6, 4))
    plt.bar(keys, [summary[k] for k in keys], color=["#2e7d32", "#1565c0", "#c62828"][:len(keys)])
    for i, k in enumerate(keys):
        plt.text(i, summary[k] + 0.005, f"{summary[k]:.3f}", ha="center")
    plt.ylabel("Non-hidden Macro F1"); plt.title(f"{MODEL_NAME} vs XGB vs CRNN")
    plt.tight_layout(); plt.savefig("bonus2_comparison.png", dpi=120)
    print("saved bonus2_comparison.png")

    torch.save(best["head"].state_dict(), "best_bonus2_head.pt")
    out = {"model": MODEL_NAME, "embed_dim": int(embed_dim), "best_hidden": best["hidden"],
           "head_sweep": results, "smoothing_curve": smooth_curve,
           "best_window": int(pp_best["window"]),
           "thresholds": [float(t) for t in np.asarray(pp_best["thr"])],
           "macro_f1_non_hidden": float(f1_nh),
           "per_class": {str(r.annotation): round(float(r.f1), 4)
                         for r in per_class.itertuples()},
           "comparison": _jsonable(summary)}
    json.dump(out, open("bonus2_results.json", "w"), indent=2)
    print("saved best_bonus2_head.pt + bonus2_results.json")

    # regenerate submission if it wins
    if not smoke and (xgb_f1 is None or f1_nh > max(v for v in (xgb_f1, crnn_f1) if v)):
        print(f"\nBonus2 wins ({f1_nh:.4f}) -> regenerating submission.csv")
        pred = dl_generate_predictions(best["head"], split_map["hidden"], "test",
                                       pp_best["thr"], smooth_window=pp_best["window"])
        pred = pred.sort_values(["filename", "annotation", "onset"]).reset_index(drop=True)
        pred.to_csv("submission.csv", index=False)
        print(f"saved submission.csv ({len(pred)} rows) from Bonus2")
    elif not smoke:
        print("\nexisting submission stays best")
    return summary


if __name__ == "__main__":
    run_bonus2(smoke="--smoke" in sys.argv)













"""[head] hidden=128
  epoch  0: val macroF1@0.5 = 0.5793
  epoch  6: val macroF1@0.5 = 0.6363
  epoch 12: val macroF1@0.5 = 0.6349
  epoch 18: val macroF1@0.5 = 0.6469
  epoch 24: val macroF1@0.5 = 0.6504
  early stop at epoch 29
  best val macroF1@0.5 = 0.6534
  local-val Macro F1 = 0.6712

[head] hidden=256
  epoch  0: val macroF1@0.5 = 0.6121
  epoch  6: val macroF1@0.5 = 0.6430
  epoch 12: val macroF1@0.5 = 0.6533
  epoch 18: val macroF1@0.5 = 0.6527
  epoch 24: val macroF1@0.5 = 0.6566
  early stop at epoch 27
  best val macroF1@0.5 = 0.6633
  local-val Macro F1 = 0.6794

[head] hidden=512
  epoch  0: val macroF1@0.5 = 0.6128
  epoch  6: val macroF1@0.5 = 0.6549
  epoch 12: val macroF1@0.5 = 0.6582
  epoch 18: val macroF1@0.5 = 0.6651
  early stop at epoch 19
  best val macroF1@0.5 = 0.6687
  local-val Macro F1 = 0.6844
  smooth W=1: local-val Macro F1 = 0.6844
  smooth W=3: local-val Macro F1 = 0.6887
  smooth W=5: local-val Macro F1 = 0.6716
  smooth W=7: local-val Macro F1 = 0.6544

=== Bonus2 (frame_mn10) non-hidden Macro F1 = 0.6755 (hidden=512, smooth W=3) ===
                annotation  precision  recall    f1  map
              bell_ringing      0.693   0.591 0.638 None
            coffee_machine      0.823   0.613 0.702 None
            cutlery_dishes      0.764   0.673 0.715 None
           door_open_close      0.591   0.525 0.556 None
                 footsteps      0.658   0.717 0.686 None
           keyboard_typing      0.892   0.880 0.886 None
                  keychain      0.762   0.640 0.696 None
              light_switch      0.361   0.514 0.424 None
                 microwave      0.733   0.741 0.737 None
             phone_ringing      0.768   0.797 0.782 None
             running_water      0.879   0.834 0.856 None
           toilet_flushing      0.816   0.735 0.773 None
            vacuum_cleaner      0.884   0.820 0.851 None
wardrobe_drawer_open_close      0.381   0.518 0.439 None
         window_open_close      0.337   0.467 0.391 None

=== Non-hidden Macro F1 ===
  xgb    : 0.6017
  crnn   : 0.5633
  frame_mn10: 0.6755
saved bonus2_comparison.png
saved best_bonus2_head.pt + bonus2_results.json"""