"""
bonus.py  --  Bonus 1: Deep-Learning-based SED (CRNN + LSTM on raw waveforms)
=============================================================================
Self-contained, functions in execution order. Both DL models output per-segment
scores on the SAME 1s/0.5-hop grid as the classical pipeline, so they reuse the
existing SED backbone (thresholds -> intervals -> evaluate.py F1 -> CSV) and the
same post-processing / error-analysis steps as RF/XGB.

Speed design (RTX 4060):
  * patches are CACHED to disk ONCE (raw log-mel -> per-segment patches), so
    epochs do no librosa work and almost no slicing -- just a fast load.
  * mixed precision (AMP) + batch 32 on GPU.
  * early stopping; CRNN sweep trimmed (2 values/HP), LSTM sweep full (3).
  * `python bonus.py --smoke` runs 1 epoch on ~50 files to verify shapes fast.

Labels + the segment grid come from the FEATURE .npz (start_time, annotations);
the waveform comes from the RAW dataset. Both are keyed by filename.

Run:  python bonus.py            (full)
      python bonus.py --smoke    (2-minute end-to-end shape check)
"""

import os
import sys
import glob
import copy
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import librosa
from sklearn.metrics import f1_score

from data import CLASS_NAMES, list_split_files, aggregate_labels, PATH_VAL
from sed_pipeline import predictions_to_intervals, evaluate_sed, tune_thresholds
from post_processing import smooth_scores

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")

PATH_RAW  = os.environ.get("SED_RAW_ROOT", "data/raw")
CACHE_DIR = "cache"                    # cached patches + meta live here, per split

SR, N_FFT, HOP, N_MELS = 32000, 1024, 320, 128
FRAMES_PER_SEG = SR // HOP             # 100 frames per 1-second window
FPS = SR / HOP
BATCH = 32

LABEL_CFG = {"overlap_thresh": 0.0, "vote": "majority"}


def _jsonable(o):
    """Recursively convert numpy types / arrays so json.dump can handle them."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# =========================================================================== #
# 1. DATA HANDLING  --  cache patches to disk ONCE, then fast loads
# =========================================================================== #
def _feature_npz_path(filename, split):
    from data import PATH_TRAIN, PATH_VAL as _PV, PATH_TEST
    root = {"train": PATH_TRAIN, "validation": _PV, "test": PATH_TEST}[split]
    return os.path.join(root, "audio_features", os.path.basename(filename))


def _segment_patches(logmel, start_times):
    n_mels, T = logmel.shape
    patches = np.zeros((len(start_times), n_mels, FRAMES_PER_SEG), np.float32)
    for i, t in enumerate(start_times):
        f0 = int(round(t * FPS)); f1 = f0 + FRAMES_PER_SEG
        seg = logmel[:, f0:min(f1, T)]
        patches[i, :, :seg.shape[1]] = seg
    return patches


def build_cache(split, max_files=None):
    """One pass: wav -> log-mel -> per-segment patches, cached to disk.
    Writes <id>_patch.npy (fp16) and <id>_meta.npz (env, labels, start).
    Accumulates global mel mean/std (for standardisation) and saves them.
    Idempotent: skips files already cached."""
    out = os.path.join(CACHE_DIR, split); os.makedirs(out, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(PATH_RAW, split, "audio", "*.wav")))
    if max_files:
        wavs = wavs[:max_files]
    print(f"[{split}] caching patches for {len(wavs)} files -> {out}")

    s = ss = n = 0.0
    for i, wp in enumerate(wavs):
        fid = os.path.basename(wp).replace(".wav", "")
        ppath = os.path.join(out, fid + "_patch.npy")
        mpath = os.path.join(out, fid + "_meta.npz")
        if os.path.exists(ppath) and os.path.exists(mpath):
            continue
        y, _ = librosa.load(wp, sr=SR, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                             hop_length=HOP, n_mels=N_MELS)
        logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        fz = dict(np.load(_feature_npz_path(fid + ".npz", split), allow_pickle=True))
        start = fz["start_time"]
        patches = _segment_patches(logmel, start)            # (N_seg,128,100)
        env = patches.mean(axis=-1)                          # (N_seg,128)
        labels = (aggregate_labels(fz["annotations"], **LABEL_CFG)
                  if "annotations" in fz else np.zeros((len(start), 15), np.int8))

        np.save(ppath, patches.astype(np.float16))
        np.savez(mpath, env=env.astype(np.float16),
                 labels=labels.astype(np.int8), start=start.astype(np.float32))
        s += logmel.sum(); ss += (logmel ** 2).sum(); n += logmel.size
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(wavs)}")

    # global standardisation stats (only meaningful when we actually cached train)
    if split == "train" and n > 0:
        mean = s / n; std = float(np.sqrt(ss / n - mean ** 2)) or 1.0
        json.dump({"mean": float(mean), "std": std}, open("mel_stats.json", "w"))
        print(f"  saved mel_stats.json  mean={mean:.2f} std={std:.2f}")
    return out


def get_mel_stats():
    d = json.load(open("mel_stats.json"))
    return d["mean"], d["std"]


def _cache_paths(filename, split):
    fid = os.path.basename(filename).replace(".npz", "")
    base = os.path.join(CACHE_DIR, split, fid)
    return base + "_patch.npy", base + "_meta.npz"


def load_cached(filename, split, mean, std, mode, specaug=False):
    """Fast load from cache. mode='crnn' -> patches; 'lstm' -> env. Returns
    (x_array, labels, start)."""
    ppath, mpath = _cache_paths(filename, split)
    meta = np.load(mpath)
    labels, start = meta["labels"], meta["start"]
    if mode == "crnn":
        x = np.load(ppath, mmap_mode="r").astype(np.float32)     # (T,128,100)
        x = (x - mean) / std
        if specaug:
            x = _specaug(x)
        x = x[:, None, :, :]                                     # (T,1,128,100)
    else:
        x = (meta["env"].astype(np.float32) - mean) / std        # (T,128)
    return x, labels, start


def _specaug(patch):
    p = patch.copy()
    f = np.random.randint(0, N_MELS // 8 + 1); f0 = np.random.randint(0, max(1, N_MELS - f))
    p[..., f0:f0 + f, :] = 0
    t = np.random.randint(0, FRAMES_PER_SEG // 8 + 1); t0 = np.random.randint(0, max(1, FRAMES_PER_SEG - t))
    p[..., t0:t0 + t] = 0
    return p


class SegSeqDataset(Dataset):
    def __init__(self, files, split, mean, std, mode="crnn", specaug=False):
        self.files, self.split, self.mean, self.std = files, split, mean, std
        self.mode, self.specaug = mode, specaug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, y, _ = load_cached(self.files[i], self.split, self.mean, self.std,
                              self.mode, self.specaug)
        return torch.tensor(x), torch.tensor(y.astype(np.float32))


def collate_pad(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True)
    mask = torch.arange(x_pad.shape[1])[None, :] < lengths[:, None]
    return x_pad, y_pad, mask


# =========================================================================== #
# 2. MODELS
# =========================================================================== #
class CRNN(nn.Module):
    def __init__(self, n_mels=N_MELS, cnn_ch=32, hidden=128, n_layers=2,
                 num_classes=15, dropout=0.3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(block(1, cnn_ch), block(cnn_ch, cnn_ch * 2),
                                 block(cnn_ch * 2, cnn_ch * 4), nn.AdaptiveAvgPool2d((1, 1)))
        self.emb = cnn_ch * 4
        self.lstm = nn.LSTM(self.emb, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x):                       # (B,T,1,128,100)
        B, T = x.shape[:2]
        z = x.reshape(B * T, *x.shape[2:])
        z = self.cnn(z).reshape(B, T, self.emb)
        z, _ = self.lstm(z)
        return self.head(z)


class SeqLSTM(nn.Module):
    def __init__(self, input_dim=N_MELS, hidden=128, n_layers=2,
                 num_classes=15, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x):                       # (B,T,128)
        z, _ = self.lstm(x)
        return self.head(z)


# =========================================================================== #
# 3. TRAIN / SELECT  (AMP + early stopping; select on per-seg macro-F1@0.5)
# =========================================================================== #
def _val_macro_f1(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y, m in loader:
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                s = torch.sigmoid(model(x.to(DEVICE))).float().cpu().numpy()
            mm = m.numpy()
            for b in range(s.shape[0]):
                ys.append(y.numpy()[b][mm[b]]); ps.append(s[b][mm[b]])
    y_true, y_score = np.concatenate(ys), np.concatenate(ps)
    return f1_score(y_true, (y_score >= 0.5).astype(int), average="macro", zero_division=0.0)


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=5):
    model = model.to(DEVICE)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_f1, best_state, bad = -1, None, 0
    every = max(1, epochs // 10)
    for ep in range(epochs):
        model.train()
        for x, y, m in train_loader:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss_mat = crit(model(x), y)
                mm = m.unsqueeze(-1).float()
                loss = (loss_mat * mm).sum() / (mm.sum() * y.shape[-1])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        vf1 = _val_macro_f1(model, val_loader)
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if ep % every == 0:
            print(f"  epoch {ep:2d}: val macroF1@0.5 = {vf1:.4f}")
        if bad >= patience:
            print(f"  early stop at epoch {ep} (no gain for {patience})"); break
    model.load_state_dict(best_state)
    print(f"  best val macroF1@0.5 = {best_f1:.4f}")
    return model, best_f1


# =========================================================================== #
# 4. GLUE  ->  reuse the SED backbone
# =========================================================================== #
def model_file_scores(model, filename, split, mean, std, mode):
    x, _, start = load_cached(filename, split, mean, std, mode)
    xt = torch.tensor(x)[None].to(DEVICE)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
        s = torch.sigmoid(model(xt)).float().cpu().numpy()[0]
    return s, start


def _whole(start):
    return np.isclose(start % 1.0, 0.0)


def dl_collect_whole_second(model, files, split, mean, std, mode):
    S, Y = [], []
    for f in files:
        s, start = model_file_scores(model, f, split, mean, std, mode)
        w = _whole(start); S.append(s[w])
        _, lab, _ = load_cached(f, split, mean, std, mode); Y.append(lab[w])
    return np.vstack(S), np.vstack(Y)


def dl_generate_predictions(model, files, split, thresholds, mean, std, mode,
                            smooth_window=1):
    thr = np.asarray(thresholds, np.float32); rows = []
    for f in files:
        s, start = model_file_scores(model, f, split, mean, std, mode)
        w = _whole(start); sc = s[w]
        if smooth_window > 1:
            sc = smooth_scores(sc, smooth_window, "mean")
        binary = (sc >= thr).astype(int)
        rows.extend(predictions_to_intervals(binary, start[w],
                    os.path.basename(f).replace(".npz", ".wav")))
    if not rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(rows)


def dl_evaluate(model, files, split, thresholds, ann_df, mean, std, mode, smooth_window=1):
    pred = dl_generate_predictions(model, files, split, thresholds, mean, std, mode, smooth_window)
    return evaluate_sed(pred, files, ann_df)


# =========================================================================== #
# 5. SWEEP  (CRNN trimmed 2/HP, LSTM full 3/HP; threshold-tuned F1)
# =========================================================================== #
def _loaders(split_map, mean, std, mode, specaug=False):
    tr = DataLoader(SegSeqDataset(split_map["train"], "train", mean, std, mode, specaug),
                    batch_size=BATCH, shuffle=True, collate_fn=collate_pad, num_workers=2)
    vl = DataLoader(SegSeqDataset(split_map["local_val"], "validation", mean, std, mode),
                    batch_size=BATCH, shuffle=False, collate_fn=collate_pad, num_workers=2)
    return tr, vl


def _make(mode, **cfg):
    return CRNN(**cfg) if mode == "crnn" else SeqLSTM(**cfg)


def sweep_model(mode, split_map, mean, std, ann_df, sweep1, sweep2, base_cfg,
                lr=1e-3, epochs=20, specaug=False):
    tr, vl = _loaders(split_map, mean, std, mode, specaug)
    results = {}

    def run(cfg, tag):
        print(f"\n[{mode}] {tag}: {cfg}")
        model, _ = train_model(_make(mode, **cfg), tr, vl, epochs=epochs, lr=lr)
        S, Y = dl_collect_whole_second(model, split_map["local_val"], "validation", mean, std, mode)
        thr = tune_thresholds(S, Y)
        f1, _ = dl_evaluate(model, split_map["local_val"], "validation", thr, ann_df, mean, std, mode)
        print(f"  -> local-val Macro F1 = {f1:.4f}")
        return {"cfg": cfg, "model": model, "thr": thr, "f1": f1}

    p1, vals1 = sweep1
    s1 = [run({**base_cfg, p1: v}, f"sweep1 {p1}={v}") for v in vals1]
    best1 = max(s1, key=lambda r: r["f1"]); results["sweep1"] = [(r["cfg"][p1], r["f1"]) for r in s1]
    p2, vals2 = sweep2
    fixed = {**base_cfg, p1: best1["cfg"][p1]}
    s2 = [run({**fixed, p2: v}, f"sweep2 {p2}={v}") for v in vals2]
    best2 = max(s2, key=lambda r: r["f1"]); results["sweep2"] = [(r["cfg"][p2], r["f1"]) for r in s2]
    best = max([best1, best2], key=lambda r: r["f1"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (key, pname) in zip(ax, [("sweep1", p1), ("sweep2", p2)]):
        a.plot([str(v) for v, _ in results[key]], [f for _, f in results[key]], marker="o")
        a.set_xlabel(pname); a.set_ylabel("Val Macro F1"); a.set_title(f"{mode.upper()} {key}"); a.grid(True)
    plt.tight_layout(); plt.savefig(f"{mode}_hp_search.png", dpi=120); print(f"saved {mode}_hp_search.png")
    joblib.dump({"results": results, "best_cfg": best["cfg"], "best_f1": best["f1"]},
                f"{mode}_hp_search.joblib")
    best["results"] = results
    best["sweep1_param"], best["sweep2_param"] = p1, p2
    return best


# =========================================================================== #
# 6. POST-PROCESSING (reuse mean smoothing + re-tune) & ERROR PLOT
# =========================================================================== #
def dl_sweep_smoothing(model, split_map, mean, std, mode, ann_df, windows=(1, 3, 5, 7)):
    rows = []
    for w in windows:
        S, Y = [], []
        for f in split_map["local_val"]:
            s, start = model_file_scores(model, f, "validation", mean, std, mode)
            ww = _whole(start)
            S.append(smooth_scores(s[ww], w, "mean") if w > 1 else s[ww])
            _, lab, _ = load_cached(f, "validation", mean, std, mode); Y.append(lab[ww])
        thr = tune_thresholds(np.vstack(S), np.vstack(Y))
        f1, _ = dl_evaluate(model, split_map["local_val"], "validation", thr, ann_df, mean, std, mode, w)
        print(f"  smooth W={w}: Macro F1 = {f1:.4f}")
        rows.append({"window": w, "macro_f1": f1, "thr": thr})
    return max(rows, key=lambda r: r["macro_f1"])


def dl_plot_file(model, filename, split, thresholds, mean, std, mode,
                 smooth_window=1, save_to=None):
    from matplotlib.colors import ListedColormap
    s, start = model_file_scores(model, filename, split, mean, std, mode)
    w = _whole(start)
    sc = smooth_scores(s[w], smooth_window, "mean") if smooth_window > 1 else s[w]
    pred = (sc >= np.asarray(thresholds, np.float32)).astype(int)
    _, lab, _ = load_cached(filename, split, mean, std, mode); gt = lab[w]
    x, _, _ = load_cached(filename, split, mean, std, "crnn")
    mel = x[:, 0].transpose(1, 0, 2).reshape(N_MELS, -1)
    times = start[w]; C = len(CLASS_NAMES)
    fig, ax = plt.subplots(3, 1, figsize=(13, 9),
                           gridspec_kw={"height_ratios": [1.1, 1.4, 1.4]})
    ax[0].imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax[0].set_ylabel("Mel bin"); ax[0].set_title(f"{os.path.basename(filename)} — log-mel")
    te = np.append(times, times[-1] + 1); ce = np.arange(C + 1)
    for a, mat, cmap, t in [(ax[1], gt, ListedColormap(["white", "#2e7d32"]), "Ground truth"),
                            (ax[2], pred, ListedColormap(["white", "#1565c0"]), "Predictions")]:
        a.pcolormesh(te, ce, mat.T, cmap=cmap, vmin=0, vmax=1, edgecolors="lightgray", linewidth=0.3)
        a.set_yticks(np.arange(C) + 0.5); a.set_yticklabels(CLASS_NAMES, fontsize=8)
        a.set_ylabel(t); a.set_ylim(C, 0)
    ax[2].set_xlabel("segment index"); plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=120); print(f"saved {save_to}")
    return fig


# =========================================================================== #
# 7. RUN
# =========================================================================== #
def run_bonus(smoke=False):
    files = list_split_files()
    if smoke:
        for k in files:
            files[k] = files[k][:50]
    split_map = {"train": files["train"], "local_val": files["local_val"],
                 "non_hidden": files["non_hidden"], "hidden": files["hidden"]}
    ann_df = pd.read_csv(os.path.join(PATH_VAL, "annotations.csv"))

    mx = 50 if smoke else None
    for sp in ("train", "validation", "test"):
        build_cache(sp, max_files=mx)
    mean, std = get_mel_stats()

    epochs = 1 if smoke else 20
    # CRNN: trimmed 2/HP ; LSTM: full 3/HP. Smoke: single value each.
    crnn_sw = (("hidden", [128] if smoke else [64, 128]),
               ("dropout", [0.3] if smoke else [0.2, 0.4]))
    lstm_sw = (("hidden", [128] if smoke else [64, 128, 256]),
               ("n_layers", [2] if smoke else [1, 2, 3]))

    summary, best_models = {}, {}
    json_out = {"config": {"SR": SR, "HOP": HOP, "N_MELS": N_MELS,
                           "frames_per_seg": FRAMES_PER_SEG, "batch": BATCH,
                           "epochs": epochs, "label_cfg": LABEL_CFG}, "models": {}}
    for mode, base_cfg, sw in [
        ("crnn", dict(cnn_ch=32, hidden=128, n_layers=2, dropout=0.3), crnn_sw),
        ("lstm", dict(input_dim=N_MELS, hidden=128, n_layers=2, dropout=0.3), lstm_sw),
    ]:
        print(f"\n{'='*60}\n{mode.upper()}\n{'='*60}")
        best = sweep_model(mode, split_map, mean, std, ann_df, sw[0], sw[1], base_cfg, epochs=epochs)
        pp = dl_sweep_smoothing(best["model"], split_map, mean, std, mode, ann_df,
                                windows=(1, 3) if smoke else (1, 3, 5, 7))
        f1_nh, _ = dl_evaluate(best["model"], split_map["non_hidden"], "validation",
                               pp["thr"], ann_df, mean, std, mode, pp["window"])
        print(f"\n[{mode}] non-hidden Macro F1 = {f1_nh:.4f} (smooth W={pp['window']})")
        summary[mode] = f1_nh
        best_models[mode] = {"model": best["model"], "cfg": best["cfg"], "thr": pp["thr"],
                             "window": pp["window"], "mode": mode, "f1_non_hidden": f1_nh}
        torch.save(best["model"].state_dict(), f"best_{mode}.pt")
        joblib.dump({k: v for k, v in best_models[mode].items() if k != "model"},
                    f"best_{mode}_meta.joblib")

        # accumulate JSON record and write after each model (crash-safe)
        json_out["models"][mode] = _jsonable({
            "best_cfg": best["cfg"],
            "best_val_f1@0.5": best["f1"],
            "sweep1": {"param": best["sweep1_param"], "values": best["results"]["sweep1"]},
            "sweep2": {"param": best["sweep2_param"], "values": best["results"]["sweep2"]},
            "post_processing": {"smooth": "mean", "best_window": pp["window"],
                                "val_f1": pp["macro_f1"]},
            "thresholds": pp["thr"],
            "macro_f1_non_hidden": f1_nh,
        })
        json.dump(json_out, open("bonus_results.json", "w"), indent=2)
        print("updated bonus_results.json")

    try:
        bm = joblib.load("best_model.joblib")
        xgb_f1 = bm.get("macro_f1_pp_non_hidden") or bm["macro_f1_non_hidden"]
    except Exception:
        xgb_f1 = None
    summary["xgb"] = xgb_f1
    print("\n=== Non-hidden Macro F1 summary ===")
    for k, v in summary.items():
        print(f"  {k:5s}: {v:.4f}" if v else f"  {k:5s}: n/a")

    keys = [k for k in summary if summary[k] is not None]
    plt.figure(figsize=(6, 4))
    plt.bar(keys, [summary[k] for k in keys], color=["#1565c0", "#6a1b9a", "#2e7d32"][:len(keys)])
    for i, k in enumerate(keys):
        plt.text(i, summary[k] + 0.005, f"{summary[k]:.3f}", ha="center")
    plt.ylabel("Non-hidden Macro F1"); plt.title("Bonus: CRNN vs LSTM vs XGB")
    plt.tight_layout(); plt.savefig("bonus_comparison.png", dpi=120); print("saved bonus_comparison.png")

    dl_best = max(best_models.values(), key=lambda m: m["f1_non_hidden"])
    json_out["comparison"] = _jsonable({k: summary[k] for k in summary})
    json_out["winner"] = (dl_best["mode"] if (xgb_f1 is None or
                          dl_best["f1_non_hidden"] > xgb_f1) else "xgb")
    json.dump(json_out, open("bonus_results.json", "w"), indent=2)
    print("wrote final bonus_results.json")

    if not smoke and (xgb_f1 is None or dl_best["f1_non_hidden"] > xgb_f1):
        print(f"\n{dl_best['mode'].upper()} wins -> regenerating submission.csv")
        pred = dl_generate_predictions(dl_best["model"], split_map["hidden"], "test",
                                       dl_best["thr"], mean, std, dl_best["mode"], dl_best["window"])
        pred = pred.sort_values(["filename", "annotation", "onset"]).reset_index(drop=True)
        pred.to_csv("submission.csv", index=False)
        print(f"saved submission.csv ({len(pred)} rows) from {dl_best['mode'].upper()}")
    elif not smoke:
        print(f"\nXGB ({xgb_f1:.4f}) still best -> keeping existing submission.csv")
    return summary


if __name__ == "__main__":
    run_bonus(smoke="--smoke" in sys.argv)






"""

[train] caching patches for 3704 files -> cache\train
  1000/3704
  1500/3704
  2000/3704
  2500/3704
  3000/3704
  3500/3704
  saved mel_stats.json  mean=-59.04 std=16.81
[validation] caching patches for 999 files -> cache\validation
  500/999
[test] caching patches for 1007 files -> cache\test
  500/1007
  1000/1007

============================================================
CRNN
============================================================

[crnn] sweep1 hidden=64: {'cnn_ch': 32, 'hidden': 64, 'n_layers': 2, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0001
  epoch  2: val macroF1@0.5 = 0.0486
  epoch  4: val macroF1@0.5 = 0.1528
  epoch  6: val macroF1@0.5 = 0.2674
  epoch  8: val macroF1@0.5 = 0.2950
  epoch 10: val macroF1@0.5 = 0.3040
  epoch 12: val macroF1@0.5 = 0.3773
  epoch 14: val macroF1@0.5 = 0.3580
  epoch 16: val macroF1@0.5 = 0.4192
  epoch 18: val macroF1@0.5 = 0.4638
  best val macroF1@0.5 = 0.4638
  -> local-val Macro F1 = 0.5628

[crnn] sweep1 hidden=128: {'cnn_ch': 32, 'hidden': 128, 'n_layers': 2, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0220
  epoch  2: val macroF1@0.5 = 0.1310
  epoch  4: val macroF1@0.5 = 0.2412
  epoch  6: val macroF1@0.5 = 0.2915
  epoch  8: val macroF1@0.5 = 0.3592
  epoch 10: val macroF1@0.5 = 0.4001
  epoch 12: val macroF1@0.5 = 0.4314
  epoch 14: val macroF1@0.5 = 0.4230
  epoch 16: val macroF1@0.5 = 0.4996
  epoch 18: val macroF1@0.5 = 0.5124
  best val macroF1@0.5 = 0.5124
  -> local-val Macro F1 = 0.5845

[crnn] sweep2 dropout=0.2: {'cnn_ch': 32, 'hidden': 128, 'n_layers': 2, 'dropout': 0.2}
  epoch  0: val macroF1@0.5 = 0.0001
  epoch  2: val macroF1@0.5 = 0.1194
  epoch  4: val macroF1@0.5 = 0.2439
  epoch  6: val macroF1@0.5 = 0.2891
  epoch  8: val macroF1@0.5 = 0.3064
  epoch 10: val macroF1@0.5 = 0.3658
  epoch 12: val macroF1@0.5 = 0.3928
  epoch 14: val macroF1@0.5 = 0.4253
  epoch 16: val macroF1@0.5 = 0.4657
  epoch 18: val macroF1@0.5 = 0.5063
  best val macroF1@0.5 = 0.5158
  -> local-val Macro F1 = 0.5921

[crnn] sweep2 dropout=0.4: {'cnn_ch': 32, 'hidden': 128, 'n_layers': 2, 'dropout': 0.4}
  epoch  0: val macroF1@0.5 = 0.0375
  epoch  2: val macroF1@0.5 = 0.1122
  epoch  4: val macroF1@0.5 = 0.2257
  epoch  6: val macroF1@0.5 = 0.2736
  epoch  8: val macroF1@0.5 = 0.3478
  epoch 10: val macroF1@0.5 = 0.4031
  epoch 12: val macroF1@0.5 = 0.4678
  epoch 14: val macroF1@0.5 = 0.4356
  epoch 16: val macroF1@0.5 = 0.5002
  epoch 18: val macroF1@0.5 = 0.4779
  best val macroF1@0.5 = 0.5002
  -> local-val Macro F1 = 0.5767
saved crnn_hp_search.png
  smooth W=1: Macro F1 = 0.5921
  smooth W=3: Macro F1 = 0.5871
  smooth W=5: Macro F1 = 0.5738
  smooth W=7: Macro F1 = 0.5588

[crnn] non-hidden Macro F1 = 0.5633 (smooth W=1)
updated bonus_results.json

============================================================
LSTM
============================================================

[lstm] sweep1 hidden=64: {'input_dim': 128, 'hidden': 64, 'n_layers': 2, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0271
  epoch  2: val macroF1@0.5 = 0.0781
  epoch  4: val macroF1@0.5 = 0.0817
  epoch  6: val macroF1@0.5 = 0.1222
  epoch  8: val macroF1@0.5 = 0.1670
  epoch 10: val macroF1@0.5 = 0.1955
  epoch 12: val macroF1@0.5 = 0.2314
  epoch 14: val macroF1@0.5 = 0.2444
  epoch 16: val macroF1@0.5 = 0.2741
  epoch 18: val macroF1@0.5 = 0.2879
  best val macroF1@0.5 = 0.3102
  -> local-val Macro F1 = 0.4383

[lstm] sweep1 hidden=128: {'input_dim': 128, 'hidden': 128, 'n_layers': 2, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0388
  epoch  2: val macroF1@0.5 = 0.0896
  epoch  4: val macroF1@0.5 = 0.1492
  epoch  6: val macroF1@0.5 = 0.1933
  epoch  8: val macroF1@0.5 = 0.2500
  epoch 10: val macroF1@0.5 = 0.2843
  epoch 12: val macroF1@0.5 = 0.2757
  epoch 14: val macroF1@0.5 = 0.3151
  epoch 16: val macroF1@0.5 = 0.3518
  epoch 18: val macroF1@0.5 = 0.3638
  best val macroF1@0.5 = 0.3700
  -> local-val Macro F1 = 0.4605

[lstm] sweep1 hidden=256: {'input_dim': 128, 'hidden': 256, 'n_layers': 2, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0718
  epoch  2: val macroF1@0.5 = 0.0833
  epoch  4: val macroF1@0.5 = 0.1831
  epoch  6: val macroF1@0.5 = 0.2509
  epoch  8: val macroF1@0.5 = 0.3010
  epoch 10: val macroF1@0.5 = 0.3061
  epoch 12: val macroF1@0.5 = 0.3508
  epoch 14: val macroF1@0.5 = 0.3696
  epoch 16: val macroF1@0.5 = 0.4060
  epoch 18: val macroF1@0.5 = 0.4106
  best val macroF1@0.5 = 0.4212
  -> local-val Macro F1 = 0.4882

[lstm] sweep2 n_layers=1: {'input_dim': 128, 'hidden': 256, 'n_layers': 1, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0806
  epoch  2: val macroF1@0.5 = 0.1035
  epoch  4: val macroF1@0.5 = 0.1759
  epoch  6: val macroF1@0.5 = 0.2304
  epoch  8: val macroF1@0.5 = 0.2695
  epoch 10: val macroF1@0.5 = 0.3190
  epoch 12: val macroF1@0.5 = 0.3274
  epoch 14: val macroF1@0.5 = 0.3481
  epoch 16: val macroF1@0.5 = 0.3521
  epoch 18: val macroF1@0.5 = 0.3719
  best val macroF1@0.5 = 0.3792
  -> local-val Macro F1 = 0.4665

[lstm] sweep2 n_layers=2: {'input_dim': 128, 'hidden': 256, 'n_layers': 2, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0637
  epoch  2: val macroF1@0.5 = 0.1338
  epoch  4: val macroF1@0.5 = 0.2103
  epoch  6: val macroF1@0.5 = 0.2649
  epoch  8: val macroF1@0.5 = 0.2972
  epoch 10: val macroF1@0.5 = 0.3565
  epoch 12: val macroF1@0.5 = 0.3727
  epoch 14: val macroF1@0.5 = 0.3965
  epoch 16: val macroF1@0.5 = 0.4113
  epoch 18: val macroF1@0.5 = 0.4206
  best val macroF1@0.5 = 0.4206
  -> local-val Macro F1 = 0.4947

[lstm] sweep2 n_layers=3: {'input_dim': 128, 'hidden': 256, 'n_layers': 3, 'dropout': 0.3}
  epoch  0: val macroF1@0.5 = 0.0379
  epoch  2: val macroF1@0.5 = 0.0558
  epoch  4: val macroF1@0.5 = 0.1515
  epoch  6: val macroF1@0.5 = 0.2340
  epoch  8: val macroF1@0.5 = 0.3170
  epoch 10: val macroF1@0.5 = 0.3375
  epoch 12: val macroF1@0.5 = 0.3417
  epoch 14: val macroF1@0.5 = 0.3823
  epoch 16: val macroF1@0.5 = 0.4066
  epoch 18: val macroF1@0.5 = 0.4203
  best val macroF1@0.5 = 0.4203
  -> local-val Macro F1 = 0.4801
saved lstm_hp_search.png
  smooth W=1: Macro F1 = 0.4947
  smooth W=3: Macro F1 = 0.4917
  smooth W=5: Macro F1 = 0.4791
  smooth W=7: Macro F1 = 0.4703

[lstm] non-hidden Macro F1 = 0.4811 (smooth W=1)
updated bonus_results.json

=== Non-hidden Macro F1 summary ===
  crnn : 0.5633
  lstm : 0.4811
  xgb  : 0.6017
saved bonus_comparison.png
wrote final bonus_results.json

XGB (0.6017) still best -> keeping existing submission.csv
"""
