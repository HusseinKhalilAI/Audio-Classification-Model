"""
improve_bonus2.py
-----------------
Every improvement on top of the CACHED frame_mn10 embeddings (emb_frame_mn10/),
in one file. Imports bonus2 + the shared SED backbone, works straight from the
cache -> each experiment runs in SECONDS (no backbone, no re-embedding).

Improvements wired in (run + compared automatically):
  1. Deeper MLP head, ALL training data, AdamW + cosine, CLASS-BALANCED loss
     (pos_weight) -> lifts the weak rare/short classes.
  2. BiGRU sequence head over each recording's per-second embeddings -> temporal
     context (the PretrainedSED paper's recipe on frozen features).
  3. Ensemble: average the best DL head's scores with the tuned XGB scores.
The best non-hidden system regenerates submission.csv if it beats 0.6828.

Optional (heavy, NOT run by default): finetune_backbone() unfreezes frame_mn10
end to end -- highest ceiling, but needs the backbone and cannot use the cache.

Run:  python improve_bonus2.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

import bonus2 as B
from challenge_data import build_feature_matrix
from sed_pipeline import (predictions_to_intervals, evaluate_sed, tune_thresholds,
                          make_sklearn_score_fn)
from post_processing import smooth_scores

DEV = B.DEVICE
FROZEN_BASELINE = 0.6828          # current best (frame_mn10 frozen MLP head)


# =========================================================================== #
# Data from cache  (flat per-segment AND per-recording sequences)
# =========================================================================== #
def load_flat(files, split):
    return B._stack(files, split)                      # (N,960), (N,15)


def load_seq(files, split):
    seqs = []
    for f in files:
        feats, lab, start = B.load_emb(f, split)
        seqs.append((feats, lab.astype(np.float32), start))
    return seqs


def pos_weight(Y, cap=20.0):
    """Per-class neg/pos ratio for class-balanced BCE; capped to avoid blow-ups."""
    pos = Y.sum(axis=0); neg = len(Y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.minimum(w, cap), dtype=torch.float32, device=DEV)


# =========================================================================== #
# Heads
# =========================================================================== #
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=512, n_classes=15, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes))

    def forward(self, x):
        return self.net(x)


class GRUHead(nn.Module):
    """BiGRU over a recording's per-second embedding sequence -> 15 logits/sec."""
    def __init__(self, in_dim, hidden=256, n_layers=2, n_classes=15, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(in_dim, hidden, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):                      # (B,T,in)
        return self.head(self.gru(self.norm(x))[0])


class GRUMLPHead(nn.Module):
    """Everything stacked: BiGRU (temporal context) -> deep MLP -> 15 logits/sec."""
    def __init__(self, in_dim, gru_hidden=256, mlp_hidden=512, n_layers=2,
                 n_classes=15, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(in_dim, gru_hidden, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden * 2, mlp_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, n_classes))

    def forward(self, x):                      # (B,T,in)
        return self.mlp(self.gru(self.norm(x))[0])


# =========================================================================== #
# Training
# =========================================================================== #
def train_flat(Xtr, Ytr, Xvl, Yvl, in_dim, hidden=512, dropout=0.3, lr=1e-3,
               epochs=120, patience=12, balanced=True):
    head = MLPHead(in_dim, hidden, 15, dropout).to(DEV)
    pw = pos_weight(Ytr) if balanced else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(Ytr, dtype=torch.float32)),
                    batch_size=1024, shuffle=True)
    Xvl_t = torch.tensor(Xvl, dtype=torch.float32, device=DEV)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); crit(head(xb), yb).backward(); opt.step()
        sched.step(); head.eval()
        with torch.no_grad():
            sv = torch.sigmoid(head(Xvl_t)).cpu().numpy()
        vf1 = f1_score(Yvl, (sv >= 0.5).astype(int), average="macro", zero_division=0.0)
        if vf1 > best:
            best, state, bad = vf1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
        if ep % max(1, epochs // 10) == 0:
            print(f"    epoch {ep:3d}: val macroF1@0.5 = {vf1:.4f}")
        if bad >= patience:
            print(f"    early stop at {ep}"); break
    head.load_state_dict(state); print(f"    best val@0.5 = {best:.4f}")
    return head


class SeqDS(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        f, l, _ = self.seqs[i]
        return torch.tensor(f, dtype=torch.float32), torch.tensor(l, dtype=torch.float32)


def collate(batch):
    xs, ys = zip(*batch)
    L = torch.tensor([x.shape[0] for x in xs])
    xp = pad_sequence(xs, batch_first=True); yp = pad_sequence(ys, batch_first=True)
    mask = torch.arange(xp.shape[1])[None, :] < L[:, None]
    return xp, yp, mask


def train_seq(head, tr_seqs, vl_seqs, Ytr_flat, lr=1e-3, epochs=80,
              patience=10, balanced=True):
    head = head.to(DEV)
    pw = pos_weight(Ytr_flat) if balanced else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = DataLoader(SeqDS(tr_seqs), batch_size=32, shuffle=True, collate_fn=collate)
    vl = DataLoader(SeqDS(vl_seqs), batch_size=32, shuffle=False, collate_fn=collate)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        head.train()
        for xp, yp, m in tr:
            xp, yp, m = xp.to(DEV), yp.to(DEV), m.to(DEV)
            opt.zero_grad()
            lm = crit(head(xp), yp); mm = m.unsqueeze(-1).float()
            ((lm * mm).sum() / (mm.sum() * yp.shape[-1])).backward(); opt.step()
        sched.step(); head.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xp, yp, m in vl:
                s = torch.sigmoid(head(xp.to(DEV))).cpu().numpy(); mm = m.numpy()
                for b in range(s.shape[0]):
                    ys.append(yp.numpy()[b][mm[b]]); ps.append(s[b][mm[b]])
        yt, pt = np.concatenate(ys), np.concatenate(ps)
        vf1 = f1_score(yt, (pt >= 0.5).astype(int), average="macro", zero_division=0.0)
        if vf1 > best:
            best, state, bad = vf1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
        if ep % max(1, epochs // 10) == 0:
            print(f"    epoch {ep:3d}: val macroF1@0.5 = {vf1:.4f}")
        if bad >= patience:
            print(f"    early stop at {ep}"); break
    head.load_state_dict(state); print(f"    best val@0.5 = {best:.4f}")
    return head


# =========================================================================== #
# Scorers (decoupled) + generic eval through the SED backbone
# =========================================================================== #
def flat_scorer(head):
    def fn(filename, split):
        feats, _, start = B.load_emb(filename, split)
        head.eval()
        with torch.no_grad():
            s = torch.sigmoid(head(torch.tensor(feats, device=DEV))).cpu().numpy()
        return s, start
    return fn


def seq_scorer(head):
    def fn(filename, split):
        feats, _, start = B.load_emb(filename, split)
        head.eval()
        with torch.no_grad():
            s = torch.sigmoid(head(torch.tensor(feats[None], device=DEV))).cpu().numpy()[0]
        return s, start
    return fn


def xgb_scorer():
    """Tuned XGB scores on the SAME grid, from the npz features (for ensembling)."""
    import joblib
    saved = joblib.load("best_xgb.joblib")
    sf = make_sklearn_score_fn(saved["model"])
    feat_files = {os.path.basename(f).replace(".npz", ""): f
                  for f in sum(B.list_split_files().values(), [])}

    def fn(filename, split):
        fid = os.path.basename(filename).replace(".npz", "")
        d = dict(np.load(feat_files[fid], allow_pickle=True))
        return sf(build_feature_matrix(d)), d["start_time"]
    return fn


def ensemble_scorer(scorers, weights=None):
    weights = weights or [1.0] * len(scorers)
    def fn(filename, split):
        accum, start = None, None
        for w, sc in zip(weights, scorers):
            s, st = sc(filename, split)
            accum = w * s if accum is None else accum + w * s
            start = st
        return accum / sum(weights), start
    return fn


def _whole(start):
    return np.isclose(start % 1.0, 0.0)


def gen_preds(scorer, files, split, thr, w=1):
    thr = np.asarray(thr, np.float32); rows = []
    for f in files:
        s, start = scorer(f, split); m = _whole(start); sc = s[m]
        if w > 1:
            sc = smooth_scores(sc, w, "mean")
        binary = (sc >= thr).astype(int)
        rows.extend(predictions_to_intervals(binary, start[m],
                    os.path.basename(f).replace(".npz", ".wav")))
    return (pd.DataFrame(rows) if rows else
            pd.DataFrame(columns=["filename", "annotation", "onset", "offset"]))


def collect_ws(scorer, files, split):
    S, Y = [], []
    for f in files:
        s, start = scorer(f, split); m = _whole(start)
        S.append(s[m]); Y.append(B.load_emb(f, split)[1][m])
    return np.vstack(S), np.vstack(Y)


def tune_and_eval(scorer, split_map, ann_df, name=""):
    """Tune thresholds on local-val, sweep smoothing, report non-hidden F1."""
    S, Y = collect_ws(scorer, split_map["local_val"], "validation")
    thr = tune_thresholds(S, Y)
    best = {"w": 1, "thr": thr,
            "f1": evaluate_sed(gen_preds(scorer, split_map["local_val"], "validation", thr, 1),
                               split_map["local_val"], ann_df)[0]}
    for w in (3, 5):
        Sw = []
        for f in split_map["local_val"]:
            s, start = scorer(f, "validation"); m = _whole(start)
            Sw.append(smooth_scores(s[m], w, "mean"))
        thrw = tune_thresholds(np.vstack(Sw), Y)
        f1w = evaluate_sed(gen_preds(scorer, split_map["local_val"], "validation", thrw, w),
                           split_map["local_val"], ann_df)[0]
        if f1w > best["f1"]:
            best = {"w": w, "thr": thrw, "f1": f1w}
    f1_nh, per = evaluate_sed(gen_preds(scorer, split_map["non_hidden"], "validation",
                                        best["thr"], best["w"]),
                              split_map["non_hidden"], ann_df)
    print(f"  [{name}] non-hidden Macro F1 = {f1_nh:.4f} (smooth W={best['w']})")
    return f1_nh, per, best


# =========================================================================== #
# OPTIONAL heavy path: unfreeze the backbone and fine-tune end to end
# =========================================================================== #
def finetune_backbone():
    """Highest ceiling, but slow and cannot use the cache (features change every
    step). Unfreezes frame_mn10 and trains it + a head on raw audio at a low LR.
    Left as an explicit call -- ask to wire the full raw-audio loop if you want it."""
    raise NotImplementedError(
        "Run the cache-based improvements first. The full fine-tune needs the "
        "backbone + a raw-audio training loop (~hours on the 4060).")


# =========================================================================== #
if __name__ == "__main__":
    files = B.list_split_files()
    split_map = {"train": files["train"], "local_val": files["local_val"],
                 "non_hidden": files["non_hidden"], "hidden": files["hidden"]}
    ann_df = pd.read_csv(os.path.join(B.PATH_VAL, "annotations.csv"))

    print("loading cached embeddings (flat + sequences)...")
    Xtr, Ytr = load_flat(split_map["train"], "train")
    Xvl, Yvl = load_flat(split_map["local_val"], "validation")
    in_dim = Xtr.shape[1]
    print(f"train {Xtr.shape}  val {Xvl.shape}")

    board = {"frozen_baseline": (FROZEN_BASELINE, None, None)}

    print("\n[1] deep MLP + class-balanced loss")
    h1 = train_flat(Xtr, Ytr, Xvl, Yvl, in_dim, hidden=512, balanced=True)
    board["mlp_balanced"] = tune_and_eval(flat_scorer(h1), split_map, ann_df, "mlp")

    print("\n[2] BiGRU sequence head")
    tr_seq = load_seq(split_map["train"], "train")
    vl_seq = load_seq(split_map["local_val"], "validation")
    h2 = train_seq(GRUHead(in_dim, hidden=256, n_layers=2), tr_seq, vl_seq, Ytr)
    board["bigru"] = tune_and_eval(seq_scorer(h2), split_map, ann_df, "bigru")

    print("\n[3] combined: BiGRU -> deep MLP + class-balanced (everything stacked)")
    h3 = train_seq(GRUMLPHead(in_dim, gru_hidden=256, mlp_hidden=512, n_layers=2),
                   tr_seq, vl_seq, Ytr)
    board["combined"] = tune_and_eval(seq_scorer(h3), split_map, ann_df, "combined")

    print("\n[4] ensemble (best DL head + XGB)")
    dl_candidates = {"mlp_balanced": (board["mlp_balanced"][0], flat_scorer(h1)),
                     "bigru": (board["bigru"][0], seq_scorer(h2)),
                     "combined": (board["combined"][0], seq_scorer(h3))}
    dl_name = max(dl_candidates, key=lambda k: dl_candidates[k][0])
    dl_sc = dl_candidates[dl_name][1]
    try:
        ens = ensemble_scorer([dl_sc, xgb_scorer()], weights=[0.6, 0.4])
        board["ensemble_dl_xgb"] = tune_and_eval(ens, split_map, ann_df, "ensemble")
    except Exception as e:
        print(f"  ensemble skipped ({e})")
        ens = None

    print("\n=== non-hidden Macro F1 ===")
    for k, v in sorted(board.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:16s}: {v[0]:.4f}")
    best_name = max(board, key=lambda k: board[k][0])
    best_f1, best_per, best_cfg = board[best_name]
    print(f"\nbest: {best_name} = {best_f1:.4f}  (frozen baseline {FROZEN_BASELINE})")
    if best_per is not None:
        print(best_per.round(3).to_string(index=False))

    out = {}
    for k, (f1, per, cfg) in board.items():
        rec = {"macro_f1": float(f1)}
        if per is not None:
            rec["per_class"] = {str(r.annotation): round(float(r.f1), 4)
                                for r in per.itertuples()}
            rec["smooth_window"] = int(cfg["w"]) if cfg else 1
        out[k] = rec
    json.dump(out, open("improve_results.json", "w"), indent=2)

    # ---- before/after plot ----
    import matplotlib.pyplot as plt
    order = ["frozen_baseline", "mlp_balanced", "bigru", "combined", "ensemble_dl_xgb"]
    order = [k for k in order if k in board]
    vals = [board[k][0] for k in order]
    colors = ["#9e9e9e"] + ["#1565c0"] * (len(order) - 1)
    colors = [("#2e7d32" if order[i] == best_name and best_name != "frozen_baseline"
               else colors[i]) for i in range(len(order))]
    plt.figure(figsize=(8, 4.5))
    plt.bar(order, vals, color=colors)
    plt.axhline(FROZEN_BASELINE, color="#9e9e9e", ls="--", lw=1,
                label=f"before improvement ({FROZEN_BASELINE:.3f})")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("Non-hidden Macro F1")
    plt.title("Bonus 2 improvements: before vs after")
    plt.ylim(min(vals) - 0.03, max(vals) + 0.03)
    plt.xticks(rotation=20, ha="right"); plt.legend()
    plt.tight_layout(); plt.savefig("improve_comparison.png", dpi=120)
    print("saved improve_comparison.png")

    if best_name != "frozen_baseline" and best_f1 > FROZEN_BASELINE:
        sc = {"mlp_balanced": flat_scorer(h1), "bigru": seq_scorer(h2),
              "combined": seq_scorer(h3), "ensemble_dl_xgb": ens}[best_name]
        pred = gen_preds(sc, split_map["hidden"], "test", best_cfg["thr"], best_cfg["w"])
        pred = pred.sort_values(["filename", "annotation", "onset"]).reset_index(drop=True)
        pred.to_csv("submission.csv", index=False)
        print(f"\nimproved -> submission.csv ({len(pred)} rows) from {best_name}")
    else:
        print("\nno improvement over frozen baseline -> submission unchanged")









"""loading cached embeddings (flat + sequences)...
train (170508, 960)  val (23404, 960)

[1] deep MLP + class-balanced loss
    epoch   0: val macroF1@0.5 = 0.5530
    epoch  12: val macroF1@0.5 = 0.6167
    epoch  24: val macroF1@0.5 = 0.6307
    epoch  36: val macroF1@0.5 = 0.6406
    epoch  48: val macroF1@0.5 = 0.6468
    epoch  60: val macroF1@0.5 = 0.6467
    epoch  72: val macroF1@0.5 = 0.6527
    epoch  84: val macroF1@0.5 = 0.6592
    epoch  96: val macroF1@0.5 = 0.6603
    epoch 108: val macroF1@0.5 = 0.6629
    best val@0.5 = 0.6638
  [mlp] non-hidden Macro F1 = 0.6804 (smooth W=1)

[2] BiGRU sequence head
    epoch   0: val macroF1@0.5 = 0.5204
    epoch   8: val macroF1@0.5 = 0.6688
    epoch  16: val macroF1@0.5 = 0.7160
    epoch  24: val macroF1@0.5 = 0.7273
    epoch  32: val macroF1@0.5 = 0.7331
    early stop at 37
    best val@0.5 = 0.7350
  [bigru] non-hidden Macro F1 = 0.7122 (smooth W=1)

[3] combined: BiGRU -> deep MLP + class-balanced (everything stacked)
    epoch   0: val macroF1@0.5 = 0.5092
    epoch   8: val macroF1@0.5 = 0.6404
    epoch  16: val macroF1@0.5 = 0.6841
    epoch  24: val macroF1@0.5 = 0.7103
    epoch  32: val macroF1@0.5 = 0.7153
    epoch  40: val macroF1@0.5 = 0.7296
    epoch  48: val macroF1@0.5 = 0.7346
    epoch  56: val macroF1@0.5 = 0.7324
    epoch  64: val macroF1@0.5 = 0.7332
    early stop at 68
    best val@0.5 = 0.7359
  [combined] non-hidden Macro F1 = 0.7258 (smooth W=1)

[4] ensemble (best DL head + XGB)
  [ensemble] non-hidden Macro F1 = 0.7301 (smooth W=1)

=== non-hidden Macro F1 ===
  ensemble_dl_xgb : 0.7301
  combined        : 0.7258
  bigru           : 0.7122
  frozen_baseline : 0.6828
  mlp_balanced    : 0.6804

best: ensemble_dl_xgb = 0.7301  (frozen baseline 0.6828)
                annotation  precision  recall    f1  map
              bell_ringing      0.626   0.573 0.599 None
            coffee_machine      0.939   0.621 0.747 None
            cutlery_dishes      0.753   0.795 0.773 None
           door_open_close      0.655   0.639 0.647 None
                 footsteps      0.733   0.726 0.730 None
           keyboard_typing      0.921   0.892 0.907 None
                  keychain      0.792   0.691 0.738 None
              light_switch      0.607   0.614 0.610 None
                 microwave      0.834   0.774 0.803 None
             phone_ringing      0.846   0.770 0.806 None
             running_water      0.887   0.876 0.881 None
           toilet_flushing      0.879   0.835 0.856 None
            vacuum_cleaner      0.907   0.854 0.880 None
wardrobe_drawer_open_close      0.470   0.609 0.531 None
         window_open_close      0.502   0.397 0.444 None
saved improve_comparison.png

improved -> submission.csv (4534 rows) from ensemble_dl_xgb"""