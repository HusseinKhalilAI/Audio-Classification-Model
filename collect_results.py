"""
collect_results.py
------------------
Greedy results collector for the whole MLPC Task 5 project. It scrapes every
artifact on disk (joblib sweeps, best models, bonus/bonus2/improve JSONs) AND
re-evaluates the cheaply-reconstructable classical models on the non-hidden test
set to regenerate their per-class tables (those weren't saved originally).

Outputs (in results_tables/):
  overall_scores.csv / .md   -- every system's non-hidden Macro F1
  per_class_f1.csv  / .md    -- 15 classes x every system with per-class data
  sweeps.md                  -- all HP sweeps + smoothing curves

Run:  python collect_results.py
Everything is wrapped in try/except so missing files are skipped, not fatal.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

OUT = "results_tables"
os.makedirs(OUT, exist_ok=True)

CLASSES = ["bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
           "footsteps", "keyboard_typing", "keychain", "light_switch", "microwave",
           "phone_ringing", "running_water", "toilet_flushing", "vacuum_cleaner",
           "wardrobe_drawer_open_close", "window_open_close"]

# hardcoded Phase-1 baseline (decision tree, untuned) -- non-hidden per-class
BASELINE = {"bell_ringing": 0.29, "coffee_machine": 0.36, "cutlery_dishes": 0.29,
            "door_open_close": 0.18, "footsteps": 0.33, "keyboard_typing": 0.36,
            "keychain": 0.32, "light_switch": 0.23, "microwave": 0.39,
            "phone_ringing": 0.48, "running_water": 0.58, "toilet_flushing": 0.29,
            "vacuum_cleaner": 0.47, "wardrobe_drawer_open_close": 0.12,
            "window_open_close": 0.06}

overall = []      # list of {system, macro_f1, source}
per_class = {}    # system -> {class -> f1}
sweeps = []       # list of (title, DataFrame)


def add_overall(system, f1, source):
    if f1 is not None:
        overall.append({"system": system, "macro_f1": round(float(f1), 4), "source": source})


# --------------------------------------------------------------------------- #
# 1. baseline (hardcoded)
# --------------------------------------------------------------------------- #
add_overall("baseline_DT", float(np.mean(list(BASELINE.values()))), "hardcoded (Phase 1)")
per_class["baseline_DT"] = BASELINE


# --------------------------------------------------------------------------- #
# 2. classical models: re-evaluate on non-hidden to recover per-class tables
# --------------------------------------------------------------------------- #
def recompute_classical():
    """Load best_rf / best_xgb / best_model and re-evaluate on non-hidden.
    evaluate_model tunes thresholds on local-val internally, then scores the
    chosen eval_split -- safe for the non-hidden final estimate."""
    try:
        from challenge_data import list_split_files
        from sed_pipeline import evaluate_model
        from post_processing import generate_predictions_smooth, _collect_smoothed
        from sed_pipeline import make_sklearn_score_fn, collect_whole_second, \
            tune_thresholds, evaluate_sed
    except Exception as e:
        print(f"  [classical] cannot import pipeline ({e}) -- skipping recompute")
        return
    files = list_split_files()
    ann_df = pd.read_csv(os.path.join(_val_dir(), "annotations.csv"))
    LABEL_CFG = {"overlap_thresh": 0.0, "vote": "majority"}

    def smoothed_per_class(saved, window, mode, thr):
        """Use the stored post-processing thresholds + window to eval non-hidden."""
        sf = make_sklearn_score_fn(saved["model"])
        if thr is None:                       # fall back: re-tune on smoothed local-val
            S, Y = _collect_smoothed(files["local_val"], sf, window, mode, LABEL_CFG)
            thr = tune_thresholds(S, Y)
        pred = generate_predictions_smooth(files["non_hidden"], sf, thr, window, mode)
        return evaluate_sed(pred, files["non_hidden"], ann_df)

    for name, path in [("RF", "best_rf.joblib"), ("XGB", "best_xgb.joblib"),
                       ("XGB_smooth", "best_model.joblib")]:
        try:
            saved = joblib.load(path)
            if name == "XGB_smooth":
                # real post-processing config lives in the pp_* keys (not best_window)
                window = int(saved.get("pp_window") or saved.get("best_window") or 3)
                mode = saved.get("pp_mode") or saved.get("smooth_mode") or "mean"
                thr = saved.get("pp_thresholds")
                macro, per = smoothed_per_class(saved, window, mode, thr)
                src = f"{path} (smoothed {mode} W={window}, non-hidden)"
            else:
                res = evaluate_model(saved["model"], files, ann_df, LABEL_CFG,
                                     eval_split="non_hidden")
                macro, per = res["macro_f1"], res["per_class"]
                src = path + " (re-eval non-hidden)"
            add_overall(name, macro, src)
            per_class[name] = {str(r.annotation): round(float(r.f1), 4)
                               for r in per.itertuples()}
            print(f"  [classical] {name}: macro {macro:.4f}")
        except Exception as e:
            print(f"  [classical] {name} skipped ({e})")
            try:
                saved = joblib.load(path)
                add_overall(name, saved.get("macro_f1_pp_non_hidden")
                            or saved.get("macro_f1_non_hidden"), path)
            except Exception:
                pass


def _val_dir():
    from challenge_data import PATH_VAL
    return PATH_VAL


recompute_classical()


# --------------------------------------------------------------------------- #
# 3. classical HP sweeps (rf_hp_search / xgb_hp_search)
# --------------------------------------------------------------------------- #
for tag, path in [("RF", "rf_hp_search.joblib"), ("XGB", "xgb_hp_search.joblib")]:
    try:
        d = joblib.load(path)
        for sw, rows in d.get("results", {}).items():
            recs = []
            for cfg, f1 in rows:
                rec = dict(cfg) if isinstance(cfg, dict) else {"value": cfg}
                rec["macro_f1"] = round(float(f1), 4)
                recs.append(rec)
            if recs:
                sweeps.append((f"{tag} {sw}", pd.DataFrame(recs)))
    except Exception as e:
        print(f"  [{tag} sweep] skipped ({e})")


# --------------------------------------------------------------------------- #
# 4. CRNN / LSTM sweeps + scores (bonus_results.json, *_hp_search.joblib)
# --------------------------------------------------------------------------- #
for tag, path in [("CRNN", "crnn_hp_search.joblib"), ("LSTM", "lstm_hp_search.joblib")]:
    try:
        d = joblib.load(path)
        for sw, rows in d.get("results", {}).items():
            recs = [{"value": v, "macro_f1": round(float(f1), 4)} for v, f1 in rows]
            sweeps.append((f"{tag} {sw} (local-val)", pd.DataFrame(recs)))
    except Exception as e:
        print(f"  [{tag} sweep] skipped ({e})")

try:
    bj = json.load(open("bonus_results.json"))
    for m in ("crnn", "lstm"):
        node = bj.get(m) or bj.get("models", {}).get(m)
        if isinstance(node, dict) and "macro_f1_non_hidden" in node:
            add_overall(m.upper(), node["macro_f1_non_hidden"], "bonus_results.json")
    # from the comparison block, ONLY take the deep models that live solely here;
    # never XGB/baseline (those are recomputed authoritatively from the joblibs)
    for k, v in (bj.get("comparison") or {}).items():
        if k.lower() in ("crnn", "lstm"):
            add_overall(k.upper(), v, "bonus_results.json")
except Exception as e:
    print(f"  [bonus_results.json] skipped ({e})")


# --------------------------------------------------------------------------- #
# 5. bonus2 (frozen pretrained) -- bonus2_results.json
# --------------------------------------------------------------------------- #
try:
    b2 = json.load(open("bonus2_results.json"))
    add_overall("frame_mn10_frozen", b2["macro_f1_non_hidden"], "bonus2_results.json")
    if "per_class" in b2:
        per_class["frame_mn10_frozen"] = b2["per_class"]
    if "head_sweep" in b2:
        sweeps.append(("frame_mn10 head sweep (local-val)",
                       pd.DataFrame(b2["head_sweep"])))
    if "smoothing_curve" in b2:
        sweeps.append(("frame_mn10 smoothing (local-val)",
                       pd.DataFrame(b2["smoothing_curve"])))
except Exception as e:
    print(f"  [bonus2_results.json] skipped ({e})")


# --------------------------------------------------------------------------- #
# 6. improvements -- improve_results.json (enriched: macro + per_class)
# --------------------------------------------------------------------------- #
try:
    im = json.load(open("improve_results.json"))
    for name, rec in im.items():
        if isinstance(rec, dict):                       # enriched format
            add_overall(name, rec.get("macro_f1"), "improve_results.json")
            if "per_class" in rec:
                per_class[name] = rec["per_class"]
        else:                                           # legacy: macro float only
            add_overall(name, rec, "improve_results.json (macro only)")
except Exception as e:
    print(f"  [improve_results.json] skipped ({e})")


# --------------------------------------------------------------------------- #
# WRITE TABLES
# --------------------------------------------------------------------------- #
# de-dup overall: keep the FIRST occurrence per system name. Sources are added
# in priority order (classical re-evals first, then JSON comparisons), so the
# authoritative recomputed number wins and stale comparison copies are ignored.
best_by = {}
for r in overall:
    if r["system"] not in best_by:
        best_by[r["system"]] = r
ov = pd.DataFrame(sorted(best_by.values(), key=lambda r: -r["macro_f1"]))
ov.to_csv(os.path.join(OUT, "overall_scores.csv"), index=False)

# per-class matrix: rows=classes, cols=systems (ordered by overall macro)
col_order = [s for s in ov["system"] if s in per_class]
pc = pd.DataFrame(index=CLASSES)
for s in col_order:
    pc[s] = [per_class[s].get(c, np.nan) for c in CLASSES]
pc.loc["MACRO"] = [round(np.nanmean(pc[s].values), 4) for s in col_order]
pc.to_csv(os.path.join(OUT, "per_class_f1.csv"))


def df_to_md(df, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:                                   # no tabulate -> manual
        cols = ([df.index.name or ""] if index else []) + list(df.columns)
        out = "| " + " | ".join(map(str, cols)) + " |\n"
        out += "| " + " | ".join("---" for _ in cols) + " |\n"
        for idx, row in df.iterrows():
            cells = ([str(idx)] if index else []) + [
                ("" if pd.isna(v) else str(v)) for v in row]
            out += "| " + " | ".join(cells) + " |\n"
        return out


with open(os.path.join(OUT, "overall_scores.md"), "w") as f:
    f.write("# Overall non-hidden Macro F1\n\n")
    f.write(df_to_md(ov) + "\n")

with open(os.path.join(OUT, "per_class_f1.md"), "w") as f:
    f.write("# Per-class F1 (non-hidden)\n\n")
    f.write(df_to_md(pc, index=True) + "\n")

with open(os.path.join(OUT, "sweeps.md"), "w") as f:
    f.write("# Hyperparameter sweeps & post-processing curves\n\n")
    for title, df in sweeps:
        f.write(f"## {title}\n\n{df_to_md(df)}\n\n")

# --------------------------------------------------------------------------- #
print("\n==================== OVERALL ====================")
print(ov.to_string(index=False))
print("\n==================== PER-CLASS ====================")
print(pc.round(3).to_string())
print(f"\nsaved -> {OUT}/overall_scores.(csv,md), per_class_f1.(csv,md), sweeps.md")
miss = [s for s in ov["system"] if s not in per_class]
if miss:
    print(f"\nper-class missing for: {miss}")
    print("(re-run improve_bonus2.py / bonus2.py once -- now patched to save per-class)")