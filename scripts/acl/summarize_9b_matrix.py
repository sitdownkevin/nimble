import json, math
from pathlib import Path
import numpy as np

root = Path("/data/nvme0/nimble_p9_9b")

SUBSETS = [
    ("aegis2", "noul"), ("boolq", "noul"), ("civil_comments", "noul"),
    ("helpsteer2", "score"), ("massive-de-DE", "choice"), ("massive-en-US", "choice"),
    ("multinli", "choice"), ("paws", "noul"), ("pubmedqa", "choice"),
    ("squad2", "noul"), ("summeval-consistency", "score"),
    ("summeval-relevance", "score"), ("vitaminc-dev", "choice"),
]

RUNS = [
    ("Exp-023 (2ep cos, 5e-5, w45, f0.35) Raw", "9b-sota-lr5e5-2ep-exp023", "p9-9b-sota-lr5e5-2ep", "raw"),
    ("Exp-023 (2ep cos, 5e-5, w45, f0.35) EMA", "9b-sota-lr5e5-2ep-exp023-ema", "p9-9b-sota-lr5e5-2ep", "ema"),
    ("Exp-024 (2ep cos, 3e-5, w45, f0.35) Raw", "9b-sota-lr3e5-2ep-exp024", "p9-9b-sota-lr3e5-2ep", "raw"),
    ("Exp-024 (2ep cos, 3e-5, w45, f0.35) EMA", "9b-sota-lr3e5-2ep-exp024-ema", "p9-9b-sota-lr3e5-2ep", "ema"),
    ("Exp-025 (1ep lin, w101, f0.35) Raw",      "9b-sota-1ep-exp025", "p9-9b-sota-1ep-budget", "raw"),
    ("Exp-025 (1ep lin, w101, f0.35) EMA",      "9b-sota-1ep-exp025-ema", "p9-9b-sota-1ep-budget", "ema"),
    ("Exp-026 (1ep cos, w45, f0.15) Raw",       "9b-sota-1ep-cos-exp026", "p9-9b-sota-1ep-cosine-sweetspot", "raw"),
    ("Exp-026 (1ep cos, w45, f0.15) EMA",       "9b-sota-1ep-cos-exp026-ema", "p9-9b-sota-1ep-cosine-sweetspot", "ema"),
    ("Exp-026 (0.5ep ckpt-168)",                "9b-sota-halfep-exp026-ckpt168", "p9-9b-sota-1ep-cosine-sweetspot", "raw"),
    ("Exp-027 (1ep cos, w45, f0.00) Raw",       "9b-sota-1ep-clean-exp027", "p9-9b-sota-1ep-cosine-clean", "raw"),
    ("Exp-027 (1ep cos, w45, f0.00) EMA",       "9b-sota-1ep-clean-exp027-ema", "p9-9b-sota-1ep-cosine-clean", "ema"),
    ("Exp-028 (1ep lin, w101, f0.00) Raw",      "9b-sota-1ep-offclean-exp028", "p9-9b-sota-1ep-official-clean", "raw"),
    ("Exp-028 (1ep lin, w101, f0.00) EMA",      "9b-sota-1ep-offclean-exp028-ema", "p9-9b-sota-1ep-official-clean", "ema"),
    ("Exp-029 (1ep lin, w101, f0.15) Raw",      "9b-sota-1ep-golden-exp029", "p9-9b-sota-1ep-golden-combo", "raw"),
    ("Exp-029 (1ep lin, w101, f0.15) EMA",      "9b-sota-1ep-golden-exp029-ema", "p9-9b-sota-1ep-golden-combo", "ema"),
    ("Exp-030 (1ep lin, w101, f0.50) Raw",      "9b-sota-1ep-flip050-exp030", "p9-9b-sota-1ep-flip050", "raw"),
    ("Exp-030 (1ep lin, w101, f0.50) EMA",      "9b-sota-1ep-flip050-exp030-ema", "p9-9b-sota-1ep-flip050", "ema"),
    ("Exp-031 Single-Adapter Exact LoRA Soup",  "9b-sota-1ep-exact-soup", "p9-9b-sota-1ep-exact-soup", "raw"),
    ("Exp-032 (1ep cos, w101, f0.35) Raw",      "9b-1ep-w101-cos-f035-exp032", "p9-9b-1ep-w101-cos-f035", "raw"),
    ("Exp-032 (1ep cos, w101, f0.35) EMA",      "9b-1ep-w101-cos-f035-exp032-ema", "p9-9b-1ep-w101-cos-f035", "ema"),
    ("Exp-033 (1ep cos, w101, f0.25) Raw",      "9b-1ep-w101-cos-f025-exp033", "p9-9b-1ep-w101-cos-f025", "raw"),
    ("Exp-033 (1ep cos, w101, f0.25) EMA",      "9b-1ep-w101-cos-f025-exp033-ema", "p9-9b-1ep-w101-cos-f025", "ema"),
    ("Exp-034 (1ep cos, w101, f0.15) Raw",      "9b-1ep-w101-cos-f015-exp034", "p9-9b-1ep-w101-cos-f015", "raw"),
    ("Exp-034 (1ep cos, w101, f0.15) EMA",      "9b-1ep-w101-cos-f015-exp034-ema", "p9-9b-1ep-w101-cos-f015", "ema"),
    ("Exp-035 (1ep lin, w101, CatFlip 0.35) Raw", "9b-1ep-w101-catflip-exp035", "p9-9b-1ep-w101-catflip", "raw"),
    ("Exp-035 (1ep lin, w101, CatFlip 0.35) EMA", "9b-1ep-w101-catflip-exp035-ema", "p9-9b-1ep-w101-catflip", "ema"),
    ("Exp-036 (1ep cos, w101, CatFlip 0.35) Raw", "9b-1ep-w101-cos-catflip-exp036", "p9-9b-1ep-w101-cos-catflip", "raw"),
    ("Exp-036 (1ep cos, w101, CatFlip 0.35) EMA", "9b-1ep-w101-cos-catflip-exp036-ema", "p9-9b-1ep-w101-cos-catflip", "ema"),
]

REF_9B = {"aegis2":81.2,"boolq":86.0,"civil_comments":70.3,"helpsteer2":39.0,"massive-de-DE":83.4,"massive-en-US":86.9,"multinli":85.3,"paws":82.8,"pubmedqa":75.6,"squad2":80.6,"summeval-consistency":75.7,"summeval-relevance":49.2,"vitaminc-dev":76.6}
REF_JEV = {"aegis2":80.4,"boolq":89.7,"civil_comments":81.0,"helpsteer2":34.1,"massive-de-DE":86.9,"massive-en-US":87.4,"multinli":82.9,"paws":89.2,"pubmedqa":77.2,"squad2":82.9,"summeval-consistency":81.2,"summeval-relevance":35.0,"vitaminc-dev":80.1}

def match_target_key(keys, target):
    if isinstance(target, bool):
        t = "true" if target else "false"
        if t in keys: return keys.index(t)
    s = str(target)
    if s in keys: return keys.index(s)
    if s.lower() in keys: return keys.index(s.lower())
    raise KeyError(f"{target!r} not in {keys}")

def eval_run(tag, temps):
    per_sub = {}
    by_kind_acc = {"choice": [], "noul": [], "score": []}
    tot_corr = tot_n = 0
    score_maes = {}
    for sub, kind in SUBSETS:
        rows_p = root / "eval_public" / sub / tag / "rows.jsonl"
        if not rows_p.exists():
            return None
        T = float(temps.get(kind, 1.0))
        sn, sb, sc, sr, maes = [], [], [], [], []
        for line in rows_p.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            probs = r["student"]["probabilities"]
            target = r["reference"]["target"]
            keys = list(probs.keys())
            vals = np.array([max(float(probs[k]), 1e-12) for k in keys], dtype=np.float64)
            logits = np.log(vals) / T
            logits -= logits.max()
            p_cal = np.exp(logits); p_cal /= p_cal.sum()
            idx = match_target_key(keys, target)
            p_true = float(p_cal[idx])
            nll = -math.log(max(p_true, 1e-12))
            onehot = np.zeros_like(p_cal); onehot[idx] = 1.0
            brier = float(np.sum((p_cal - onehot) ** 2))
            pred_idx = int(np.argmax(p_cal))
            conf = float(p_cal[pred_idx])
            corr = 1.0 if pred_idx == idx else 0.0
            sn.append(nll); sb.append(brier); sc.append(conf); sr.append(corr)
            if kind == "score":
                num_keys = np.array([float(k) for k in keys], dtype=np.float64)
                exp_val = float(np.sum(p_cal * num_keys))
                maes.append(abs(exp_val - float(target)))
        sc = np.array(sc); sr = np.array(sr)
        bins = np.linspace(0.0, 1.0, 11)
        sece = 0.0
        for i in range(10):
            m = (sc >= bins[i]) & (sc <= bins[i+1] if i == 9 else sc < bins[i+1])
            if m.any():
                sece += (m.sum() / len(sc)) * abs(sr[m].mean() - sc[m].mean())
        acc = float(sr.mean())
        per_sub[sub] = {"accuracy": acc, "nll": float(np.mean(sn)), "brier": float(np.mean(sb)), "ece": float(sece), "n": len(sr)}
        by_kind_acc[kind].append(acc)
        tot_corr += int(sr.sum()); tot_n += len(sr)
        if kind == "score":
            score_maes[sub] = float(np.mean(maes))
    return {
        "macro_acc": float(np.mean([v["accuracy"] for v in per_sub.values()])),
        "micro_acc": tot_corr / tot_n,
        "choice_macro": float(np.mean(by_kind_acc["choice"])),
        "noul_macro": float(np.mean(by_kind_acc["noul"])),
        "score_macro": float(np.mean(by_kind_acc["score"])),
        "macro_nll": float(np.mean([v["nll"] for v in per_sub.values()])),
        "macro_brier": float(np.mean([v["brier"] for v in per_sub.values()])),
        "macro_ece": float(np.mean([v["ece"] for v in per_sub.values()])),
        "en_to_de": (per_sub["massive-de-DE"]["accuracy"] - per_sub["massive-en-US"]["accuracy"]) * 100.0,
        "score_maes": score_maes,
        "per_subset": per_sub,
    }

report = {}
print("=== 9B FULL #3 MATRIX ON 324 HOLDOUT + 13 REAL HUMAN PUBLIC BENCHMARKS (3,880 ROWS) ===")
print(f"{'Run':<40} | {'324Acc':>6} | {'Macro':>6} | {'Micro':>6} | {'Ch/No/Sc':>17} | {'Raw NLL/Br/ECE':>17} | {'Cal NLL/Br/ECE':>17} | {'EN->DE':>6} | HS2/SC/SR MAE")
print("-" * 165)

for label, tag, run_dir, variant in RUNS:
    cal_p = root / "runs" / run_dir / "calibration_report.json"
    temps = {"choice": 1.0, "noul": 1.0, "score": 1.0}
    acc324 = float("nan")
    lodo324 = {}
    if cal_p.exists():
        cal = json.loads(cal_p.read_text())
        if variant in cal:
            temps = cal[variant]["global_fit_temperatures"]
            acc324 = cal[variant]["uncalibrated_summary"]["all"]["accuracy"] * 100
            lodo324 = {
                "raw_acc": acc324,
                "pairs": cal[variant]["uncalibrated_summary"]["pairs"]["accuracy"] * 100,
                "raw_nll": cal[variant]["uncalibrated_summary"]["all"]["nll"],
                "raw_brier": cal[variant]["uncalibrated_summary"]["all"]["brier"],
                "raw_ece": cal[variant]["uncalibrated_summary"]["all"]["ece"],
                "lodo_nll": cal[variant]["lodo_calibrated_summary"]["all"]["nll"],
                "lodo_brier": cal[variant]["lodo_calibrated_summary"]["all"]["brier"],
                "lodo_ece": cal[variant]["lodo_calibrated_summary"]["all"]["ece"],
            }
    raw_res = eval_run(tag, {"choice": 1.0, "noul": 1.0, "score": 1.0})
    if raw_res is None:
        continue
    cal_res = eval_run(tag, temps)
    report[tag] = {"label": label, "run_dir": run_dir, "variant": variant, "temps": temps, "holdout324": lodo324, "public_raw": raw_res, "public_locked_temp": cal_res}
    cns = f"{raw_res['choice_macro']*100:4.1f}/{raw_res['noul_macro']*100:4.1f}/{raw_res['score_macro']*100:4.1f}"
    r_cal = f"{raw_res['macro_nll']:5.3f}/{raw_res['macro_brier']:5.3f}/{raw_res['macro_ece']:5.3f}"
    l_cal = f"{cal_res['macro_nll']:5.3f}/{cal_res['macro_brier']:5.3f}/{cal_res['macro_ece']:5.3f}"
    maes = raw_res["score_maes"]
    mae_s = f"{maes['helpsteer2']:.3f}/{maes['summeval-consistency']:.3f}/{maes['summeval-relevance']:.3f}"
    print(f"{label:<40} | {acc324:5.2f}% | {raw_res['macro_acc']*100:5.2f}% | {raw_res['micro_acc']*100:5.2f}% | {cns:>17} | {r_cal:>17} | {l_cal:>17} | {raw_res['en_to_de']:+5.2f} | {mae_s}")

print(f"{'(Ref) Bespoke-Nimble-9B':<40} | 90.12% | 74.80% | 75.90% |  81.6/ 80.2/ 54.6 |   ---/0.399/0.139 |   ---/0.399/0.139 | -3.50 | 0.962/0.569/0.698")
print(f"{'(Ref) Jev 1.13.0 (Closed)':<40} | 93.21% | 76.00% | 77.30% |  82.9/ 84.6/ 50.1 |   ---/0.347/0.092 |   ---/0.347/0.092 | -0.50 | 0.967/0.468/0.770")

out_p = root / "eval_public/public_9b_p3_matrix_report.json"
out_p.write_text(json.dumps(report, indent=2))
print(f"\nSaved {out_p}")
