import json, math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from scipy.optimize import minimize_scalar

ROOT = Path("/data/nvme0/nimble_p9_9b")
PUB = ROOT / "eval_public"
BENCHES = sorted([p.name for p in PUB.iterdir() if p.is_dir() and not p.name.startswith(".")])

# Zero-leakage pre-locked temperatures from 0.8B 5-Fold OOF (zero holdout/public leakage)
OOF_TEMPS_08B = {"all": 4.95, "choice": 4.6203, "noul": 6.0613, "score": 4.5590}

def load_public_rows(run_name):
    rows = []
    per_bench = {}
    for b in BENCHES:
        rp = PUB / b / run_name / "rows.jsonl"
        if not rp.exists():
            return None, None
        b_rows = [json.loads(line) for line in rp.read_text().splitlines() if line.strip()]
        per_bench[b] = b_rows
        for r in b_rows:
            r["_bench"] = b
            rows.append(r)
    return rows, per_bench

def softmax_temp(logits_dict, target, T):
    keys = list(logits_dict.keys())
    vals = np.array([logits_dict[k] for k in keys], dtype=np.float64) / T
    vals -= vals.max()
    probs = np.exp(vals)
    probs /= probs.sum()
    t_str = "true" if target is True else ("false" if target is False else str(target))
    t_idx = keys.index(t_str) if t_str in keys else [k.lower() for k in keys].index(t_str.lower())
    p_true = float(probs[t_idx])
    p_top = float(probs.max())
    pred = keys[int(np.argmax(probs))]
    nll = -math.log(max(p_true, 1e-12))
    one_hot = np.zeros_like(probs)
    one_hot[t_idx] = 1.0
    brier = float(np.sum((probs - one_hot) ** 2))
    return int(np.argmax(probs)) == t_idx, p_top, nll, brier

def compute_ece(confs, corrs, n_bins=15):
    confs = np.asarray(confs, dtype=np.float64)
    corrs = np.asarray(corrs, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(corrs[mask].mean() - confs[mask].mean())
    return float(ece)

def fit_temps_on_324(run_dir):
    """Fit Global 1-T and 3-Kind T strictly on synthetic 324 Holdout (after.json), zero leakage to 3,880 public benchmarks."""
    p = ROOT / "runs" / run_dir / "after.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    preds = d.get("predictions") or d.get("rows")
    if not preds:
        return None
    by_kind = {"all": [], "choice": [], "noul": [], "score": []}
    for r in preds:
        logits = r.get("logits")
        target = r.get("target")
        kind = r.get("kind", "choice")
        if not isinstance(logits, dict) or target is None:
            continue
        keys = list(logits.keys())
        t_str = "true" if target is True else ("false" if target is False else str(target))
        t_idx = keys.index(t_str) if t_str in keys else [k.lower() for k in keys].index(t_str.lower())
        arr = np.asarray([logits[k] for k in keys], dtype=np.float64)
        by_kind["all"].append((arr, int(t_idx)))
        if kind in by_kind:
            by_kind[kind].append((arr, int(t_idx)))
    if not by_kind["all"]:
        return None
    out = {}
    for k, items in by_kind.items():
        if not items:
            out[k] = 1.0
            continue
        def nll_fn(log_t):
            T = math.exp(log_t)
            tot = 0.0
            for arr, y in items:
                z = arr / T
                z -= z.max()
                tot += -(z[y] - math.log(np.exp(z).sum()))
            return tot / len(items)
        res = minimize_scalar(nll_fn, bounds=(math.log(0.2), math.log(15.0)), method="bounded")
        out[k] = float(math.exp(res.x))
    return out

def eval_run_metrics(rows, per_bench, temps=None):
    bench_accs = {}
    kind_c, kind_n = {}, {}
    confs, corrs, nlls, briers = [], [], [], []
    for b, b_rows in per_bench.items():
        bc = 0
        for r in b_rows:
            k = r["type"]
            T = 1.0 if temps is None else (temps if isinstance(temps, (int, float)) else temps.get(k, 1.0))
            logits = r["raw_student"]["fields"]["decision"]["logits"]
            target = r["reference"]["target"]
            ok, p_top, nll, brier = softmax_temp(logits, target, T)
            bc += int(ok)
            confs.append(p_top); corrs.append(int(ok)); nlls.append(nll); briers.append(brier)
            kind_c[k] = kind_c.get(k, 0) + int(ok)
            kind_n[k] = kind_n.get(k, 0) + 1
        bench_accs[b] = bc / len(b_rows) * 100.0
    macro = float(np.mean(list(bench_accs.values())))
    micro = float(np.mean(corrs) * 100.0)
    en_de = bench_accs.get("massive-de-DE", float("nan")) - bench_accs.get("massive-en-US", float("nan"))
    return {
        "macro": macro, "micro": micro, "en_de_gap": en_de,
        "choice": kind_c.get("choice", 0) / max(kind_n.get("choice", 1), 1) * 100.0,
        "noul": kind_c.get("noul", 0) / max(kind_n.get("noul", 1), 1) * 100.0,
        "score": kind_c.get("score", 0) / max(kind_n.get("score", 1), 1) * 100.0,
        "nll": float(np.mean(nlls)), "brier": float(np.mean(briers)), "ece": compute_ece(confs, corrs),
        "per_bench": bench_accs,
        "corrs": np.array(corrs, dtype=np.int8),
    }

def paired_stats(corrs_a, corrs_b, n_boot=10000, seed=42):
    a = np.asarray(corrs_a, dtype=np.int8)
    b = np.asarray(corrs_b, dtype=np.int8)
    b_only = int(((b == 1) & (a == 0)).sum())
    a_only = int(((b == 0) & (a == 1)).sum())
    disc = b_only + a_only
    p_mcnemar = float(binomtest(b_only, disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0
    diff = (b - a).astype(np.float64) * 100.0
    mean_diff = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return {"delta_pp": mean_diff, "b_only": b_only, "a_only": a_only, "mcnemar_p": p_mcnemar, "ci95": [ci_lo, ci_hi]}

def length_bucket_compare(rows_v1, rows_v3, label):
    buckets = [("<256", 0, 256), ("256-512", 256, 512), ("512-1024", 512, 1024), (">=1024", 1024, 10**9)]
    out = []
    print(f"\n=== LENGTH-BUCKETED ACCURACY CURVE: {label} (3,880 real human rows) ===")
    print(f"{'Bucket (v1 tok)':16s} | {'N':>5s} | {'v1 Acc':>8s} | {'v3 Acc':>8s} | {'Delta':>8s} | {'McNemar p':>10s}")
    for bname, lo, hi in buckets:
        idx = [i for i, r in enumerate(rows_v1) if lo <= r["raw_student"]["fields"]["decision"]["prompt_token_count"] < hi]
        if not idx:
            continue
        c1 = np.array([int(rows_v1[i]["student"]["correct"]) for i in idx])
        c3 = np.array([int(rows_v3[i]["student"]["correct"]) for i in idx])
        st = paired_stats(c1, c3, n_boot=2000)
        a1, a3 = float(c1.mean()*100), float(c3.mean()*100)
        print(f"{bname:16s} | {len(idx):5d} | {a1:7.2f}% | {a3:7.2f}% | {a3-a1:+7.2f}pp | {st['mcnemar_p']:.4g}")
        out.append({"bucket": bname, "n": len(idx), "v1_acc": a1, "v3_acc": a3, "delta_pp": a3 - a1, "mcnemar_p": st["mcnemar_p"]})
    return out

def main():
    report = {}
    # Auto-detect Exp-025 run dir
    run_dirs = [p.name for p in (ROOT / "runs").iterdir() if p.is_dir()]
    r025 = "p9-9b-sota-1ep-budget"
    r028 = "p9-9b-sota-1ep-official-clean"
    candidates = [
        ("Exp-037 Official v1 9B", "9b-official-v1-exp037", "p9-9b-official-v1-repro"),
        ("Exp-025 9B Main s17", "9b-sota-1ep-exp025", r025),
        ("Exp-038 9B Main s42", "9b-main-seed42-exp038", "p9-9b-main-seed42"),
        ("Exp-039 9B Main s100", "9b-main-seed100-exp039", "p9-9b-main-seed100"),
        ("Exp-028 9B -PairFlip", "9b-sota-1ep-offclean-exp028", r028),
        ("Exp-042 9B -CleanCtx", "9b-loo-noclean-exp042", "acl-9b-loo-noclean"),
        ("Exp-043 9B -PairBatch", "9b-loo-nopairbatch-exp043", "acl-9b-loo-nopairbatch"),
        ("Exp-044 9B -OrderPres", "9b-loo-noorder-exp044", "acl-9b-loo-noorder"),
        ("Exp-040 Qwen2.5-7B v1", "qwen25-7b-v1-exp040", "acl-qwen25-7b-v1"),
        ("Exp-041 Qwen2.5-7B v3+P6", "qwen25-7b-v3p6-exp041", "acl-qwen25-7b-v3p6"),
        ("ZS Qwen3.5-9B v1", "zs-9b-v1", ""),
        ("ZS Qwen3.5-9B v3", "zs-9b-v3", ""),
        ("ZS Qwen2.5-7B v1", "zs-qwen25-7b-v1", ""),
        ("ZS Qwen2.5-7B v3", "zs-qwen25-7b-v3", ""),
    ]
    loaded = {}
    print("=== ALL AVAILABLE RUNS (3,880 PUBLIC BENCHMARKS, RAW T=1.0) ===")
    print(f"{'Model':26s} | {'Macro':>7s} | {'Micro':>7s} | {'Choice':>6s} | {'Noul':>6s} | {'Score':>6s} | {'EN->DE':>7s} | {'NLL':>6s} | {'Brier':>6s} | {'ECE':>6s}")
    for lbl, rn, rdir in candidates:
        rows, pb = load_public_rows(rn)
        if rows is None:
            continue
        m = eval_run_metrics(rows, pb, temps=None)
        loaded[lbl] = (rows, pb, m, rdir)
        print(f"{lbl:26s} | {m['macro']:6.2f}% | {m['micro']:6.2f}% | {m['choice']:5.1f}% | {m['noul']:5.1f}% | {m['score']:5.1f}% | {m['en_de_gap']:+6.2f} | {m['nll']:6.3f} | {m['brier']:6.3f} | {m['ece']:6.3f}")
        report[lbl] = {k: v for k, v in m.items() if k != "corrs"}

    print("\n=== PAIRED MCNEMAR & 95% BOOTSTRAP CI (3,880 ROWS) ===")
    pairs_to_test = [
        ("Exp-037 Official v1 9B", "Exp-025 9B Main s17"),
        ("Exp-037 Official v1 9B", "Exp-038 9B Main s42"),
        ("Exp-037 Official v1 9B", "Exp-039 9B Main s100"),
        ("Exp-040 Qwen2.5-7B v1", "Exp-041 Qwen2.5-7B v3+P6"),
        ("Exp-042 9B -CleanCtx", "Exp-025 9B Main s17"),
        ("Exp-028 9B -PairFlip", "Exp-025 9B Main s17"),
        ("Exp-043 9B -PairBatch", "Exp-028 9B -PairFlip"),
        ("Exp-043 9B -PairBatch", "Exp-025 9B Main s17"),
        ("Exp-044 9B -OrderPres", "Exp-025 9B Main s17"),
        ("ZS Qwen3.5-9B v1", "ZS Qwen3.5-9B v3"),
        ("ZS Qwen2.5-7B v1", "ZS Qwen2.5-7B v3"),
    ]
    report["paired_tests"] = {}
    for a_lbl, b_lbl in pairs_to_test:
        if a_lbl in loaded and b_lbl in loaded:
            st = paired_stats(loaded[a_lbl][2]["corrs"], loaded[b_lbl][2]["corrs"])
            key = f"{b_lbl} vs {a_lbl}"
            report["paired_tests"][key] = st
            print(f"{key:50s} | dMicro={st['delta_pp']:+6.2f}pp (95% CI [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}]) | +:{st['b_only']} -:{st['a_only']} | p={st['mcnemar_p']:.4g}")

    print("\n=== FAIR CALIBRATION ABLATION (ZERO-LEAKAGE: Temps fitted ONLY on 324 Holdout, applied identically to BOTH v1 and Ours on 3,880 Human Benchmarks) ===")
    report["calibration_ablation"] = {}
    for lbl in ["Exp-037 Official v1 9B", "Exp-025 9B Main s17", "Exp-038 9B Main s42", "Exp-039 9B Main s100", "Exp-040 Qwen2.5-7B v1", "Exp-041 Qwen2.5-7B v3+P6"]:
        if lbl not in loaded:
            continue
        rows, pb, m_raw, rdir = loaded[lbl]
        temps = fit_temps_on_324(rdir)
        m_oof = eval_run_metrics(rows, pb, temps=OOF_TEMPS_08B)
        if temps:
            m_1t = eval_run_metrics(rows, pb, temps=temps["all"])
            m_3t = eval_run_metrics(rows, pb, temps=temps)
            print(f"{lbl:26s} | Raw(T=1): NLL={m_raw['nll']:.3f} Brier={m_raw['brier']:.3f} ECE={m_raw['ece']:.3f} | 1-T({temps['all']:.2f}): NLL={m_1t['nll']:.3f} Brier={m_1t['brier']:.3f} ECE={m_1t['ece']:.3f} | 3-Kind({temps['choice']:.2f},{temps['noul']:.2f},{temps['score']:.2f}): NLL={m_3t['nll']:.3f} Brier={m_3t['brier']:.3f} ECE={m_3t['ece']:.3f} | 08B-OOF: NLL={m_oof['nll']:.3f} Brier={m_oof['brier']:.3f} ECE={m_oof['ece']:.3f}")
            report["calibration_ablation"][lbl] = {
                "temps_324": temps,
                "raw": {"nll": m_raw["nll"], "brier": m_raw["brier"], "ece": m_raw["ece"]},
                "global_1t_324": {"nll": m_1t["nll"], "brier": m_1t["brier"], "ece": m_1t["ece"]},
                "kind_3t_324": {"nll": m_3t["nll"], "brier": m_3t["brier"], "ece": m_3t["ece"]},
                "oof_08b_3t": {"nll": m_oof["nll"], "brier": m_oof["brier"], "ece": m_oof["ece"]},
            }
        else:
            print(f"{lbl:26s} | Raw(T=1): NLL={m_raw['nll']:.3f} Brier={m_raw['brier']:.3f} ECE={m_raw['ece']:.3f} | 08B-OOF: NLL={m_oof['nll']:.3f} Brier={m_oof['brier']:.3f} ECE={m_oof['ece']:.3f} (no 324 after.json in {rdir})")

    report["length_buckets"] = {}
    if "Exp-037 Official v1 9B" in loaded and "Exp-025 9B Main s17" in loaded:
        report["length_buckets"]["qwen35_9b_hybrid"] = length_bucket_compare(
            loaded["Exp-037 Official v1 9B"][0], loaded["Exp-025 9B Main s17"][0], "Qwen3.5-9B (Hybrid DeltaNet: v1 vs v3+P6)"
        )
    if "Exp-040 Qwen2.5-7B v1" in loaded and "Exp-041 Qwen2.5-7B v3+P6" in loaded:
        report["length_buckets"]["qwen25_7b_softmax"] = length_bucket_compare(
            loaded["Exp-040 Qwen2.5-7B v1"][0], loaded["Exp-041 Qwen2.5-7B v3+P6"][0], "Qwen2.5-7B-Instruct (Pure Softmax: v1 vs v3+P6)"
        )

    if "ZS Qwen3.5-9B v1" in loaded and "ZS Qwen3.5-9B v3" in loaded:
        report["length_buckets"]["zs_qwen35_9b_hybrid"] = length_bucket_compare(
            loaded["ZS Qwen3.5-9B v1"][0], loaded["ZS Qwen3.5-9B v3"][0], "ZERO-SHOT Qwen3.5-9B (Hybrid DeltaNet: v1 vs v3)"
        )
    if "ZS Qwen2.5-7B v1" in loaded and "ZS Qwen2.5-7B v3" in loaded:
        report["length_buckets"]["zs_qwen25_7b_softmax"] = length_bucket_compare(
            loaded["ZS Qwen2.5-7B v1"][0], loaded["ZS Qwen2.5-7B v3"][0], "ZERO-SHOT Qwen2.5-7B-Instruct (Pure Softmax: v1 vs v3)"
        )

    out_path = ROOT / "acl_analysis" / "acl_full_analysis_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
