"""Aggregate all P11 + P12/P13 ACL reviewer-proof ablation results, paired statistical tests,
length-bucketed curves, 2x2 topological interaction effects, 3-seed EMA vs Raw, and Shared-Prefix latency."""

import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path("/data/nvme0/nimble_p9_9b")
PUB = ROOT / "eval_public"
BENCHES = sorted([p.name for p in PUB.iterdir() if p.is_dir() and not p.name.startswith(".")])


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


def softmax_metrics(logits_dict, target, T=1.0):
    keys = list(logits_dict.keys())
    vals = np.array([logits_dict[k] for k in keys], dtype=np.float64) / T
    vals -= vals.max()
    probs = np.exp(vals)
    probs /= probs.sum()
    t_str = "true" if target is True else ("false" if target is False else str(target))
    t_idx = keys.index(t_str) if t_str in keys else [k.lower() for k in keys].index(t_str.lower())
    p_true = float(probs[t_idx])
    p_top = float(probs.max())
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
        lo, hi = bins[i], bins[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(corrs[mask].mean() - confs[mask].mean())
    return float(ece)


def eval_run_metrics(rows, per_bench):
    bench_accs = {}
    kind_bench = {"choice": [], "noul": [], "score": []}
    kind_c, kind_n = {}, {}
    confs, corrs, nlls, briers = [], [], [], []
    for b, b_rows in per_bench.items():
        bc = 0
        k0 = b_rows[0]["type"]
        for r in b_rows:
            k = r["type"]
            logits = r["raw_student"]["fields"]["decision"]["logits"]
            target = r["reference"]["target"]
            ok, p_top, nll, brier = softmax_metrics(logits, target, T=1.0)
            bc += int(ok)
            confs.append(p_top)
            corrs.append(int(ok))
            nlls.append(nll)
            briers.append(brier)
            kind_c[k] = kind_c.get(k, 0) + int(ok)
            kind_n[k] = kind_n.get(k, 0) + 1
        acc_b = bc / len(b_rows) * 100.0
        bench_accs[b] = acc_b
        if k0 in kind_bench:
            kind_bench[k0].append(acc_b)
    macro = float(np.mean(list(bench_accs.values())))
    micro = float(np.mean(corrs) * 100.0)
    en_de = bench_accs.get("massive-de-DE", float("nan")) - bench_accs.get("massive-en-US", float("nan"))
    return {
        "macro": round(macro, 2),
        "micro": round(micro, 2),
        "choice_macro": round(float(np.mean(kind_bench["choice"])), 2),
        "bool_macro": round(float(np.mean(kind_bench["noul"])), 2),
        "score_macro": round(float(np.mean(kind_bench["score"])), 2),
        "choice_micro": round(kind_c.get("choice", 0) / max(kind_n.get("choice", 1), 1) * 100.0, 2),
        "bool_micro": round(kind_c.get("noul", 0) / max(kind_n.get("noul", 1), 1) * 100.0, 2),
        "score_micro": round(kind_c.get("score", 0) / max(kind_n.get("score", 1), 1) * 100.0, 2),
        "en_de_gap": round(en_de, 2),
        "nll": round(float(np.mean(nlls)), 4),
        "brier": round(float(np.mean(briers)), 4),
        "ece": round(compute_ece(confs, corrs, n_bins=10), 4),
        "ece15": round(compute_ece(confs, corrs, n_bins=15), 4),
        "per_bench": {k: round(v, 2) for k, v in bench_accs.items()},
        "corrs": np.array(corrs, dtype=np.int8),
    }


def load_324_summary(run_dir):
    if not run_dir:
        return None
    p = ROOT / "runs" / run_dir / "run_report.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out = {}
    for split in ("before", "after", "after_ema", "after_tta"):
        s = d.get(split)
        if not s:
            continue
        out[split] = {
            "acc": round(s["all"]["accuracy"] * 100, 2),
            "pairs": round(s.get("pairs", {}).get("accuracy", 0.0) * 100, 2),
            "choice": round(s.get("choice", {}).get("accuracy", 0.0) * 100, 2),
            "bool": round(s.get("noul", {}).get("accuracy", 0.0) * 100, 2),
            "score": round(s.get("score", {}).get("accuracy", 0.0) * 100, 2),
            "nll": round(s["all"].get("nll", 0.0), 4),
            "brier": round(s["all"].get("brier", 0.0), 4),
            "ece": round(s["all"].get("ece", 0.0), 4),
        }
    out["training_wall_seconds"] = round(d.get("training_wall_seconds", 0.0), 2)
    out["peak_gpu_allocated_gib"] = round(d.get("peak_gpu_allocated_gib", 0.0), 2)
    return out


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
    return {
        "delta_micro_pp": round(mean_diff, 2),
        "b_only": b_only,
        "a_only": a_only,
        "mcnemar_p": p_mcnemar,
        "ci95": [round(ci_lo, 2), round(ci_hi, 2)],
    }


def length_bucket_compare(rows_v1, rows_v3, label):
    buckets = [("<256", 0, 256), ("256-512", 256, 512), ("512-1024", 512, 1024), (">=1024", 1024, 10**9)]
    out = []
    print(f"\n=== LENGTH-BUCKETED CURVE: {label} ===")
    for bname, lo, hi in buckets:
        idx = [i for i, r in enumerate(rows_v1) if lo <= r["raw_student"]["fields"]["decision"]["prompt_token_count"] < hi]
        if not idx:
            continue
        c1 = np.array([int(rows_v1[i]["student"]["correct"]) for i in idx])
        c3 = np.array([int(rows_v3[i]["student"]["correct"]) for i in idx])
        st = paired_stats(c1, c3, n_boot=5000)
        a1, a3 = round(float(c1.mean() * 100), 2), round(float(c3.mean() * 100), 2)
        print(f"{bname:10s} (n={len(idx):4d}): v1={a1:6.2f}% -> v3={a3:6.2f}% ({a3-a1:+6.2f} pp, p={st['mcnemar_p']:.4g})")
        out.append({"bucket": bname, "n": len(idx), "v1_acc": a1, "v3_acc": a3, "delta_pp": round(a3 - a1, 2), "mcnemar_p": st["mcnemar_p"]})
    return out


def main():
    runs_spec = [
        # Zero-Shot 9B 2x2 Topology
        ("ZS 9B v1", "zs-9b-v1", ""),
        ("ZS 9B context_first_clean", "zs-9b-context-first-clean", ""),
        ("ZS 9B schema_first_clean", "zs-9b-schema-first-clean", ""),
        ("ZS 9B sandwich_v2", "zs-9b-sandwich-v2", ""),
        ("ZS 9B sandwich_v3", "zs-9b-v3", ""),
        # Fine-Tuned 9B Main & 3-Seed Raw vs EMA
        ("Exp-037 9B Official v1", "9b-official-v1-exp037", "p9-9b-official-v1-repro"),
        ("Exp-025 9B Main s17 Raw", "9b-sota-1ep-exp025", "p9-9b-sota-1ep-budget"),
        ("Exp-025 9B Main s17 EMA", "9b-sota-1ep-exp025-ema", "p9-9b-sota-1ep-budget"),
        ("Exp-038 9B Main s42 Raw", "9b-main-seed42-exp038", "p9-9b-main-seed42"),
        ("Exp-038 9B Main s42 EMA", "9b-main-seed42-exp038-ema", "p9-9b-main-seed42"),
        ("Exp-039 9B Main s100 Raw", "9b-main-seed100-exp039", "p9-9b-main-seed100"),
        ("Exp-039 9B Main s100 EMA", "9b-main-seed100-exp039-ema", "p9-9b-main-seed100"),
        # Fine-Tuned 9B 2x2 Prompt Topology
        ("Exp-047 9B FT context_first_clean Raw", "9b-ft-context-first-clean-exp047", "acl-9b-ft-context-first-clean"),
        ("Exp-047 9B FT context_first_clean EMA", "9b-ft-context-first-clean-exp047-ema", "acl-9b-ft-context-first-clean"),
        ("Exp-048 9B FT schema_first_clean Raw", "9b-ft-schema-first-clean-exp048", "acl-9b-ft-schema-first-clean"),
        ("Exp-048 9B FT schema_first_clean EMA", "9b-ft-schema-first-clean-exp048-ema", "acl-9b-ft-schema-first-clean"),
        # Complete 1:1 LOO Ablations on 9B
        ("Exp-042 9B LOO -CleanContext", "9b-loo-noclean-exp042", "acl-9b-loo-noclean"),
        ("Exp-028 9B LOO -PairFlip", "9b-sota-1ep-offclean-exp028", "p9-9b-sota-1ep-official-clean"),
        ("Exp-043 9B LOO -PairBatch", "9b-loo-nopairbatch-exp043", "acl-9b-loo-nopairbatch"),
        ("Exp-044 9B LOO -OrderPres", "9b-loo-noorder-exp044", "acl-9b-loo-noorder"),
        ("Exp-049 9B LOO -FP32Head (BF16 inf)", "9b-loo-nofp32head-bf16inf-exp049", "acl-9b-loo-nofp32head"),
        ("Exp-049 9B LOO -FP32Head (FP32 inf)", "9b-loo-nofp32head-exp049", "acl-9b-loo-nofp32head"),
        ("Exp-050 9B LOO -ChoiceShuffle", "9b-loo-noshuffle-exp050", "acl-9b-loo-noshuffle"),
        ("Exp-051a 9B bs=4x2 LengthBucket", "9b-bs4-lenbucket-exp051a", "acl-9b-bs4-lenbucket"),
        ("Exp-051b 9B bs=4x2 Random", "9b-bs4-random-exp051b", "acl-9b-bs4-random"),
        # Cross-Architecture Pure Softmax Controls (Qwen2.5-7B & Same-Gen Qwen3-8B)
        ("ZS Qwen2.5-7B v1", "zs-qwen25-7b-v1", ""),
        ("ZS Qwen2.5-7B v3", "zs-qwen25-7b-v3", ""),
        ("Exp-040 Qwen2.5-7B v1", "qwen25-7b-v1-exp040", "acl-qwen25-7b-v1"),
        ("Exp-041 Qwen2.5-7B v3+P6", "qwen25-7b-v3p6-exp041", "acl-qwen25-7b-v3p6"),
        ("ZS Qwen3-8B v1", "zs-qwen3-8b-v1", ""),
        ("ZS Qwen3-8B v3", "zs-qwen3-8b-v3", ""),
        ("Exp-052 Qwen3-8B v1", "qwen3-8b-v1-exp052", "acl-qwen3-8b-v1"),
        ("Exp-053 Qwen3-8B v3+P6 Raw", "qwen3-8b-v3p6-exp053", "acl-qwen3-8b-v3p6"),
        ("Exp-053 Qwen3-8B v3+P6 EMA", "qwen3-8b-v3p6-exp053-ema", "acl-qwen3-8b-v3p6"),
    ]

    loaded = {}
    report = {"public_3880": {}, "holdout_324": {}, "paired_tests": {}, "length_buckets": {}}
    print(f"{'Label':38s} | {'324':>6s} | {'Macro':>6s} | {'Micro':>6s} | {'Cho(M)':>6s} | {'Boo(M)':>6s} | {'Sco(M)':>6s} | {'NLL':>6s} | {'Brier':>6s} | {'ECE':>6s}")
    print("-" * 118)
    for lbl, rn, rdir in runs_spec:
        rows, pb = load_public_rows(rn)
        s324 = load_324_summary(rdir)
        if s324:
            report["holdout_324"][lbl] = s324
        if rows is None:
            print(f"[MISSING] {lbl} ({rn})")
            continue
        m = eval_run_metrics(rows, pb)
        loaded[lbl] = (rows, pb, m, s324)
        report["public_3880"][lbl] = {k: v for k, v in m.items() if k != "corrs"}
        a324 = f"{s324['after_ema']['acc']:.2f}" if (s324 and lbl.endswith("EMA") and "after_ema" in s324) else (f"{s324['after']['acc']:.2f}" if (s324 and "after" in s324) else "  —  ")
        print(f"{lbl:38s} | {a324:>6s} | {m['macro']:6.2f} | {m['micro']:6.2f} | {m['choice_macro']:6.2f} | {m['bool_macro']:6.2f} | {m['score_macro']:6.2f} | {m['nll']:6.4f} | {m['brier']:6.4f} | {m['ece']:6.4f}")

    # Paired tests
    pairs = [
        ("ZS 9B v1", "ZS 9B sandwich_v3"),
        ("ZS 9B context_first_clean", "ZS 9B sandwich_v3"),
        ("ZS 9B schema_first_clean", "ZS 9B sandwich_v3"),
        ("ZS 9B sandwich_v2", "ZS 9B sandwich_v3"),
        ("Exp-047 9B FT context_first_clean Raw", "Exp-025 9B Main s17 Raw"),
        ("Exp-048 9B FT schema_first_clean Raw", "Exp-025 9B Main s17 Raw"),
        ("Exp-049 9B LOO -FP32Head (BF16 inf)", "Exp-025 9B Main s17 Raw"),
        ("Exp-050 9B LOO -ChoiceShuffle", "Exp-025 9B Main s17 Raw"),
        ("Exp-051b 9B bs=4x2 Random", "Exp-051a 9B bs=4x2 LengthBucket"),
        ("ZS Qwen3-8B v1", "ZS Qwen3-8B v3"),
        ("Exp-052 Qwen3-8B v1", "Exp-053 Qwen3-8B v3+P6 Raw"),
        ("Exp-040 Qwen2.5-7B v1", "Exp-041 Qwen2.5-7B v3+P6"),
        ("Exp-037 9B Official v1", "Exp-025 9B Main s17 Raw"),
    ]
    print("\n=== PAIRED MCNEMAR & 95% BOOTSTRAP CI (3,880 HUMAN BENCHMARKS) ===")
    for a_lbl, b_lbl in pairs:
        if a_lbl in loaded and b_lbl in loaded:
            st = paired_stats(loaded[a_lbl][2]["corrs"], loaded[b_lbl][2]["corrs"])
            d_macro = round(loaded[b_lbl][2]["macro"] - loaded[a_lbl][2]["macro"], 2)
            st["delta_macro_pp"] = d_macro
            key = f"{b_lbl} vs {a_lbl}"
            report["paired_tests"][key] = st
            print(f"{key:64s} | dMacro={d_macro:+6.2f}pp | dMicro={st['delta_micro_pp']:+6.2f}pp {st['ci95']} | +:{st['b_only']} -:{st['a_only']} | p={st['mcnemar_p']:.4g}")

    # Length buckets across all 3 backbones (Zero-Shot & Fine-Tuned)
    for k, a_lbl, b_lbl in [
        ("zs_qwen35_9b", "ZS 9B v1", "ZS 9B sandwich_v3"),
        ("ft_qwen35_9b", "Exp-037 9B Official v1", "Exp-025 9B Main s17 Raw"),
        ("zs_qwen3_8b", "ZS Qwen3-8B v1", "ZS Qwen3-8B v3"),
        ("ft_qwen3_8b", "Exp-052 Qwen3-8B v1", "Exp-053 Qwen3-8B v3+P6 Raw"),
        ("zs_qwen25_7b", "ZS Qwen2.5-7B v1", "ZS Qwen2.5-7B v3"),
        ("ft_qwen25_7b", "Exp-040 Qwen2.5-7B v1", "Exp-041 Qwen2.5-7B v3+P6"),
    ]:
        if a_lbl in loaded and b_lbl in loaded:
            report["length_buckets"][k] = length_bucket_compare(loaded[a_lbl][0], loaded[b_lbl][0], f"{k}: {a_lbl} -> {b_lbl}")

    # Attach shared prefix & zs_324_2x2 if available
    sp_path = ROOT / "acl_analysis" / "shared_prefix_report.json"
    if sp_path.exists():
        report["shared_prefix"] = json.loads(sp_path.read_text())
    z324_path = ROOT / "acl_analysis" / "zs_324_2x2_topology.json"
    if z324_path.exists():
        report["zs_324_2x2"] = json.loads(z324_path.read_text())

    out_path = ROOT / "acl_analysis" / "acl_p12_ablation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
