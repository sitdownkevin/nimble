#!/usr/bin/env python3
"""Pure-Python statistical & mechanistic analyzer for Multimodal Nimble experiments on jev-real-ad-1000."""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / ".cache" / "runs"


def mcnemar_exact_p(rows_a, rows_b):
    map_a = {r["id"]: bool(r["correct"]) for r in rows_a}
    map_b = {r["id"]: bool(r["correct"]) for r in rows_b}
    b01 = b10 = 0
    for rid, ca in map_a.items():
        if rid not in map_b:
            continue
        cb = map_b[rid]
        if ca and not cb:
            b10 += 1
        elif not ca and cb:
            b01 += 1
    n = b01 + b10
    if n == 0:
        return 1.0, b10, b01
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail), b10, b01


def paired_bootstrap_ci(rows_a, rows_b, n_boot=2000, seed=0):
    """Paired bootstrap 95% CI for accuracy difference (A - B) in percentage points."""
    map_a = {r["id"]: (1.0 if r["correct"] else 0.0) for r in rows_a}
    map_b = {r["id"]: (1.0 if r["correct"] else 0.0) for r in rows_b}
    ids = sorted(set(map_a) & set(map_b))
    n = len(ids)
    diffs = [map_a[i] - map_b[i] for i in ids]
    obs = sum(diffs) / n * 100.0
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100.0
        samples.append(s)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return {"diff_pp": round(obs, 2), "ci95_pp": [round(lo, 2), round(hi, 2)], "n": n}


def load_eval_rows(run_name, filename):
    p = RUNS_DIR / run_name / "eval" / filename
    if not p.exists():
        return None
    return json.loads(p.read_text())["rows"]


def main():
    reports = {}
    for rp in sorted(RUNS_DIR.glob("mm*/mm_run_report.json")):
        name = rp.parent.name
        reports[name] = json.loads(rp.read_text())

    hist_path = ROOT / "scripts/multimodal_baseline/baseline_reports/jev_real_ad_1000_summary.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}

    # Load historical per-row test predictions if pulled
    hist_rows = {}
    for hname in ("nimble-0.8b-mm-700", "nimble-0.8b-text-700", "nimble-4b-text-700"):
        hp = RUNS_DIR / "historical_evals" / f"{hname}_test_final.json"
        if hp.exists():
            hist_rows[hname] = json.loads(hp.read_text())["rows"]

    comparisons = {}
    def add_cmp(label, rows_a, rows_b):
        if rows_a is None or rows_b is None:
            return
        p, a_only, b_only = mcnemar_exact_p(rows_a, rows_b)
        boot = paired_bootstrap_ci(rows_a, rows_b)
        comparisons[label] = {
            **boot,
            "mcnemar_p": round(p, 6),
            "wins_a_losses_b": [a_only, b_only],
        }

    # Key paired comparisons on Test (150) and Pooled Val+Test (300)
    pairs_to_test = [
        ("08b_mm_v3_vs_text_v3_ema_test", load_eval_rows("mm01_08b_v3_frozen", "test_final_ema.json"), load_eval_rows("mm02_08b_text_v3_sota", "test_final_ema.json")),
        ("08b_mm_v3_vs_text_v3_ema_pooled300",
         (load_eval_rows("mm01_08b_v3_frozen", "val_final_ema.json") or []) + (load_eval_rows("mm01_08b_v3_frozen", "test_final_ema.json") or []),
         (load_eval_rows("mm02_08b_text_v3_sota", "val_final_ema.json") or []) + (load_eval_rows("mm02_08b_text_v3_sota", "test_final_ema.json") or [])),
        ("08b_rawad_mm_vs_text_ema_test", load_eval_rows("mm08_08b_rawad_mm_frozen", "test_final_ema.json"), load_eval_rows("mm07_08b_rawad_text", "test_final_ema.json")),
        ("08b_rawad_mm_vs_text_ema_pooled300",
         (load_eval_rows("mm08_08b_rawad_mm_frozen", "val_final_ema.json") or []) + (load_eval_rows("mm08_08b_rawad_mm_frozen", "test_final_ema.json") or []),
         (load_eval_rows("mm07_08b_rawad_text", "val_final_ema.json") or []) + (load_eval_rows("mm07_08b_rawad_text", "test_final_ema.json") or [])),
        ("4b_mm_merger_vs_text_v3_tta_test", load_eval_rows("mm11_4b_v3_merger_lora", "test_final_ema_tta5.json"), load_eval_rows("mm13_4b_text_v3_sota", "test_final_ema_tta5.json")),
        ("4b_mm_merger_vs_text_v3_tta_pooled300",
         (load_eval_rows("mm11_4b_v3_merger_lora", "val_final_ema_tta5.json") or []) + (load_eval_rows("mm11_4b_v3_merger_lora", "test_final_ema_tta5.json") or []),
         (load_eval_rows("mm13_4b_text_v3_sota", "val_final_ema_tta5.json") or []) + (load_eval_rows("mm13_4b_text_v3_sota", "test_final_ema_tta5.json") or [])),
        ("4b_mm_lora_all_vs_text_v3_ema_pooled300",
         (load_eval_rows("mm12_4b_v3_lora_all", "val_final_ema.json") or []) + (load_eval_rows("mm12_4b_v3_lora_all", "test_final_ema.json") or []),
         (load_eval_rows("mm13_4b_text_v3_sota", "val_final_ema.json") or []) + (load_eval_rows("mm13_4b_text_v3_sota", "test_final_ema.json") or [])),
    ]
    for lbl, ra, rb in pairs_to_test:
        if ra and rb:
            add_cmp(lbl, ra, rb)

    if "nimble-0.8b-mm-700" in hist_rows:
        add_cmp("08b_mm01_ema_vs_hist_mm_v1_test", load_eval_rows("mm01_08b_v3_frozen", "test_final_ema.json"), hist_rows["nimble-0.8b-mm-700"])
    if "nimble-4b-text-700" in hist_rows:
        add_cmp("4b_mm11_tta_vs_hist_4b_text_v1_test", load_eval_rows("mm11_4b_v3_merger_lora", "test_final_ema_tta5.json"), hist_rows["nimble-4b-text-700"])

    # Build compact table summary
    table = {}
    for name, d in reports.items():
        cfg = d["config"]
        raw = d["raw"]
        ema = d["ema"]
        tta = d["tta"]
        table[name] = {
            "model": cfg["model"],
            "modality": cfg["modality"],
            "state_fields": cfg["state_fields"],
            "vision_mode": cfg["vision_mode"],
            "vision_lr": cfg["vision_lr"],
            "trainable_params": cfg["trainable_params"],
            "visual_trainable_params": cfg["visual_trainable_params"],
            "val_step0_acc": round(d["val_step0"]["accuracy"] * 100, 2),
            "raw_val_acc": round(raw["val_uncalibrated"]["accuracy"] * 100, 2),
            "raw_test_acc": round(raw["test_uncalibrated"]["accuracy"] * 100, 2),
            "raw_test_nll": round(raw["test_uncalibrated"]["nll"], 4),
            "ema_val_acc": round(ema["val_uncalibrated"]["accuracy"] * 100, 2),
            "ema_val_oof_nll": round(ema["val_oof_calibrated"]["nll"], 4),
            "ema_val_oof_brier": round(ema["val_oof_calibrated"]["brier"], 4),
            "ema_val_oof_ece": round(ema["val_oof_calibrated"]["ece"], 4),
            "ema_test_acc": round(ema["test_uncalibrated"]["accuracy"] * 100, 2),
            "ema_test_ooc_acc": round((ema["test_uncalibrated"].get("ooc_accuracy") or 0) * 100, 2),
            "ema_test_raw_nll": round(ema["test_uncalibrated"]["nll"], 4),
            "ema_test_raw_brier": round(ema["test_uncalibrated"]["brier"], 4),
            "ema_test_raw_ece": round(ema["test_uncalibrated"]["ece"], 4),
            "ema_locked_T": round(ema["locked_val_temperature"], 4),
            "ema_test_cal_nll": round(ema["test_locked_calibrated"]["nll"], 4),
            "ema_test_cal_brier": round(ema["test_locked_calibrated"]["brier"], 4),
            "ema_test_cal_ece": round(ema["test_locked_calibrated"]["ece"], 4),
            "tta5_val_acc": round(tta["val_uncalibrated"]["accuracy"] * 100, 2) if tta else None,
            "tta5_test_acc": round(tta["test_uncalibrated"]["accuracy"] * 100, 2) if tta else None,
            "tta5_locked_T": round(tta["locked_val_temperature"], 4) if tta else None,
            "tta5_test_cal_nll": round(tta["test_locked_calibrated"]["nll"], 4) if tta else None,
            "tta5_test_cal_brier": round(tta["test_locked_calibrated"]["brier"], 4) if tta else None,
            "tta5_test_cal_ece": round(tta["test_locked_calibrated"]["ece"], 4) if tta else None,
            "peak_gib": round(d["peak_gpu_allocated_gib"], 2),
            "wall_sec": round(d["training_wall_seconds"], 1),
        }

    out = {
        "historical_baselines": hist,
        "experiments": table,
        "paired_significance": comparisons,
    }
    out_path = RUNS_DIR / "multimodal_sota_full_report.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(table)} experiments, {len(comparisons)} paired tests)")
    for k, v in table.items():
        print(
            f"  {k:30s} | step0={v['val_step0_acc']:5.2f}% | EMA val={v['ema_val_acc']:5.2f}% test={v['ema_test_acc']:5.2f}% "
            f"(cal NLL={v['ema_test_cal_nll']:.4f} Brier={v['ema_test_cal_brier']:.4f} ECE={v['ema_test_cal_ece']:.4f}) "
            f"| TTA5 val={v['tta5_val_acc']:5.2f}% test={v['tta5_test_acc']:5.2f}% (cal NLL={v['tta5_test_cal_nll']:.4f})"
        )
    print("\nPaired tests:")
    for k, v in comparisons.items():
        print(f"  {k}: Δ={v['diff_pp']:+.2f} pp {v['ci95_pp']} (wins/losses={v['wins_a_losses_b']}, McNemar p={v['mcnemar_p']})")


if __name__ == "__main__":
    main()
