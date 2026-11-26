#!/usr/bin/env python3
"""Orchestrate the Multimodal Nimble ablation suite and compile paired statistical comparisons."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE_08B = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
BASE_4B = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")


def mcnemar_exact_p(rows_a, rows_b):
    """Two-sided exact binomial McNemar p-value comparing per-row correctness of two runs."""
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


EXPERIMENTS = {
    # Part I: 0.8B Component & Vision-Tower Ablation (standard full state)
    "mm01_08b_v3_frozen": {
        "desc": "0.8B MM: sandwich_v3 + FP32 head + dyn shuffle + cos 3ep + EMA (vision=frozen)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "frozen", "vision_lr": None,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm02_08b_text_v3_sota": {
        "desc": "0.8B Text-Only SOTA Control: sandwich_v3 + FP32 head + dyn shuffle + cos 3ep + EMA",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "text", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "frozen", "vision_lr": None,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm03_08b_v3_merger_lora": {
        "desc": "0.8B MM: sandwich_v3 + FP32 + EMA + Vision Merger LoRA (vision_lr=5e-5)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "merger_lora", "vision_lr": 5e-5,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm04_08b_v3_merger_full": {
        "desc": "0.8B MM: sandwich_v3 + FP32 + EMA + Vision Merger Full Unfreeze (vision_lr=5e-5)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "merger_full", "vision_lr": 5e-5,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm05_08b_v3_lora_all_vlr3e5": {
        "desc": "0.8B MM: sandwich_v3 + FP32 + EMA + Full Vision+Merger LoRA (decoupled vision_lr=3e-5)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "lora_all", "vision_lr": 3e-5,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm06_08b_v3_lora_all_uniform": {
        "desc": "0.8B MM: sandwich_v3 + FP32 + EMA + Full Vision+Merger LoRA (uniform lr=1e-4)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "lora_all", "vision_lr": None,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    # Part II: Raw-Ad Direct Visual Perception (no pre-extracted VLM OCR/caption)
    "mm07_08b_rawad_text": {
        "desc": "0.8B Raw-Ad Text-Only (title+body only, no VLM OCR/caption)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "text", "state_fields": "raw_ad", "prompt_format": "sandwich_v3",
        "vision_mode": "frozen", "vision_lr": None,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm08_08b_rawad_mm_frozen": {
        "desc": "0.8B Raw-Ad MM (image + title + body, vision=frozen)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "raw_ad", "prompt_format": "sandwich_v3",
        "vision_mode": "frozen", "vision_lr": None,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm09_08b_rawad_mm_lora_all": {
        "desc": "0.8B Raw-Ad MM (image + title + body, vision=lora_all, vision_lr=3e-5)",
        "model": BASE_08B[0], "revision": BASE_08B[1],
        "modality": "mm", "state_fields": "raw_ad", "prompt_format": "sandwich_v3",
        "vision_mode": "lora_all", "vision_lr": 3e-5,
        "lr": 1e-4, "epochs": 3.0, "warmup": 20, "batch": 4, "accum": 2,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    # Part III: Cross-Scale 4B Multimodal SOTA
    "mm10_4b_v3_frozen": {
        "desc": "4B MM SOTA: sandwich_v3 + FP32 + dyn shuffle + cos 2.5ep + EMA (vision=frozen)",
        "model": BASE_4B[0], "revision": BASE_4B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "frozen", "vision_lr": None,
        "lr": 5e-5, "epochs": 2.5, "warmup": 22, "batch": 2, "accum": 4,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm11_4b_v3_merger_lora": {
        "desc": "4B MM SOTA: sandwich_v3 + FP32 + dyn shuffle + cos 2.5ep + EMA (vision=merger_lora, vlr=2e-5)",
        "model": BASE_4B[0], "revision": BASE_4B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "merger_lora", "vision_lr": 2e-5,
        "lr": 5e-5, "epochs": 2.5, "warmup": 22, "batch": 2, "accum": 4,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm12_4b_v3_lora_all": {
        "desc": "4B MM SOTA: sandwich_v3 + FP32 + dyn shuffle + cos 2.5ep + EMA (vision=lora_all, vlr=2e-5)",
        "model": BASE_4B[0], "revision": BASE_4B[1],
        "modality": "mm", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "lora_all", "vision_lr": 2e-5,
        "lr": 5e-5, "epochs": 2.5, "warmup": 22, "batch": 2, "accum": 4,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
    "mm13_4b_text_v3_sota": {
        "desc": "4B Text-Only SOTA Control: sandwich_v3 + FP32 + dyn shuffle + cos 2.5ep + EMA",
        "model": BASE_4B[0], "revision": BASE_4B[1],
        "modality": "text", "state_fields": "all", "prompt_format": "sandwich_v3",
        "vision_mode": "frozen", "vision_lr": None,
        "lr": 5e-5, "epochs": 2.5, "warmup": 22, "batch": 2, "accum": 4,
        "fp32": True, "dyn_shuffle": True, "ema": 0.992, "tta": 5,
    },
}


def run_one(name, cfg, data_dir: Path, runs_dir: Path, project: str):
    out_dir = runs_dir / name
    report_path = out_dir / "mm_run_report.json"
    if report_path.exists():
        print(f"[SKIP] {name} already completed ({report_path})", flush=True)
        return json.loads(report_path.read_text())

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_jev_mm.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--model", cfg["model"],
        "--revision", cfg["revision"],
        "--modality", cfg["modality"],
        "--state-fields", cfg["state_fields"],
        "--prompt-format", cfg["prompt_format"],
        "--vision-mode", cfg["vision_mode"],
        "--learning-rate", str(cfg["lr"]),
        "--lr-scheduler", "cosine",
        "--weight-decay", "0.01",
        "--lora-rank", "16",
        "--batch-size", str(cfg["batch"]),
        "--gradient-accumulation", str(cfg["accum"]),
        "--epochs", str(cfg["epochs"]),
        "--warmup-steps", str(cfg["warmup"]),
        "--save-steps", "200",
        "--eval-steps", "22",
        "--eval-batch-size", "4",
        "--ema-decay", str(cfg["ema"]),
        "--tta-views", str(cfg["tta"]),
        "--seed", "17",
        "--swanlab-project", project,
        "--swanlab-run-name", name,
    ]
    if cfg["vision_lr"] is not None:
        cmd.extend(["--vision-lr", str(cfg["vision_lr"])])
    if cfg["fp32"]:
        cmd.append("--fp32-candidate-head")
    if cfg["dyn_shuffle"]:
        cmd.append("--dynamic-choice-shuffle")

    print(f"\n========== RUNNING {name}: {cfg['desc']} ==========", flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(report_path.read_text())


def summarize_suite(runs_dir: Path):
    summary = {}
    for name, cfg in EXPERIMENTS.items():
        rp = runs_dir / name / "mm_run_report.json"
        if rp.exists():
            data = json.loads(rp.read_text())
            summary[name] = {
                "desc": cfg["desc"],
                "val_step0": data["val_step0"],
                "raw": data["raw"],
                "ema": data["ema"],
                "tta": data["tta"],
                "wall_sec": round(data["training_wall_seconds"], 1),
                "peak_gib": round(data["peak_gpu_allocated_gib"], 2),
                "trainable_params": data["config"]["trainable_params"],
                "visual_trainable_params": data["config"]["visual_trainable_params"],
            }
    out = runs_dir / "mm_ablation_matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved suite summary -> {out} ({len(summary)} runs)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="Subset of experiment keys to run")
    ap.add_argument("--swanlab-project", default="nimble-multimodal-sota")
    args = ap.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    keys = args.only if args.only else list(EXPERIMENTS.keys())
    for k in keys:
        if k not in EXPERIMENTS:
            raise KeyError(f"Unknown experiment {k}; valid: {list(EXPERIMENTS)}")
        run_one(k, EXPERIMENTS[k], args.data_dir, args.runs_dir, args.swanlab_project)
    summarize_suite(args.runs_dir)


if __name__ == "__main__":
    main()
