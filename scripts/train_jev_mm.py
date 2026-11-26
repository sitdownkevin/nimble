#!/usr/bin/env python3
"""Train and evaluate Multimodal Nimble on jev-real-ad-1000 (700 train / 150 val / 150 test).

Supports controlled ablations over:
  - Prompt format: ``v1`` vs ``sandwich_v3`` (with clean de-escaped context)
  - Readout precision: BF16 full-vocab head vs exact ``FP32`` candidate projection head
  - Vision tower adaptation: ``frozen``, ``merger_lora``, ``merger_full``, ``lora_all`` (+ decoupled ``--vision-lr``)
  - Modality & text fields: ``mm`` vs ``text``; ``all`` (with OCR/caption) vs ``raw_ad`` (image + title + body only)
  - Regularization & calibration: dynamic per-epoch choice shuffle, warmup-aware FP32 EMA, multi-view cyclic TTA,
    5-fold OOF validation calibration, and strict zero-leakage validation-locked temperature scaling on ``test``.
"""

import argparse
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.multimodal import (
    VISION_MODES,
    DynamicMMDataset,
    MMCandidateCollator,
    apply_locked_temperature,
    build_mm_runtime_row,
    cyclic_shifts_for,
    encode_mm_one,
    evaluate_kfold_oof_temperature,
    image_for,
    select_lora_targets,
    unlock_vision_merger_parameters,
)
from nimble.training.deltanet_ema import (
    EMACallback,
    ParameterEMA,
    save_extra_trainable_params,
    unlock_deltanet_parameters,
)
from nimble.training.model_loading import load_base
from nimble.training.schema_train import (
    CandidateTrainer,
    DynamicShuffleCallback,
    PeriodicEvalAndSwanLabCallback,
    SaveSchemaContract,
    candidate_logits,
    decision_result,
    evaluate,
    flatten_eval_summary,
    summarize,
)

INSTRUCTIONS = "这条广告属于以下哪个行业类目？"
OOC = "以上都不是"
OOC_DESCRIPTION = "广告所属行业不在以上候选类目中"
RAW_AD_KEYS = ("title", "body")


def ooc_metrics(rows):
    ooc_rows = [r for r in rows if r["target"] == OOC]
    in_rows = [r for r in rows if r["target"] != OOC]
    return {
        "ooc_count": len(ooc_rows),
        "ooc_accuracy": (sum(r["correct"] for r in ooc_rows) / len(ooc_rows)) if ooc_rows else None,
        "in_cat_count": len(in_rows),
        "in_cat_accuracy": (sum(r["correct"] for r in in_rows) / len(in_rows)) if in_rows else None,
    }


def load_jev_split(data_dir: Path, split_file: str, modality: str = "mm", state_fields: str = "all"):
    """Load ``kev_data/<split_file>.jsonl`` into raw multimodal/text items ready for ``encode_mm_one``."""
    items = []
    path = data_dir / f"{split_file}.jsonl"
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        full_state = dict(r["state"])
        image_rel = full_state.pop("image", None)
        if state_fields == "raw_ad":
            state = {k: full_state[k] for k in RAW_AD_KEYS if k in full_state and str(full_state[k]).strip()}
        elif state_fields == "all":
            state = full_state
        else:
            raise ValueError(f"Unsupported state_fields: {state_fields}")

        q = r["questions"]["industry"]
        criteria = {
            opt: (desc if isinstance(desc, str) and desc.strip() else (OOC_DESCRIPTION if opt == OOC else opt))
            for opt, desc in q["criteria"].items()
        }
        if state:
            context, schema = adapt_input({
                "state": state,
                "questions": {"decision": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
            })
        else:
            _, schema = adapt_input({
                "state": {"_placeholder": "1"},
                "questions": {"decision": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
            })
            context = ""

        img = None
        if modality == "mm":
            if not image_rel:
                raise ValueError(f"Missing state.image in {r['id']} from {path}")
            img = image_for(data_dir / image_rel)

        items.append({
            "id": r["id"],
            "family": r["id"],
            "kind": "choice",
            "context": context,
            "schema": schema,
            "image": img,
            "target": q["label"],
        })
    return items


def encode_eval_items(items, tokenizer, prompt_format, max_length):
    rows = []
    for item in items:
        enc = encode_mm_one(
            tokenizer,
            item["context"],
            item["schema"],
            image=item.get("image"),
            prompt_format=prompt_format,
            shuffle_seed=None,
            row_id=item["id"],
            kind=item["kind"],
            max_length=max_length,
        )
        rows.append(build_mm_runtime_row(enc, item["id"], item["family"], item["kind"], item["target"]))
    return rows


def evaluate_mm_cyclic_tta(
    model, raw_items, tokenizer, max_length, collator, prompt_format, base_report, output,
    fp32_candidate_head=False, num_views=5,
):
    """Multi-view cyclic option shift TTA over multimodal/text choice items."""
    model.eval()
    acc_logits = {
        r["id"]: {k: [v - sum(r["logits"].values()) / len(r["logits"])] for k, v in r["logits"].items()}
        for r in base_report["rows"]
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in raw_items:
            n_choices = len(next(iter(item["schema"].values()))["choices"])
            shifts = cyclic_shifts_for(n_choices, num_views)
            for shift in shifts:
                if shift == 0:
                    continue
                enc = encode_mm_one(
                    tokenizer,
                    item["context"],
                    item["schema"],
                    image=item.get("image"),
                    prompt_format=prompt_format,
                    shuffle_seed=None,
                    row_id=item["id"],
                    kind=item["kind"],
                    cyclic_shift=shift,
                    max_length=max_length,
                )
                rt = build_mm_runtime_row(enc, item["id"], item["family"], item["kind"], item["target"])
                batch = {k: v.to("cuda") for k, v in collator([rt]).items()}
                logits = candidate_logits(model, batch, fp32_candidate_head=fp32_candidate_head)[0].cpu()
                res = decision_result(rt, logits)
                mean_l = sum(res["logits"].values()) / len(res["logits"])
                for k, v in res["logits"].items():
                    acc_logits[item["id"]][k].append(v - mean_l)

    rows = []
    for r0 in base_report["rows"]:
        rid = r0["id"]
        keys = list(r0["logits"].keys())
        avg_logits = [sum(acc_logits[rid][k]) / len(acc_logits[rid][k]) for k in keys]
        row_stub = {"choices": keys, "kind": r0["kind"], "labels": keys.index(str(r0["target"]))}
        res = decision_result(row_stub, torch.tensor(avg_logits, dtype=torch.float64))
        rows.append({"id": rid, "family": r0["family"], "kind": r0["kind"], "target": r0["target"], **res})
    report = {"summary": summarize(rows), "rows": rows}
    output.write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    print(json.dumps({"evaluation": output.name, "summary": report["summary"]}, ensure_ascii=False), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Path to jev-real-ad-1000/kev_data")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    ap.add_argument("--modality", default="mm", choices=["mm", "text"])
    ap.add_argument("--state-fields", default="all", choices=["all", "raw_ad"])
    ap.add_argument("--prompt-format", default="sandwich_v3", choices=["v1", "sandwich_v3", "sandwich_v3_rawjson"])
    ap.add_argument("--vision-mode", default="frozen", choices=list(VISION_MODES))
    ap.add_argument("--vision-lr", type=float, default=None, help="Optional separate LR for .visual. parameters")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lr-scheduler", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0, help="Override steps; if 0, computed from epochs")
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=88)
    ap.add_argument("--eval-steps", type=int, default=22)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--fp32-candidate-head", action="store_true")
    ap.add_argument("--dynamic-choice-shuffle", action="store_true")
    ap.add_argument("--unlock-deltanet", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.992)
    ap.add_argument("--tta-views", type=int, default=5, help="Number of cyclic choice views at final eval (0/1 disables)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--swanlab-project", default="nimble-multimodal-sota")
    ap.add_argument("--swanlab-run-name", default=None)
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory {args.output_dir} already exists and is non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0_data = time.monotonic()
    train_split = "train_all_pos_start" if args.modality == "mm" else "train_all"
    val_split = "validation_pos_start" if args.modality == "mm" else "validation"
    test_split = "test_pos_start" if args.modality == "mm" else "test"

    raw_train = load_jev_split(args.data_dir, train_split, modality=args.modality, state_fields=args.state_fields)
    raw_val = load_jev_split(args.data_dir, val_split, modality=args.modality, state_fields=args.state_fields)
    raw_test = load_jev_split(args.data_dir, test_split, modality=args.modality, state_fields=args.state_fields)

    train_ds = DynamicMMDataset(
        raw_train,
        tokenizer,
        max_length=args.max_length,
        base_seed=args.seed,
        prompt_format=args.prompt_format,
        preserve_ordered_kinds=True,
        dynamic_shuffle=args.dynamic_choice_shuffle,
    )
    val_rows = encode_eval_items(raw_val, tokenizer, args.prompt_format, args.max_length)
    test_rows = encode_eval_items(raw_test, tokenizer, args.prompt_format, args.max_length)
    data_prep_sec = time.monotonic() - t0_data

    eff_bs = args.batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(len(train_ds) / eff_bs)
    max_steps = args.max_steps if args.max_steps > 0 else int(round(steps_per_epoch * args.epochs))

    train_lens = [len(r["input_ids"]) for r in train_ds.records]
    vis_lens = [r["image_tokens"] for r in train_ds.records]
    audit = {
        "dataset": "jev-real-ad-1000",
        "modality": args.modality,
        "state_fields": args.state_fields,
        "prompt_format": args.prompt_format,
        "training_rows": len(train_ds),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_mean_tokens": round(sum(train_lens) / len(train_lens), 1),
        "train_max_tokens": max(train_lens),
        "train_mean_image_tokens": round(sum(vis_lens) / len(vis_lens), 1),
        "data_prep_seconds": round(data_prep_sec, 2),
    }
    print(json.dumps({"audit": audit}, ensure_ascii=False), flush=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    base = load_base(args.model, args.revision)
    effective_vision_mode = args.vision_mode if args.modality == "mm" else "frozen"
    if args.modality == "text" and hasattr(base, "model") and hasattr(base.model, "visual"):
        delattr(base.model, "visual")
        torch.cuda.empty_cache()

    targets = select_lora_targets(base, vision_mode=effective_vision_mode)
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    extra_unlocked = []
    extra_scalars = 0
    if effective_vision_mode == "merger_full":
        u_names, u_cnt = unlock_vision_merger_parameters(model)
        extra_unlocked.extend(u_names)
        extra_scalars += u_cnt
        print(f"Unfroze full visual.merger: {len(u_names)} tensors ({u_cnt:,} params)", flush=True)
    if args.unlock_deltanet:
        u_names, u_cnt = unlock_deltanet_parameters(model)
        extra_unlocked.extend(u_names)
        extra_scalars += u_cnt
        print(f"Unfroze DeltaNet/RMSNorm: {len(u_names)} tensors ({u_cnt:,} params)", flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    vis_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ".visual." in n)
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.3f}%, visual trainable={vis_trainable:,})",
        flush=True,
    )

    contract = {
        "task": "multimodal_schema_candidate_classification_v1",
        "model": args.model,
        "revision": args.revision,
        "modality": args.modality,
        "state_fields": args.state_fields,
        "prompt_format": args.prompt_format,
        "vision_mode": effective_vision_mode,
        "vision_lr": args.vision_lr,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "epochs": args.epochs,
        "max_steps": max_steps,
        "warmup_steps": args.warmup_steps,
        "fp32_candidate_head": args.fp32_candidate_head,
        "dynamic_choice_shuffle": args.dynamic_choice_shuffle,
        "unlock_deltanet": args.unlock_deltanet,
        "ema_decay": args.ema_decay,
        "tta_views": args.tta_views,
        "seed": args.seed,
        "trainable_params": trainable_params,
        "visual_trainable_params": vis_trainable,
        "extra_unlocked_scalars": extra_scalars,
        "gpu": torch.cuda.get_device_name(),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
        "data_audit": audit,
    }
    (args.output_dir / "schema_config.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")

    collator = MMCandidateCollator(tokenizer.pad_token_id)

    swanlab_mod = None
    if args.swanlab_project:
        import swanlab as swanlab_mod
        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            swanlab_mod.login(api_key=api_key, save=False)
        swanlab_mod.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_run_name or args.output_dir.name,
            config=contract,
        )

    eval_dir = args.output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    initial_val = evaluate(
        model, val_rows, collator, eval_dir / "val_step_0.json",
        fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=args.eval_batch_size,
    )
    if swanlab_mod is not None:
        swanlab_mod.log(flatten_eval_summary(initial_val["summary"], prefix="val"), step=0)

    save_extra = bool(extra_unlocked)
    callbacks = [SaveSchemaContract(contract, save_extra_params=save_extra, save_hybrid=False)]
    ema = None
    if args.ema_decay > 0:
        ema = ParameterEMA(model, decay=args.ema_decay, warmup_steps=args.warmup_steps)
        callbacks.append(EMACallback(ema))
    if args.dynamic_choice_shuffle:
        callbacks.append(DynamicShuffleCallback(train_ds))
    periodic_cb = PeriodicEvalAndSwanLabCallback(
        val_rows, collator, eval_dir,
        eval_steps=args.eval_steps,
        swanlab_mod=swanlab_mod,
        fp32_candidate_head=args.fp32_candidate_head,
        eval_batch_size=args.eval_batch_size,
    )
    callbacks.append(periodic_cb)

    hf_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        logging_nan_inf_filter=False,
    )

    trainer = CandidateTrainer(
        model=model,
        args=hf_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        fp32_candidate_head=args.fp32_candidate_head,
        vision_lr=args.vision_lr,
    )

    torch.cuda.reset_peak_memory_stats()
    t_train = time.monotonic()
    train_res = trainer.train()
    wall_sec = time.monotonic() - t_train
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final_step = trainer.state.global_step

    # --- Full Dual-Split Final Evaluation (Validation + Test, unpadded bs=1) ---
    val_raw = evaluate(
        model, val_rows, collator, eval_dir / "val_final_raw.json",
        fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=1,
    )
    test_raw = evaluate(
        model, test_rows, collator, eval_dir / "test_final_raw.json",
        fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=1,
    )
    val_cal_raw = evaluate_kfold_oof_temperature(val_raw["rows"], summarize, n_folds=5, seed=args.seed)
    test_cal_raw = apply_locked_temperature(test_raw["rows"], val_cal_raw["locked_temperature"], summarize)

    val_ema = test_ema = val_cal_ema = test_cal_ema = None
    val_tta = test_tta = val_cal_tta = test_cal_tta = None

    if ema is not None:
        with ema.apply_to(model):
            val_ema = evaluate(
                model, val_rows, collator, eval_dir / "val_final_ema.json",
                fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=1,
            )
            test_ema = evaluate(
                model, test_rows, collator, eval_dir / "test_final_ema.json",
                fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=1,
            )
            val_cal_ema = evaluate_kfold_oof_temperature(val_ema["rows"], summarize, n_folds=5, seed=args.seed)
            test_cal_ema = apply_locked_temperature(test_ema["rows"], val_cal_ema["locked_temperature"], summarize)

            if args.tta_views and args.tta_views > 1:
                val_tta = evaluate_mm_cyclic_tta(
                    model, raw_val, tokenizer, args.max_length, collator, args.prompt_format,
                    val_ema, eval_dir / f"val_final_ema_tta{args.tta_views}.json",
                    fp32_candidate_head=args.fp32_candidate_head, num_views=args.tta_views,
                )
                test_tta = evaluate_mm_cyclic_tta(
                    model, raw_test, tokenizer, args.max_length, collator, args.prompt_format,
                    test_ema, eval_dir / f"test_final_ema_tta{args.tta_views}.json",
                    fp32_candidate_head=args.fp32_candidate_head, num_views=args.tta_views,
                )
                val_cal_tta = evaluate_kfold_oof_temperature(val_tta["rows"], summarize, n_folds=5, seed=args.seed)
                test_cal_tta = apply_locked_temperature(test_tta["rows"], val_cal_tta["locked_temperature"], summarize)

            ema_dir = args.output_dir / "ema_adapter"
            ema_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ema_dir))
            tokenizer.save_pretrained(str(ema_dir))
            (ema_dir / "schema_config.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
            (ema_dir / "locked_temperature.json").write_text(
                json.dumps({"locked_val_temperature": val_cal_ema["locked_temperature"]}, indent=2) + "\n"
            )
            if save_extra:
                save_extra_trainable_params(model, ema_dir)
    elif args.tta_views and args.tta_views > 1:
        val_tta = evaluate_mm_cyclic_tta(
            model, raw_val, tokenizer, args.max_length, collator, args.prompt_format,
            val_raw, eval_dir / f"val_final_raw_tta{args.tta_views}.json",
            fp32_candidate_head=args.fp32_candidate_head, num_views=args.tta_views,
        )
        test_tta = evaluate_mm_cyclic_tta(
            model, raw_test, tokenizer, args.max_length, collator, args.prompt_format,
            test_raw, eval_dir / f"test_final_raw_tta{args.tta_views}.json",
            fp32_candidate_head=args.fp32_candidate_head, num_views=args.tta_views,
        )
        val_cal_tta = evaluate_kfold_oof_temperature(val_tta["rows"], summarize, n_folds=5, seed=args.seed)
        test_cal_tta = apply_locked_temperature(test_tta["rows"], val_cal_tta["locked_temperature"], summarize)

    trainer.save_model(str(args.output_dir))
    if save_extra:
        save_extra_trainable_params(model, args.output_dir)
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()

    full_report = {
        "config": contract,
        "val_step0": initial_val["summary"]["all"],
        "raw": {
            "locked_val_temperature": val_cal_raw["locked_temperature"],
            "val_uncalibrated": {**val_raw["summary"]["all"], **ooc_metrics(val_raw["rows"])},
            "val_oof_calibrated": val_cal_raw["oof_calibrated_summary"]["all"],
            "test_uncalibrated": {**test_raw["summary"]["all"], **ooc_metrics(test_raw["rows"])},
            "test_locked_calibrated": test_cal_raw["calibrated_summary"]["all"],
        },
        "ema": (
            {
                "locked_val_temperature": val_cal_ema["locked_temperature"],
                "val_uncalibrated": {**val_ema["summary"]["all"], **ooc_metrics(val_ema["rows"])},
                "val_oof_calibrated": val_cal_ema["oof_calibrated_summary"]["all"],
                "test_uncalibrated": {**test_ema["summary"]["all"], **ooc_metrics(test_ema["rows"])},
                "test_locked_calibrated": test_cal_ema["calibrated_summary"]["all"],
            }
            if val_ema is not None
            else None
        ),
        "tta": (
            {
                "views": args.tta_views,
                "locked_val_temperature": val_cal_tta["locked_temperature"],
                "val_uncalibrated": {**val_tta["summary"]["all"], **ooc_metrics(val_tta["rows"])},
                "val_oof_calibrated": val_cal_tta["oof_calibrated_summary"]["all"],
                "test_uncalibrated": {**test_tta["summary"]["all"], **ooc_metrics(test_tta["rows"])},
                "test_locked_calibrated": test_cal_tta["calibrated_summary"]["all"],
            }
            if val_tta is not None
            else None
        ),
        "training": train_res.metrics,
        "optimizer_steps": final_step,
        "training_wall_seconds": wall_sec,
        "peak_gpu_allocated_gib": peak_gib,
    }
    (args.output_dir / "mm_run_report.json").write_text(
        json.dumps(full_report, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    )

    if swanlab_mod is not None:
        final_logs = {}
        for tag, blk in (("raw", full_report["raw"]), ("ema", full_report["ema"]), ("tta", full_report["tta"])):
            if blk is None:
                continue
            final_logs[f"final/{tag}/locked_T"] = float(blk["locked_val_temperature"])
            for split_key in ("val_uncalibrated", "val_oof_calibrated", "test_uncalibrated", "test_locked_calibrated"):
                for m_key in ("accuracy", "nll", "brier", "ece", "ooc_accuracy", "in_cat_accuracy"):
                    val = blk[split_key].get(m_key)
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        final_logs[f"final/{tag}/{split_key}/{m_key}"] = float(val)
        swanlab_mod.log(final_logs, step=final_step)
        swanlab_mod.finish()

    print(json.dumps(full_report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
