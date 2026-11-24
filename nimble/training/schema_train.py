"""Train and evaluate Qwen LoRA adapters for schema-conditioned decisions."""

import argparse
import hashlib
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput

from nimble.scoring.parallel_schema import MODEL_ID, REVISION, SYSTEM_PROMPT, choice_key, prepare_prompts
from nimble.training.deltanet_ema import (
    EMACallback,
    ParameterEMA,
    evaluate_lodo_temperature_calibration,
    load_extra_trainable_params,
    save_extra_trainable_params,
    unlock_deltanet_parameters,
)
from nimble.training.model_loading import load_base
from nimble.training.pair_contrastive import PairBatchSampler, pairwise_flip_loss
from nimble.training.schema_data import (
    as_scoring, encode_scoring, prepare_cv_data, prepare_data, read_rows, resolve_checkpoint, runtime_record,
)

MAX_CHOICES = 26


class CandidateCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        length = max(len(r["input_ids"]) for r in rows)
        for r in rows:
            n = len(r["candidate_ids"])
            if not 1 <= n <= MAX_CHOICES or len(set(r["candidate_ids"])) != n:
                raise ValueError("Invalid candidate token set")
            if "labels" in r and not 0 <= r["labels"] < n:
                raise ValueError("Gold label points outside the real candidates")
        batch = {
            "input_ids": torch.tensor([[self.pad_id] * (length - len(r["input_ids"])) + r["input_ids"] for r in rows]),
            "attention_mask": torch.tensor([[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
            "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows]),
            "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows]),
        }
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([r["labels"] for r in rows])
        if any(r.get("pair_id") for r in rows):
            pid_map = {}
            gids = []
            for r in rows:
                pid = r.get("pair_id")
                if not pid or r.get("is_aux", False):
                    gids.append(-1)
                else:
                    if pid not in pid_map:
                        pid_map[pid] = len(pid_map)
                    gids.append(pid_map[pid])
            batch["pair_group_ids"] = torch.tensor(gids, dtype=torch.long)
        if any("sample_weight" in r for r in rows):
            batch["sample_weights"] = torch.tensor([float(r.get("sample_weight", 1.0)) for r in rows], dtype=torch.float32)
        return batch


def _unwrap_backbone_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", base)
    backbone = getattr(inner, "language_model", inner)
    head = base.get_output_embeddings()
    if head is None:
        raise RuntimeError("Model has no output embeddings for candidate projection")
    return backbone, head.weight


def candidate_logits(model, inputs, fp32_candidate_head=False):
    if fp32_candidate_head:
        backbone, head_weight = _unwrap_backbone_and_head(model)
        hidden = backbone(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                          use_cache=False).last_hidden_state[:, -1, :]
        cand_rows = head_weight[inputs["candidate_ids"]]
        selected = (hidden.float().unsqueeze(1) * cand_rows.float()).sum(dim=-1)
        return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    selected = logits.gather(1, inputs["candidate_ids"])
    return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)


class CandidateTrainer(Trainer):
    def __init__(self, *args, fp32_candidate_head=False, pair_batch_sampling=False,
                 pair_sampling_mode="length_bucket", flip_loss_weight=0.0, flip_margin=2.0,
                 flip_multiclass_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.fp32_candidate_head = fp32_candidate_head
        self.pair_batch_sampling = pair_batch_sampling
        self.pair_sampling_mode = pair_sampling_mode
        self.flip_loss_weight = flip_loss_weight
        self.flip_margin = flip_margin
        self.flip_multiclass_only = flip_multiclass_only
        self._last_ce_loss = None
        self._last_flip_loss = None

    def get_train_dataloader(self):
        if not getattr(self, "pair_batch_sampling", False):
            return super().get_train_dataloader()
        sampler = PairBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            sampling_mode=getattr(self, "pair_sampling_mode", "length_bucket"),
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = candidate_logits(model, inputs, fp32_candidate_head=getattr(self, "fp32_candidate_head", False))
        per_row_ce = torch.nn.functional.cross_entropy(logits, inputs["labels"], reduction="none")
        weights = inputs.get("sample_weights")
        if weights is not None:
            w = weights.to(per_row_ce.device)
            ce_loss = (per_row_ce * w).mean()
        else:
            ce_loss = per_row_ce.mean()
        flip_w = getattr(self, "flip_loss_weight", 0.0)
        if flip_w > 0 and "pair_group_ids" in inputs:
            f_loss = pairwise_flip_loss(
                logits, inputs["labels"], inputs["pair_group_ids"],
                margin=getattr(self, "flip_margin", 2.0),
                multiclass_only=getattr(self, "flip_multiclass_only", False),
            )
            loss = ce_loss + flip_w * f_loss
            if self is not None:
                self._last_ce_loss = float(ce_loss.detach().item())
                self._last_flip_loss = float(f_loss.detach().item())
        else:
            loss = ce_loss
            if self is not None:
                self._last_ce_loss = float(ce_loss.detach().item())
                self._last_flip_loss = 0.0
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite candidate loss")
        return (loss, SequenceClassifierOutput(loss=loss, logits=logits)) if return_outputs else loss


class DynamicChoiceDataset(torch.utils.data.Dataset):
    """Keeps ordered/boolean rows cached while dynamically reshuffling choice rows per epoch."""

    def __init__(self, records, raw_rows, tokenizer, max_length, base_seed,
                 preserve_ordered_kinds=True, sync_pair_shuffle=True, prompt_format="v1"):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_seed = base_seed
        self.preserve_ordered_kinds = preserve_ordered_kinds
        self.sync_pair_shuffle = sync_pair_shuffle
        self.prompt_format = prompt_format
        self.choice_items = [
            (idx, as_scoring(raw, True), raw)
            for idx, (rec, raw) in enumerate(zip(self.records, raw_rows))
            if rec["kind"] == "choice"
        ]
        self.current_epoch = -1
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if epoch == self.current_epoch:
            return
        self.current_epoch = epoch
        epoch_seed = f"{self.base_seed}:ep{epoch}"
        for idx, scoring_row, raw_row in self.choice_items:
            enc = encode_scoring(
                scoring_row, self.tokenizer, self.max_length,
                shuffle_seed=epoch_seed,
                preserve_ordered_kinds=self.preserve_ordered_kinds,
                sync_pair_shuffle=self.sync_pair_shuffle,
                prompt_format=self.prompt_format,
            )
            self.records[idx] = runtime_record(enc, raw_row)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class DynamicShuffleCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch)) if state.epoch is not None else 0
        self.dataset.set_epoch(epoch)
        return control


def decision_result(row, logits):
    logits = logits[:len(row["choices"])].double()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite candidate logits")
    probs = logits.softmax(-1)
    index = logits.argmax().item()
    gold = row.get("labels")
    choices = row["choices"]
    result = {"prediction": choices[index], "probabilities": {choice_key(k): v for k, v in zip(choices, probs.tolist())},
              "logits": {choice_key(k): v for k, v in zip(choices, logits.tolist())}}
    if gold is not None:
        result.update(correct=index == gold, nll=-logits.log_softmax(-1)[gold].item(),
                      brier=sum((p - int(i == gold)) ** 2 for i, p in enumerate(probs.tolist())))
    if row.get("kind") == "score":
        result["prediction"] = int(choices[index])
        result["expected_score"] = sum(int(v) * p for v, p in zip(choices, probs.tolist()))
        if gold is not None:
            result["score_absolute_error"] = abs(result["expected_score"] - int(choices[gold]))
    if row.get("kind") == "noul":
        result["probability_true"] = result["probabilities"]["true"]
    return result


def expected_calibration_error(rows, bins=10):
    buckets = [[] for _ in range(bins)]
    for r in rows:
        p = max(r["probabilities"].values())
        idx = min(bins - 1, int(p * bins))
        buckets[idx].append((p, 1.0 if r["correct"] else 0.0))
    n = len(rows)
    return sum((len(b) / n) * abs(sum(x[0] for x in b) / len(b) - sum(x[1] for x in b) / len(b)) for b in buckets if b)


def summarize(rows):
    summary = {}
    for kind in ("all", "choice", "noul", "score"):
        selected = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not selected:
            continue
        group = {"count": len(selected), "correct": sum(r["correct"] for r in selected)}
        group["accuracy"] = group["correct"] / len(selected)
        if all("probabilities" in r for r in selected):
            group["ece"] = expected_calibration_error(selected)
        for metric in ("nll", "brier", "score_absolute_error"):
            if all(metric in r for r in selected):
                group[metric] = sum(r[metric] for r in selected) / len(selected)
        summary[kind] = group
    if all(r.get("pair_id") for r in rows):
        pairs = {}
        for r in rows:
            pairs.setdefault(r["pair_id"], []).append(r)
        if pairs and all(len(g) == 2 for g in pairs.values()):
            both = sum(all(x["correct"] for x in g) for g in pairs.values())
            summary["pairs"] = {"count": len(pairs), "both_correct": both, "accuracy": both / len(pairs)}
    return summary


def evaluate(model, data, collator, output, fp32_candidate_head=False, eval_batch_size=1):
    model.eval()
    rows = []
    # Match Trainer's BF16 autocast in standalone evaluation and after reload.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(data), eval_batch_size):
            chunk = data[start:start + eval_batch_size]
            batch = {k: v.to("cuda") for k, v in collator(chunk).items()}
            logits_batch = candidate_logits(model, batch, fp32_candidate_head=fp32_candidate_head).cpu()
            for idx, r in enumerate(chunk):
                result = decision_result(r, logits_batch[idx])
                rows.append({"id": r["id"], "family": r["family"], "pair_id": r.get("pair_id"),
                             "kind": r["kind"], "target": r["target"], **result})
    report = {"summary": summarize(rows), "rows": rows}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"evaluation": output.name, "summary": report["summary"]}), flush=True)
    return report


class SaveSchemaContract(TrainerCallback):
    def __init__(self, contract, save_extra_params=False):
        self.contract = contract
        self.save_extra_params = save_extra_params

    def on_save(self, args, state, control, model=None, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        (ckpt_dir / "schema_config.json").write_text(json.dumps(self.contract, indent=2) + "\n")
        if self.save_extra_params and model is not None:
            save_extra_trainable_params(model, ckpt_dir)


class EpochValidation(TrainerCallback):
    """Measure inner validation after full epochs without consulting outer eval."""
    def __init__(self, validation, collator, output_dir, stop_after_epochs=None,
                 fp32_candidate_head=False, eval_batch_size=1):
        self.validation, self.collator = validation, collator
        self.output_dir = Path(output_dir)
        self.stop_after_epochs = stop_after_epochs
        self.fp32_candidate_head = fp32_candidate_head
        self.eval_batch_size = eval_batch_size
        path = self.output_dir / 'epoch_metrics.json'
        self.history = json.loads(path.read_text()) if path.exists() else []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if math.isclose(state.epoch, epoch, abs_tol=1e-6) and epoch > 0:
            if self.validation is not None and not any(r['epoch'] == epoch for r in self.history):
                was_training = model.training
                report = evaluate(model, self.validation, self.collator,
                                  self.output_dir / f'epoch-{epoch}.json',
                                  fp32_candidate_head=self.fp32_candidate_head,
                                  eval_batch_size=self.eval_batch_size)
                model.train(was_training)
                self.history.append({'epoch': epoch, 'step': state.global_step, 'summary': report['summary']})
                (self.output_dir / 'epoch_metrics.json').write_text(json.dumps(self.history, indent=2) + '\n')
            if self.stop_after_epochs and epoch >= self.stop_after_epochs:
                control.should_training_stop = True
        return control


def flatten_eval_summary(summary, prefix="eval"):
    metrics = {}
    for kind, group in summary.items():
        for key, value in group.items():
            if key in ("count", "correct"):
                continue
            metrics[f"{prefix}/{kind}/{key}"] = float(value)
    return metrics


class PeriodicEvalAndSwanLabCallback(TrainerCallback):
    """Log training metrics to SwanLab and run periodic eval on the validation set."""
    def __init__(self, validation, collator, output_dir, eval_steps=0, swanlab_mod=None,
                 fp32_candidate_head=False, eval_batch_size=1):
        self.validation = validation
        self.collator = collator
        self.output_dir = Path(output_dir)
        self.eval_steps = eval_steps
        self.swanlab = swanlab_mod
        self.fp32_candidate_head = fp32_candidate_head
        self.eval_batch_size = eval_batch_size
        self.last_eval_step = -1
        self.last_log_time = None
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.swanlab is None or not logs:
            return control
        now = time.monotonic()
        payload = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                payload[f"train/{k}"] = float(v)
        if self.last_log_time is not None and state.global_step > self.last_log_step:
            dt = now - self.last_log_time
            if dt > 0:
                payload["train/steps_per_sec"] = (state.global_step - self.last_log_step) / dt
        self.last_log_time = now
        self.last_log_step = state.global_step
        if payload:
            self.swanlab.log(payload, step=state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps and state.global_step > 0 and state.global_step % self.eval_steps == 0:
            if state.global_step != self.last_eval_step and model is not None:
                was_training = model.training
                report = evaluate(model, self.validation, self.collator,
                                  self.output_dir / f"step-{state.global_step}.json",
                                  fp32_candidate_head=self.fp32_candidate_head,
                                  eval_batch_size=self.eval_batch_size)
                model.train(was_training)
                self.last_eval_step = state.global_step
                if self.swanlab is not None:
                    self.swanlab.log(flatten_eval_summary(report["summary"]), step=state.global_step)
        return control


def train(args):
    if any(v <= 0 for v in (args.max_steps, args.learning_rate, args.batch_size, args.gradient_accumulation,
                            args.max_length, args.lora_rank, args.save_steps)):
        raise ValueError("Training sizes, steps and learning rate must be positive")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("Warmup must be nonnegative and smaller than max-steps")
    if args.stop_after_epochs is not None and args.stop_after_epochs <= 0:
        raise ValueError('stop-after-epochs must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Use a new output directory or explicitly resume a checkpoint")
    set_seed(args.seed)
    model_id, revision = resolve_checkpoint(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_train_rows = None
    sync_pairs = args.dynamic_choice_shuffle or args.pair_batch_sampling or args.flip_loss_weight > 0
    if args.cv_folds > 0:
        training, validation, audit, raw_train_rows = prepare_cv_data(
            args.data_dir, args.validation, tokenizer, args.max_length,
            num_folds=args.cv_folds, fold_index=args.cv_fold, cv_seed=args.cv_seed,
            model_id=model_id, revision=revision,
            preserve_ordered_kinds=args.preserve_ordered_kinds,
            sync_pair_shuffle=sync_pairs,
            return_raw_train=True,
            prompt_format=args.prompt_format,
        )
    else:
        training, validation, audit = prepare_data(
            args.data_dir, args.validation, tokenizer, args.max_length, model_id, revision
        )
        raw_train_rows = read_rows(args.data_dir / "train.jsonl")
        if args.preserve_ordered_kinds or sync_pairs or args.prompt_format != "v1":
            manifest = json.loads((args.data_dir / "manifest.json").read_text())
            s_seed = manifest.get("tokenization_reference", {}).get("shuffle_seed", args.seed)
            training = [
                runtime_record(
                    encode_scoring(
                        as_scoring(r, True), tokenizer, args.max_length,
                        shuffle_seed=s_seed,
                        preserve_ordered_kinds=args.preserve_ordered_kinds,
                        sync_pair_shuffle=sync_pairs,
                        prompt_format=args.prompt_format,
                    ),
                    r,
                )
                for r in raw_train_rows
            ]
        if args.prompt_format != "v1":
            raw_val_rows = read_rows(args.validation)
            validation = [
                runtime_record(
                    encode_scoring(as_scoring(r, False), tokenizer, args.max_length, prompt_format=args.prompt_format),
                    r,
                )
                for r in raw_val_rows
            ]
            audit["prompt_format"] = args.prompt_format
            audit["train_max_tokens"] = max(len(r["input_ids"]) for r in training)
            audit["validation_max_tokens"] = max(len(r["input_ids"]) for r in validation)
    dynamic_ds = None
    if args.dynamic_choice_shuffle:
        dynamic_ds = DynamicChoiceDataset(
            training, raw_train_rows, tokenizer, args.max_length, base_seed=args.seed,
            preserve_ordered_kinds=args.preserve_ordered_kinds, sync_pair_shuffle=True,
            prompt_format=args.prompt_format,
        )
        training = dynamic_ds
    prompt_hash = hashlib.sha256((Path(__file__).parents[1] / "scoring/parallel_schema.py").read_bytes()).hexdigest()
    contract = {"task": "schema_candidate_classification_v1", "model": model_id, "revision": revision,
                "system_prompt": SYSTEM_PROMPT, "prompt_code_sha256": prompt_hash, "max_length": args.max_length,
                "lora_rank": args.lora_rank, "seed": args.seed, "data_audit": audit,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation, "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps, "lr_scheduler": args.lr_scheduler, "weight_decay": args.weight_decay}
    if args.prompt_format != "v1":
        contract["prompt_format"] = args.prompt_format
    if args.cv_folds > 0:
        contract.update(cv_folds=args.cv_folds, cv_fold=args.cv_fold, cv_seed=args.cv_seed)
    if args.preserve_ordered_kinds:
        contract["preserve_ordered_kinds"] = True
    if args.dynamic_choice_shuffle:
        contract["dynamic_choice_shuffle"] = True
    if args.fp32_candidate_head:
        contract["fp32_candidate_head"] = True
    if args.unlock_deltanet:
        contract["unlock_deltanet"] = True
    if args.ema_decay > 0:
        contract["ema_decay"] = args.ema_decay
    contract["inference_precision"] = "fp32_candidate_head" if args.fp32_candidate_head else "bf16_autocast"
    if args.eval_each_epoch or args.stop_after_epochs:
        contract.update(eval_each_epoch=args.eval_each_epoch, stop_after_epochs=args.stop_after_epochs)
    if args.resume_from_checkpoint:
        previous = json.loads((Path(args.resume_from_checkpoint) / "schema_config.json").read_text())
        # The first pilot already used Trainer's BF16 autocast before this was explicit.
        previous.setdefault("inference_precision", "bf16_autocast")
        if any(previous.get(k) != v for k, v in contract.items()):
            raise ValueError("Resume contract differs; preserve the original data and optimization settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    base = load_base(model_id, revision)
    if args.drop_vision and hasattr(base, "model") and hasattr(base.model, "visual"):
        delattr(base.model, "visual")
        torch.cuda.empty_cache()
    targets = [name for name, layer in base.named_modules() if isinstance(layer, torch.nn.Linear) and ".language_model." in name]
    if not targets:
        raise RuntimeError("No language linear layers matched")
    model = get_peft_model(base, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                          lora_dropout=.05, target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    unlocked_names, unlocked_scalars = [], 0
    if args.unlock_deltanet:
        unlocked_names, unlocked_scalars = unlock_deltanet_parameters(model)
        print(f"Unlocked {len(unlocked_names)} DeltaNet/RMSNorm tensors ({unlocked_scalars:,} params)", flush=True)
        contract["unlocked_deltanet_tensors"] = len(unlocked_names)
        contract["unlocked_deltanet_params"] = unlocked_scalars
    model.print_trainable_parameters()
    contract.update(versions={p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                    gpu=torch.cuda.get_device_name(), target_modules=targets)
    (args.output_dir / "schema_config.json").write_text(json.dumps(contract, indent=2) + "\n")
    collator = CandidateCollator(tokenizer.pad_token_id)
    swanlab_mod = None
    if args.swanlab_project:
        import swanlab as swanlab_mod
        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            swanlab_mod.login(api_key=api_key)
        swanlab_mod.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_run_name or args.output_dir.name,
            config={
                "model": model_id,
                "revision": revision,
                "prompt_format": args.prompt_format,
                "learning_rate": args.learning_rate,
                "lr_scheduler": args.lr_scheduler,
                "weight_decay": args.weight_decay,
                "lora_rank": args.lora_rank,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
                "max_length": args.max_length,
                "max_steps": args.max_steps,
                "warmup_steps": args.warmup_steps,
                "stop_after_epochs": args.stop_after_epochs,
                "eval_steps": args.eval_steps,
                "cv_folds": args.cv_folds,
                "cv_fold": args.cv_fold if args.cv_folds > 0 else None,
                "cv_seed": args.cv_seed if args.cv_folds > 0 else None,
                "preserve_ordered_kinds": args.preserve_ordered_kinds,
                "dynamic_choice_shuffle": args.dynamic_choice_shuffle,
                "fp32_candidate_head": args.fp32_candidate_head,
                "drop_vision": args.drop_vision,
                "pair_batch_sampling": args.pair_batch_sampling,
                "pair_sampling_mode": args.pair_sampling_mode,
                "flip_loss_weight": args.flip_loss_weight,
                "flip_margin": args.flip_margin,
                "flip_multiclass_only": args.flip_multiclass_only,
                "unlock_deltanet": args.unlock_deltanet,
                "unlocked_deltanet_tensors": len(unlocked_names),
                "unlocked_deltanet_params": unlocked_scalars,
                "ema_decay": args.ema_decay,
                "seed": args.seed,
                "gpu": contract["gpu"],
            },
        )
    initial = (evaluate(model, validation, collator, args.output_dir / "before.json",
                        fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=args.eval_batch_size)
               if not args.resume_from_checkpoint else None)
    if swanlab_mod is not None and initial is not None:
        swanlab_mod.log(flatten_eval_summary(initial["summary"]), step=0)
    args_hf = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.max_steps, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps, lr_scheduler_type=args.lr_scheduler, weight_decay=args.weight_decay,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=1, report_to="none", remove_unused_columns=False, label_names=["labels"],
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, max_grad_norm=1.,
        optim="adamw_torch", logging_nan_inf_filter=False,
    )
    callbacks = [SaveSchemaContract(contract, save_extra_params=args.unlock_deltanet)]
    ema = None
    if args.ema_decay > 0:
        ema = ParameterEMA(model, decay=args.ema_decay, warmup_steps=args.warmup_steps)
        callbacks.append(EMACallback(ema))
    if dynamic_ds is not None:
        callbacks.append(DynamicShuffleCallback(dynamic_ds))
    if args.eval_each_epoch or args.stop_after_epochs:
        callbacks.append(EpochValidation(validation if args.eval_each_epoch else None, collator,
                                         args.output_dir, args.stop_after_epochs,
                                         fp32_candidate_head=args.fp32_candidate_head,
                                         eval_batch_size=args.eval_batch_size))
    periodic_cb = None
    if args.eval_steps > 0 or swanlab_mod is not None:
        periodic_cb = PeriodicEvalAndSwanLabCallback(validation, collator, args.output_dir,
                                                     eval_steps=args.eval_steps, swanlab_mod=swanlab_mod,
                                                     fp32_candidate_head=args.fp32_candidate_head,
                                                     eval_batch_size=args.eval_batch_size)
        callbacks.append(periodic_cb)
    trainer = CandidateTrainer(model=model, args=args_hf, train_dataset=training, data_collator=collator,
                               processing_class=tokenizer, callbacks=callbacks,
                               fp32_candidate_head=args.fp32_candidate_head,
                               pair_batch_sampling=args.pair_batch_sampling,
                               pair_sampling_mode=args.pair_sampling_mode,
                               flip_loss_weight=args.flip_loss_weight,
                               flip_margin=args.flip_margin,
                               flip_multiclass_only=args.flip_multiclass_only)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    # Always run final evaluation unpadded (batch_size=1) for exact apples-to-apples comparison.
    final = evaluate(model, validation, collator, args.output_dir / "after.json",
                     fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=1)
    cal_report = {"raw": evaluate_lodo_temperature_calibration(final["rows"], summarize)}
    final_ema = None
    if ema is not None:
        with ema.apply_to(model):
            final_ema = evaluate(model, validation, collator, args.output_dir / "after_ema.json",
                                 fp32_candidate_head=args.fp32_candidate_head, eval_batch_size=1)
            ema_dir = args.output_dir / "ema_adapter"
            ema_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ema_dir))
            tokenizer.save_pretrained(ema_dir)
            (ema_dir / "schema_config.json").write_text(json.dumps(contract, indent=2) + "\n")
            if args.unlock_deltanet:
                save_extra_trainable_params(model, ema_dir)
        cal_report["ema"] = evaluate_lodo_temperature_calibration(final_ema["rows"], summarize)
    (args.output_dir / "calibration_report.json").write_text(json.dumps(cal_report, indent=2) + "\n")
    if swanlab_mod is not None:
        if periodic_cb is None or periodic_cb.last_eval_step != trainer.state.global_step:
            swanlab_mod.log(flatten_eval_summary(final["summary"]), step=trainer.state.global_step)
        if final_ema is not None:
            swanlab_mod.log(flatten_eval_summary(final_ema["summary"], prefix="eval_ema"),
                            step=trainer.state.global_step)
        if cal_report["raw"] is not None:
            swanlab_mod.log(flatten_eval_summary(cal_report["raw"]["lodo_calibrated_summary"], prefix="eval_cal"),
                            step=trainer.state.global_step)
        swanlab_mod.finish()
    trainer.save_model(str(args.output_dir))
    if args.unlock_deltanet:
        save_extra_trainable_params(model, args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    report = {"before": initial["summary"] if initial else None, "after": final["summary"],
              "after_ema": final_ema["summary"] if final_ema else None,
              "lodo_calibrated_raw": cal_report["raw"]["lodo_calibrated_summary"] if cal_report["raw"] else None,
              "lodo_calibrated_ema": cal_report["ema"]["lodo_calibrated_summary"] if cal_report.get("ema") else None,
              "training": result.metrics, "optimizer_steps": trainer.state.global_step,
              "training_wall_seconds": elapsed, "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "limitations": ["Model-generated, non-human-reviewed labels", f"Only {len(validation)} exploratory validation examples",
                              "Test split not used", "Single seeded run; no hyperparameter search or calibration",
                              "Schema classification data, not the original MiniCheck curriculum"]}
    if args.eval_each_epoch:
        report['limitations'][3] = 'Hyperparameter selection uses this inner validation set; requires separate outer evaluation'
    elif args.stop_after_epochs:
        report['limitations'][3] = 'Fixed epoch count; no outer-evaluation checkpoint selection; consult run plan for recipe provenance'
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def score(args):
    contract = json.loads((args.adapter / "schema_config.json").read_text())
    prompt_hash = hashlib.sha256((Path(__file__).parents[1] / "scoring/parallel_schema.py").read_bytes()).hexdigest()
    if contract["task"] != "schema_candidate_classification_v1" or contract["prompt_code_sha256"] != prompt_hash:
        raise ValueError("Saved adapter task or prompt implementation differs")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    schema = json.loads(args.schema_file.read_text())
    prepared = prepare_prompts(
        tokenizer, args.context, schema, contract["max_length"],
        prompt_format=contract.get("prompt_format", "v1"),
    )
    model = PeftModel.from_pretrained(load_base(contract["model"], contract["revision"]), args.adapter).eval()
    if contract.get("unlock_deltanet"):
        load_extra_trainable_params(model, args.adapter)
    collator = CandidateCollator(tokenizer.pad_token_id)
    fp32_head = bool(contract.get("fp32_candidate_head", False))
    fields = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, name in enumerate(prepared.names):
            row = {"input_ids": prepared.full_ids[i], "candidate_ids": prepared.candidate_ids[i], "choices": prepared.choices[i]}
            if schema[name]["type"] == "boolean":
                row["kind"] = "noul"
            batch = {k: v.to("cuda") for k, v in collator([row]).items()}
            fields[name] = decision_result(row, candidate_logits(model, batch, fp32_candidate_head=fp32_head)[0].cpu())
    print(json.dumps({"output": {name: r["prediction"] for name, r in fields.items()}, "fields": fields}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument('--model', default=MODEL_ID)
    fit.add_argument('--revision', help='Pinned model revision; required when changing the default model')
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--max-steps", type=int, default=30)
    fit.add_argument("--warmup-steps", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=1)
    fit.add_argument("--gradient-accumulation", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=2e-5)
    fit.add_argument("--lr-scheduler", default="linear", choices=["linear", "cosine"])
    fit.add_argument("--weight-decay", type=float, default=0.0)
    fit.add_argument("--max-length", type=int, default=1024)
    fit.add_argument("--lora-rank", type=int, default=16)
    fit.add_argument("--save-steps", type=int, default=10)
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--resume-from-checkpoint")
    fit.add_argument('--eval-each-epoch', action='store_true', help='Measure validation at complete epoch boundaries')
    fit.add_argument('--stop-after-epochs', type=int, help='Stop at an epoch boundary while retaining the max-steps LR schedule')
    fit.add_argument('--eval-steps', type=int, default=0, help='Run validation every N optimizer steps')
    fit.add_argument('--eval-batch-size', type=int, default=1, help='Batch size during evaluation')
    fit.add_argument('--cv-folds', type=int, default=0, help='Enable K-fold source-family inner CV (0 = disabled)')
    fit.add_argument('--cv-fold', type=int, default=0, help='Fold index in [0, cv-folds - 1] when --cv-folds > 0')
    fit.add_argument('--cv-seed', type=int, default=17, help='Deterministic seed for source-family CV partitioning')
    fit.add_argument('--preserve-ordered-kinds', action='store_true', help='Keep score ascending and noul [False, True] during training')
    fit.add_argument('--dynamic-choice-shuffle', action='store_true', help='Pair-synchronized per-epoch shuffling for choice rows')
    fit.add_argument('--fp32-candidate-head', action='store_true', help='Project only candidate rows in exact FP32 (matching CudaCandidateScorer)')
    fit.add_argument('--drop-vision', action='store_true', help='Delete unused vision encoder before LoRA wrapping')
    fit.add_argument('--prompt-format', default='v1', choices=['v1', 'sandwich_v2'],
                     help='Prompt layout: v1 (original JSON) or sandwich_v2 (schema-first, clean de-escaped context, tail anchor)')
    fit.add_argument('--pair-batch-sampling', action='store_true', help='Co-batch contrastive pairs')
    fit.add_argument('--pair-sampling-mode', default='length_bucket', choices=['length_bucket', 'stratified'],
                     help='PairBatchSampler ordering: length_bucket or kind-stratified')
    fit.add_argument('--flip-loss-weight', type=float, default=0.0, help='Weight for bidirectional pairwise log-odds flip loss')
    fit.add_argument('--flip-margin', type=float, default=2.0, help='Target average per-sibling log-odds margin for flip loss')
    fit.add_argument('--flip-multiclass-only', action='store_true', help='Apply flip loss only to K>=3 (choice/score) pairs')
    fit.add_argument('--unlock-deltanet', action='store_true', help='Unfreeze Gated DeltaNet A_log/dt_bias/conv1d and RMSNorm parameters')
    fit.add_argument('--ema-decay', type=float, default=0.0, help='FP32 EMA decay over trainable params (0 disables EMA)')
    fit.add_argument('--swanlab-project', help='SwanLab project name for real-time logging')
    fit.add_argument('--swanlab-run-name', help='SwanLab experiment/run name')
    predict = commands.add_parser("score")
    predict.add_argument("--adapter", type=Path, required=True)
    predict.add_argument("--context", required=True)
    predict.add_argument("--schema-file", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
