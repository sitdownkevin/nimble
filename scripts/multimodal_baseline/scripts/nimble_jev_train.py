#!/usr/bin/env python3
"""Train nimble's schema classifier (github.com/bespokelabsai/nimble, cloned to /root/projects/nimble) on jev-real-ad-1000.

nimble's own loader only accepts its curation provenance (evidence certificates, variant groups, frozen manifests), so
this wrapper leaves nimble's model, prompt, loss, collator and evaluation untouched and replaces only its data
preparation (`prepare_data`) with a converter from our labelled text data (kev_data/*.jsonl, the same records as the kev
text runs). It also adds SwanLab logging: training loss at every logged step, validation scores every N optimizer
steps (and at step 0 and the end), and one test evaluation after training. Nothing is selected on the test split.

    SWANLAB_API_KEY=... CUDA_VISIBLE_DEVICES=<gpu> /root/projects/kev/.venv/bin/python scripts/nimble_jev_train.py \
        --output-dir runs/nimble-0.8b-text --eval-every 20
"""
import argparse, json, os, sys
from pathlib import Path

NIMBLE = Path("/root/projects/nimble")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NIMBLE))

import swanlab
from transformers import TrainerCallback
import nimble.training.schema_train as st
from nimble.training.schema_data import as_scoring, encode_scoring, runtime_record

INSTRUCTIONS = "这条广告属于以下哪个行业类目？"
OOC_DESCRIPTION = "广告所属行业不在以上候选类目中"
SHUFFLE_SEED = 17                      # nimble shuffles option order per training record with a fixed seed; validation keeps order
STATE = {}                             # filled by prepare(): validation/test rows, collator, sizes


def load_split(name):
    """kev_data/<name>.jsonl (text-only kev records) -> nimble raw records with an explicit model input and reference."""
    rows = []
    for line in (ROOT / "kev_data" / f"{name}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        q = r["questions"]["industry"]
        criteria = {}
        for option, desc in q["criteria"].items():
            # nimble's schema needs a nonempty string description per option (None is not accepted)
            criteria[option] = desc if isinstance(desc, str) and desc.strip() else (OOC_DESCRIPTION if option == "以上都不是" else option)
        rows.append({
            "id": r["id"], "family": r["id"], "source_family": r["id"], "provenance": {"source_id": r["id"]},
            "input": {"state": r["state"], "questions": {"decision": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}},
            "reference": {"target": q["label"]},
        })
    return rows


def prepare(directory, validation_path, tokenizer, max_length, model_id, revision):
    """Drop-in replacement for nimble.training.schema_data.prepare_data: (train_rows, validation_rows, audit)."""
    raw_train, raw_val, raw_test = load_split("train_all"), load_split("validation"), load_split("test")
    train_data, lengths = [], []
    for raw in raw_train:
        enc = encode_scoring(as_scoring(raw, True), tokenizer, max_length, SHUFFLE_SEED)
        train_data.append(runtime_record(enc, raw)); lengths.append(len(enc["prompt_token_ids"]))
    def inference(raws):
        return [runtime_record(encode_scoring(as_scoring(r, False), tokenizer, max_length), r) for r in raws]
    val_data, test_data = inference(raw_val), inference(raw_test)
    STATE.update(val=val_data, test=test_data, collator=st.CandidateCollator(tokenizer.pad_token_id))
    audit = {"source": "jev-real-ad-1000 kev_data (text fields only; images are not used by nimble)",
             "training_rows": len(train_data), "validation_rows": len(val_data), "test_rows": len(test_data),
             "train_max_tokens": max(lengths), "train_mean_tokens": sum(lengths) // len(lengths),
             "validation_order": "original choice order; no shuffling", "test_split_used": "once, after training"}
    print(json.dumps({"audit": audit}, ensure_ascii=False), flush=True)
    return train_data, val_data, audit


class SwanLogging(TrainerCallback):
    def __init__(self, every, out_dir):
        self.every = every
        self.out = Path(out_dir) / "eval"
        self.out.mkdir(parents=True, exist_ok=True)

    def _log_report(self, report, step, split):
        s = report["summary"]
        swanlab.log({f"{split}/accuracy": s["all"]["accuracy"], f"{split}/nll": s["all"]["nll"], f"{split}/brier": s["all"]["brier"]}, step=step)
        for kind in ("choice", "noul", "score"):
            if kind in s:
                swanlab.log({f"{split}/accuracy_{kind}": s[kind]["accuracy"]}, step=step)

    def _validate(self, model, step):
        was = model.training
        report = st.evaluate(model, STATE["val"], STATE["collator"], self.out / f"validation-step-{step:04d}.json")
        model.train(was)
        self._log_report(report, step, "eval")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._validate(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step and state.global_step % self.every == 0:
            self._validate(model, state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            swanlab.log({k: v for k, v in logs.items() if isinstance(v, (int, float))}, step=state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        self._validate(model, step)
        was = model.training
        report = st.evaluate(model, STATE["test"], STATE["collator"], self.out / "test-final.json")
        model.train(was)
        self._log_report(report, step, "test")


class JevCandidateTrainer(st.CandidateTrainer):
    """nimble's CandidateTrainer with the SwanLab callback added (nimble's own loss and training loop are unchanged)."""
    EVAL_EVERY, OUT = 20, None

    def __init__(self, *args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [SwanLogging(self.EVAL_EVERY, self.OUT)]
        super().__init__(*args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=264)          # 700 records, batch 1 x 8 accumulation, 3 epochs
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=88)          # one checkpoint per epoch
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--swanlab-project", default="jev-real-ad-1000")
    ap.add_argument("--swanlab-run", default="nimble-0.8b-text-700")
    a = ap.parse_args()

    st.prepare_data = prepare                                     # the only change to nimble's data path
    JevCandidateTrainer.EVAL_EVERY, JevCandidateTrainer.OUT = a.eval_every, a.output_dir
    st.CandidateTrainer = JevCandidateTrainer

    key = os.environ.get("SWANLAB_API_KEY")
    if key:
        swanlab.login(api_key=key, save=False)
    swanlab.init(project=a.swanlab_project, experiment_name=a.swanlab_run,
                 config={"trainer": "nimble", "model": a.model, "revision": a.revision, "learning_rate": a.learning_rate,
                         "max_steps": a.max_steps, "warmup_steps": a.warmup_steps, "lora_rank": a.lora_rank,
                         "batch_size": 1, "gradient_accumulation": 8, "max_length": a.max_length, "seed": 17,
                         "eval_every": a.eval_every, "data": "jev-real-ad-1000 text (kev_data)"})
    sys.argv = ["schema_train", "train", "--model", a.model, "--revision", a.revision,
                "--data-dir", str(ROOT / "kev_data"), "--validation", str(ROOT / "kev_data" / "validation.jsonl"),
                "--output-dir", str(a.output_dir), "--learning-rate", str(a.learning_rate), "--lora-rank", str(a.lora_rank),
                "--seed", "17", "--batch-size", "1", "--gradient-accumulation", "8", "--max-length", str(a.max_length),
                "--max-steps", str(a.max_steps), "--warmup-steps", str(a.warmup_steps), "--save-steps", str(a.save_steps)]
    try:
        st.main()
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
