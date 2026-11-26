#!/usr/bin/env python3
"""Train nimble with the advertisement image (multimodal nimble), same recipe as the text run (nimble-0.8b-text-700):
LoRA rank 16 on the language model, lr 1e-4, batch 1 x 8 accumulation, 3 epochs (264 steps), seed 17, option order
shuffled per training record, validation every N steps, and the test split once after training.

The data is kev_data/*_pos_start.jsonl: the same 700/150/150 advertisements, with the image (kev's preprocessing) in the
prompt. nimble's trainer, loss and evaluation are used as they are; scripts/nimble_mm.py supplies the image inputs.

    SWANLAB_API_KEY=... CUDA_VISIBLE_DEVICES=<gpu> /root/projects/kev/.venv/bin/python scripts/nimble_mm_train.py \
        --output-dir runs/nimble-0.8b-mm-700 --eval-every 20
"""
import argparse, os, sys

import swanlab

import nimble_mm as M          # sets up the nimble and kev import paths first
import nimble_jev_train as J
import nimble.training.schema_train as st

SHUFFLE_SEED = 17


def prepare_mm(directory, validation_path, tokenizer, max_length, model_id, revision):
    """Replaces nimble's data preparation: (train rows, validation rows, audit) with the image in every prompt."""
    def build(rows, train):
        out, lengths = [], []
        for r in rows:
            context, schema = M.schema_for(r)
            enc = M.encode_one(tokenizer, context, schema, M.image_for(r["image_path"]),
                               shuffle_seed=SHUFFLE_SEED if train else None, row_id=r["id"])
            out.append(M.runtime_row(enc, r["id"], r["id"], "choice", r["label"]))
            lengths.append(len(enc["prompt_token_ids"]))
        return out, lengths

    train_data, train_lengths = build(M.load_mm_split("train_all_pos_start"), True)
    val_data, _ = build(M.load_mm_split("validation_pos_start"), False)
    test_data, _ = build(M.load_mm_split("test_pos_start"), False)
    J.STATE.update(val=val_data, test=test_data, collator=M.MMCollator(tokenizer.pad_token_id))
    audit = {"source": "jev-real-ad-1000 kev_data *_pos_start.jsonl (image in every prompt, kev preprocessing)",
             "training_rows": len(train_data), "validation_rows": len(val_data), "test_rows": len(test_data),
             "train_max_tokens": max(train_lengths), "train_mean_tokens": sum(train_lengths) // len(train_lengths),
             "validation_order": "original choice order; no shuffling", "test_split_used": "once, after training"}
    print(f"audit: {audit}", flush=True)
    return train_data, val_data, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--revision", default="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=264)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=88)
    ap.add_argument("--max-length", type=int, default=M.MAX_LENGTH)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--swanlab-project", default="jev-real-ad-1000")
    ap.add_argument("--swanlab-run", default="nimble-0.8b-mm-700")
    a = ap.parse_args()

    st.prepare_data = prepare_mm
    M.install(st)
    J.JevCandidateTrainer.EVAL_EVERY, J.JevCandidateTrainer.OUT = a.eval_every, a.output_dir
    st.CandidateTrainer = J.JevCandidateTrainer

    key = os.environ.get("SWANLAB_API_KEY")
    if key:
        swanlab.login(api_key=key, save=False)
    swanlab.init(project=a.swanlab_project, experiment_name=a.swanlab_run,
                 config={"trainer": "nimble", "modality": "text + advertisement image", "model": a.model,
                         "revision": a.revision, "learning_rate": a.learning_rate, "max_steps": a.max_steps,
                         "warmup_steps": a.warmup_steps, "lora_rank": a.lora_rank, "batch_size": 1,
                         "gradient_accumulation": 8, "max_length": a.max_length, "seed": 17,
                         "eval_every": a.eval_every, "data": "jev-real-ad-1000 *_pos_start (700/150/150)"})
    sys.argv = ["schema_train", "train", "--model", a.model, "--revision", a.revision,
                "--data-dir", str(M.DATA), "--validation", str(M.DATA / "validation_pos_start.jsonl"),
                "--output-dir", a.output_dir, "--learning-rate", str(a.learning_rate), "--lora-rank", str(a.lora_rank),
                "--seed", "17", "--batch-size", "1", "--gradient-accumulation", "8", "--max-length", str(a.max_length),
                "--max-steps", str(a.max_steps), "--warmup-steps", str(a.warmup_steps), "--save-steps", str(a.save_steps)]
    try:
        st.main()
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
