#!/usr/bin/env python3
"""Score the nimble-trained adapter (runs/nimble-4b-text-700) on kev's frozen evaluation suites.

Each (record, question) of a suite is one nimble decision: the suite state plus that question, mapped to nimble's own
model input (nimble.evaluation.evaluate_pilot.adapt_input) and scored with nimble's own candidate-logit scorer and
evaluation (nimble.training.schema_train.evaluate). Only the questions of the suite's "clean" variant are scored, the
same question set kev.benchmark reports.

Text mapping: kev's structured instructions and option descriptions are flattened with kev.api.render (the same text kev
shows its model); an empty text falls back to the question id (instructions) or the option name (options), because
nimble's schema needs a nonempty description per field and option.

nimble can only score fields with at most 26 choices (one letter code each, MAX_CHOICES in schema_train); the
decision-v7 banking77 intent questions have 77 options and are skipped. Accuracy is reported over the scorable questions,
and also with the skipped ones counted as wrong (the same denominator as kev's 1,200-question clean set).

    --check-validation   score our own validation split first (not a locked test): must match the training run's 0.8267
    --suite <name>       read a locked test split once (decision-v7 or transfer-v4)
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

NIMBLE = Path("/root/projects/nimble")
ROOT = Path(__file__).resolve().parents[1]
KEV = Path("/root/projects/kev")
sys.path.insert(0, str(NIMBLE))
sys.path.insert(0, str(KEV))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoTokenizer
from peft import PeftModel
import nimble.training.schema_train as st
from nimble.training.schema_data import as_scoring, encode_scoring, runtime_record
from kev.api import render
import nimble_jev_train as jev

MAX_CHOICES = st.MAX_CHOICES
SUITES = {
    "decision-v7": (KEV / "evals/v7/decision-v7/test.jsonl", ROOT / "runs/eval-test-decision-v7-mm4b/rows.json"),
    "transfer-v4": (KEV / "evals/v4/transfer-v4/test.jsonl", ROOT / "runs/eval-test-transfer-v4-mm4b/rows.json"),
}


def text(value):
    """kev's rendered text of a value (str, object or list), or None when it is empty. Strings are kept exactly as they are."""
    if isinstance(value, str):
        return value if value.strip() else None
    flat = render(value).strip() if value is not None else ""
    return flat or None


def sanitize(q, qid):
    """kev question -> nimble question (type, instructions text, criteria with a nonempty string per option)."""
    kind, crit = q["type"], q.get("criteria")
    instructions = text(q.get("instructions")) or qid
    if kind == "choice":
        criteria = {k: text(v) or k for k, v in crit.items()}
    elif kind == "noul":
        crit = crit or {}
        criteria = {k: text(crit.get(k)) or k for k in ("true", "false")}
    elif kind == "score":
        criteria = [text(v) or str(i) for i, v in enumerate(crit)]
    else:
        raise ValueError(f"unsupported question type {kind}")
    return {"type": kind, "instructions": instructions, "criteria": criteria}


def load_model(adapter, model_id, revision):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = st.load_base(model_id, revision)
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    return tok, model


def suite_rows(path, clean_keys, tok, max_length):
    """-> (runtime rows, task by row id, skipped questions with more than 26 choices)."""
    rows, tasks, skipped = [], {}, []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        rid = rec["_meta"]["id"]
        for qid, q in rec["questions"].items():
            key = (rid, qid)
            if key not in clean_keys:
                continue
            if q["type"] == "choice" and len(q["criteria"]) > MAX_CHOICES:
                skipped.append(key)
                continue
            raw = {"id": f"{rid}::{qid}", "family": rid, "source_family": rid,
                   "provenance": {"source_id": rid},
                   "input": {"state": rec["state"], "questions": {"decision": sanitize(q, qid)}},
                   "reference": {"target": q["label"]}}
            rows.append(runtime_record(encode_scoring(as_scoring(raw, False), tok, max_length), raw))
            tasks[raw["id"]] = clean_keys[key]
    return rows, tasks, skipped


def clean_question_keys(rows_path):
    rows = json.load(open(rows_path))
    return {(r["id"], r["question"]): r["task"] for r in rows if r.get("variant") == "clean"}


def accuracy(rows):
    return (sum(r["correct"] for r in rows) / len(rows)) if rows else float("nan")


def check_validation(args, tok, model):
    raws = jev.load_split("validation")
    rows = [runtime_record(encode_scoring(as_scoring(r, False), tok, args.max_length), r) for r in raws]
    collator = st.CandidateCollator(tok.pad_token_id)
    report = st.evaluate(model, rows, collator, args.out / "validation-check.json")
    acc = report["summary"]["all"]["accuracy"]
    print(json.dumps({"check": "validation", "accuracy": acc, "expected_from_training": 0.8266666666666667, "n": len(rows)}), flush=True)


def run_suite(args, tok, model, name):
    suite_path, rows_path = SUITES[name]
    clean = clean_question_keys(rows_path)
    rows, tasks, skipped = suite_rows(suite_path, clean, tok, args.max_length)
    collator = st.CandidateCollator(tok.pad_token_id)
    report = st.evaluate(model, rows, collator, args.out / f"{name}-test.json")
    results = {r["id"]: r for r in report["rows"]}
    scored = [results[r["id"]] for r in rows]
    for r in scored:
        r["task"] = tasks[r["id"]]
    by_type, by_task = defaultdict(list), defaultdict(list)
    for r in scored:
        by_type[r["kind"]].append(r)
        by_task[r["task"]].append(r)
    total_clean = len(clean)
    summary = {
        "suite": name,
        "clean_questions": total_clean,
        "scored": len(scored),
        "skipped_over_26_choices": len(skipped),
        "accuracy_scored": accuracy(scored),
        "accuracy_all_clean_skipped_as_wrong": sum(r["correct"] for r in scored) / total_clean,
        "by_type": {k: {"n": len(v), "accuracy": accuracy(v)} for k, v in sorted(by_type.items())},
        "by_task": {k: {"n": len(v), "accuracy": accuracy(v)} for k, v in sorted(by_task.items(), key=lambda kv: -accuracy(kv[1]))},
        "skipped_by_task": dict(Counter(clean[k] for k in skipped)),
    }
    (args.out / f"{name}-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("suite", "clean_questions", "scored", "skipped_over_26_choices",
                                              "accuracy_scored", "accuracy_all_clean_skipped_as_wrong")}, ensure_ascii=False), flush=True)
    print(json.dumps({"by_type": summary["by_type"]}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=ROOT / "runs/nimble-4b-text-700")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--revision", default="1001bb4d826a52d1f399e183466143f4da7b741b")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/generalization-nimble-4b")
    ap.add_argument("--check-validation", action="store_true")
    ap.add_argument("--suite", choices=list(SUITES))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(args.adapter, args.model, args.revision)
    if args.check_validation:
        check_validation(args, tok, model)
    if args.suite:
        run_suite(args, tok, model, args.suite)


if __name__ == "__main__":
    main()
