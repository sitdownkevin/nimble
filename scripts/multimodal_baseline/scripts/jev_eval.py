#!/usr/bin/env python3
"""Evaluate TypeSafe's hosted Jev on the jev-real-ad-1000 text test split (kev_data/test.jsonl, 150 records).

Each record is one System One request: the same text state the kev and nimble text models saw, one choice question over
the 25 retrieved candidates plus "以上都不是" (out-of-candidate). Jev's answer is the chosen option; probabilities give
NLL and Brier. Jev is zero-shot here (no training on this data), so this is a reference point, not a fine-tuned model.

    --dry-run            build and validate every request (kev's SystemOneRequest schema), call nothing
    (default)            call https://api.typesafe.ai/v1/systemone, key from the JEV_API_KEY environment variable only

The key is read from the environment and never written to disk or printed. Calls are sequential with retries on
429/5xx; the run stops at the first 401/403 or after too many consecutive failures.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
KEV = Path("/root/projects/kev")
sys.path.insert(0, str(KEV))

URL = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = "这条广告属于以下哪个行业类目？"
OOC = "以上都不是"


def load_requests(limit=None):
    out = []
    for line in (ROOT / "kev_data" / "test.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        q = r["questions"]["industry"]
        out.append({
            "id": r["id"],
            "gold": q["label"],
            "body": {
                "state": r["state"],
                "model": None,                       # filled in per run
                "questions": {"industry": {"type": "choice", "instructions": INSTRUCTIONS,
                                           "criteria": q["criteria"]}},
            },
        })
    return out[:limit] if limit else out


def validate(body):
    """Every request must pass kev's System One request model (the same shape TypeSafe's API documents)."""
    from kev.api import SystemOneRequest
    SystemOneRequest.model_validate(body)


def call(body, key, timeout=90, retries=4):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", "replace")[:300]
            if e.code in (401, 403):
                raise PermissionError(f"HTTP {e.code}: the key was rejected ({body_text[:120]})")
            last = (e.code, body_text)
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue
            return e.code, {"error": body_text}
        except (urllib.error.URLError, TimeoutError) as e:
            last = (0, str(e))
            time.sleep(min(2 ** attempt, 20))
    return last[0], {"error": str(last[1])}


def score(gold, answer, criteria_keys):
    """-> (predicted option, probability of gold, nll, brier) from a Jev choice answer."""
    pred = answer.get("choice")
    probs = answer.get("probabilities") or {}
    p_gold = float(probs.get(gold, 0.0))
    total = sum(float(v) for v in probs.values()) or 1.0
    norm = {k: float(v) / total for k, v in probs.items()}
    # Jev returns probabilities rounded to 0.01, so a true probability below 0.005 shows as 0.0; floor the gold
    # probability at 0.005 instead of 1e-9 (otherwise every rounded-to-zero miss costs ~20 nats)
    nll = -math.log(max(norm.get(gold, 0.0), 0.005)) if probs else None
    brier = sum((norm.get(k, 0.0) - (1.0 if k == gold else 0.0)) ** 2 for k in criteria_keys) if probs else None
    return pred, p_gold, nll, brier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev-latest", help="model id sent as the request's model field")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/jev-eval-test")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-failures", type=int, default=10, help="stop after this many consecutive failed calls")
    args = ap.parse_args()

    items = load_requests(args.limit)
    for it in items:
        it["body"]["model"] = args.model
        validate(it["body"])
    print(json.dumps({"requests": len(items), "model": args.model, "endpoint": URL,
                      "sample_state": items[0]["body"]["state"], "options": len(items[0]["body"]["questions"]["industry"]["criteria"])},
                     ensure_ascii=False), flush=True)
    if args.dry_run:
        print("dry run: all requests validated, nothing sent", flush=True)
        return

    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        sys.exit("JEV_API_KEY is not set; export it in the environment for this run")
    if args.out.exists():
        sys.exit(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)

    rows, failures, consecutive = [], 0, 0
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            t0 = time.perf_counter()
            try:
                status, resp = call(it["body"], key)
            except PermissionError as e:
                sys.exit(str(e))
            ms = round((time.perf_counter() - t0) * 1000, 1)
            ans = (resp.get("answers") or {}).get("industry") if status == 200 else None
            if ans is None:
                failures += 1; consecutive += 1
                row = {"id": it["id"], "gold": it["gold"], "error": resp.get("error") or resp, "status": status, "ms": ms}
            else:
                consecutive = 0
                pred, p_gold, nll, brier = score(it["gold"], ans, it["body"]["questions"]["industry"]["criteria"].keys())
                row = {"id": it["id"], "gold": it["gold"], "pred": pred, "correct": pred == it["gold"],
                       "p_gold": p_gold, "nll": nll, "brier": brier, "confidence": ans.get("confidence"),
                       "probabilities": ans.get("probabilities"), "status": status, "ms": ms,
                       "usage": resp.get("usage")}
                rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if n % 25 == 0:
                print(json.dumps({"done": n, "of": len(items), "failures": failures}), flush=True)
            if consecutive >= args.max_failures:
                print(json.dumps({"stopped": "too many consecutive failures", "at": n, "last": row.get("error")}, ensure_ascii=False), flush=True)
                break

    scored = [r for r in rows]
    acc = sum(r["correct"] for r in scored) / len(scored) if scored else float("nan")
    ooc = [r for r in scored if r["gold"] == OOC]
    summary = {"model": args.model, "records": len(items), "scored": len(scored), "failures": failures,
               "accuracy": acc, "accuracy_ooc": (sum(r["correct"] for r in ooc) / len(ooc)) if ooc else None,
               "n_ooc": len(ooc),
               "nll": (sum(r["nll"] for r in scored if r["nll"] is not None) / max(1, sum(1 for r in scored if r["nll"] is not None))),
               "brier": (sum(r["brier"] for r in scored if r["brier"] is not None) / max(1, sum(1 for r in scored if r["brier"] is not None))),
               "mean_ms": (sum(r["ms"] for r in scored) / len(scored)) if scored else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
