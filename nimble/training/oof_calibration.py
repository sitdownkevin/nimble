"""Pure-Python 5-Fold OOF aggregation, 3-kind temperature fitting, locked holdout calibration, and logit blending."""

import json
import math
from pathlib import Path


def expected_calibration_error(rows, bins=10):
    buckets = [[] for _ in range(bins)]
    for r in rows:
        p = max(r["probabilities"].values())
        idx = min(bins - 1, int(p * bins))
        buckets[idx].append((p, 1.0 if r["correct"] else 0.0))
    n = len(rows)
    return sum((len(b) / n) * abs(sum(x[0] for x in b) / len(b) - sum(x[1] for x in b) / len(b)) for b in buckets if b)


def summarize_rows(rows):
    summary = {}
    for kind in ("all", "choice", "noul", "score"):
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not sel:
            continue
        g = {"count": len(sel), "correct": sum(1 for r in sel if r["correct"])}
        g["accuracy"] = g["correct"] / len(sel)
        if all("probabilities" in r for r in sel):
            g["ece"] = expected_calibration_error(sel)
        for m in ("nll", "brier", "score_absolute_error"):
            if all(m in r for r in sel):
                g[m] = sum(r[m] for r in sel) / len(sel)
        summary[kind] = g
    if all(r.get("pair_id") for r in rows):
        pairs = {}
        for r in rows:
            pairs.setdefault(r["pair_id"], []).append(r)
        if pairs and all(len(v) == 2 for v in pairs.values()):
            both = sum(1 for v in pairs.values() if all(x["correct"] for x in v))
            summary["pairs"] = {"count": len(pairs), "both_correct": both, "accuracy": both / len(pairs)}
    return summary


def _gold_key(row):
    return str(row["target"]).lower() if row["kind"] == "noul" else str(row["target"])


def _row_nll_at_temp(row, temp):
    gold = _gold_key(row)
    vals = [v / temp for v in row["logits"].values()]
    m = max(vals)
    lse = m + math.log(sum(math.exp(v - m) for v in vals))
    return -(row["logits"][gold] / temp - lse)


def fit_temperature_1d(rows, t_min=0.25, t_max=10.0, iters=50):
    if not rows:
        return 1.0
    a, b = math.log(t_min), math.log(t_max)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)

    def obj(log_t):
        t = math.exp(log_t)
        return sum(_row_nll_at_temp(r, t) for r in rows) / len(rows)

    fc, fd = obj(c), obj(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = obj(d)
    return math.exp(0.5 * (a + b))


def fit_kind_temperatures(rows):
    return {
        kind: fit_temperature_1d([r for r in rows if r["kind"] == kind])
        for kind in ("choice", "noul", "score")
    }


def apply_locked_temperatures(rows, temps):
    """Apply pre-fitted 3-kind temperatures to rows without refitting."""
    out = []
    for r in rows:
        t = float(temps.get(r["kind"], 1.0))
        keys = list(r["logits"].keys())
        scaled = [r["logits"][k] / t for k in keys]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        z = sum(exps)
        probs = {k: e / z for k, e in zip(keys, exps)}
        gold = _gold_key(r)
        best = max(probs, key=probs.get)
        nr = dict(r)
        nr["probabilities"] = probs
        nr["correct"] = (best == gold)
        nr["nll"] = -math.log(max(probs[gold], 1e-15))
        nr["brier"] = sum((probs[k] - (1.0 if k == gold else 0.0)) ** 2 for k in keys)
        if r["kind"] == "score":
            exp_s = sum(int(k) * probs[k] for k in keys)
            nr["expected_score"] = exp_s
            nr["score_absolute_error"] = abs(exp_s - int(r["target"]))
        out.append(nr)
    return out


def aggregate_oof_reports(fold_report_paths):
    """Aggregate K fold evaluation JSONs into full OOF summary + nested LOF-calibrated OOF + locked temperatures."""
    per_fold = {}
    all_rows = []
    seen_ids = set()
    for fold_idx, path in sorted(fold_report_paths.items()):
        data = json.loads(Path(path).read_text())
        rows = data["rows"]
        for r in rows:
            if r["id"] in seen_ids:
                raise ValueError(f"Duplicate OOF record ID across folds: {r['id']}")
            seen_ids.add(r["id"])
        per_fold[str(fold_idx)] = {"count": len(rows), "summary": summarize_rows(rows)}
        all_rows.extend([{**r, "_fold": fold_idx} for r in rows])

    # Nested leave-one-fold-out calibration (zero leakage across folds)
    nested_cal_rows = []
    fold_temps = {}
    folds = sorted(fold_report_paths.keys())
    for held in folds:
        fit_r = [r for r in all_rows if r["_fold"] != held]
        ev_r = [r for r in all_rows if r["_fold"] == held]
        ts = fit_kind_temperatures(fit_r)
        fold_temps[str(held)] = ts
        nested_cal_rows.extend(apply_locked_temperatures(ev_r, ts))

    locked_temps = fit_kind_temperatures(all_rows)
    return {
        "num_folds": len(folds),
        "total_oof_rows": len(all_rows),
        "per_fold": per_fold,
        "oof_uncalibrated_summary": summarize_rows(all_rows),
        "oof_nested_calibrated_summary": summarize_rows(nested_cal_rows),
        "nested_fold_temperatures": fold_temps,
        "locked_oof_temperatures": locked_temps,
    }


def blend_centered_reports(reports_list, weights=None):
    """Blend centered candidate logits across multiple evaluation reports sharing identical IDs."""
    if not reports_list:
        raise ValueError("reports_list must be nonempty")
    w = [1.0 / len(reports_list)] * len(reports_list) if weights is None else list(weights)
    w_sum = sum(w)
    w = [x / w_sum for x in w]
    maps = [{r["id"]: r for r in rep["rows"]} for rep in reports_list]
    out = []
    for r0 in reports_list[0]["rows"]:
        rid = r0["id"]
        keys = list(r0["logits"].keys())
        avg_logits = {}
        for k in keys:
            val = 0.0
            for wi, m in zip(w, maps):
                l = m[rid]["logits"]
                ml = sum(l.values()) / len(l)
                val += wi * (l[k] - ml)
            avg_logits[k] = val
        mx = max(avg_logits.values())
        exps = {k: math.exp(avg_logits[k] - mx) for k in keys}
        z = sum(exps.values())
        probs = {k: exps[k] / z for k in keys}
        best = max(probs, key=probs.get)
        gold = _gold_key(r0)
        nr = {
            "id": rid, "family": r0["family"], "pair_id": r0.get("pair_id"),
            "kind": r0["kind"], "target": r0["target"],
            "correct": best == gold,
            "logits": avg_logits, "probabilities": probs,
            "nll": -math.log(max(probs[gold], 1e-15)),
            "brier": sum((probs[k] - (1.0 if k == gold else 0.0)) ** 2 for k in keys),
        }
        if r0["kind"] == "score":
            exp_s = sum(int(k) * probs[k] for k in keys)
            nr["expected_score"] = exp_s
            nr["score_absolute_error"] = abs(exp_s - int(r0["target"]))
        out.append(nr)
    return {"summary": summarize_rows(out), "rows": out}
