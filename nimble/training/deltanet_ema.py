"""Gated DeltaNet dynamics unlocking, parameter EMA, and LODO 3-kind temperature calibration."""

import contextlib
import math
from pathlib import Path

import torch
from transformers import TrainerCallback

DELTANET_EXTRA_FILENAME = "deltanet_extra.pt"

_DELTANET_UNLOCK_SUFFIXES = (
    ".A_log",
    ".dt_bias",
    ".conv1d.weight",
    ".conv1d.bias",
    ".q_norm.weight",
    ".k_norm.weight",
    ".linear_attn.norm.weight",
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "language_model.norm.weight",
)


def is_deltanet_unlock_param(name):
    """Match non-LoRA Gated DeltaNet decay/conv and RMSNorm parameters inside language_model."""
    if "lora_" in name or "visual" in name or "embed_tokens" in name or "lm_head" in name:
        return False
    if "language_model" not in name:
        return False
    return any(name.endswith(sfx) for sfx in _DELTANET_UNLOCK_SUFFIXES)


def unlock_deltanet_parameters(model):
    """Unfreeze Gated DeltaNet dynamic tensors (A_log, dt_bias, conv1d, norms) inside language_model."""
    unlocked = []
    total_scalars = 0
    for name, param in model.named_parameters():
        if is_deltanet_unlock_param(name):
            param.requires_grad = True
            unlocked.append(name)
            total_scalars += param.numel()
    return unlocked, total_scalars


def collect_extra_trainable_state(model):
    """Collect all trainable non-LoRA tensors so they persist cleanly alongside PEFT adapters."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            state[name] = param.detach().cpu().clone()
    return state


def save_extra_trainable_params(model, directory):
    state = collect_extra_trainable_state(model)
    if not state:
        return None
    path = Path(directory) / DELTANET_EXTRA_FILENAME
    torch.save(state, path)
    return path


def load_extra_trainable_params(model, directory, strict=True):
    path = Path(directory) / DELTANET_EXTRA_FILENAME
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    param_map = dict(model.named_parameters())
    loaded = 0
    for name, tensor in state.items():
        if name not in param_map:
            # Handle optional PEFT prefix differences if any.
            alt = name.replace("base_model.model.", "", 1) if name.startswith("base_model.model.") else f"base_model.model.{name}"
            if alt in param_map:
                name = alt
            elif strict:
                raise KeyError(f"Saved extra parameter not found in model: {name}")
            else:
                continue
        with torch.no_grad():
            param_map[name].copy_(tensor.to(device=param_map[name].device, dtype=param_map[name].dtype))
        loaded += 1
    return loaded


class ParameterEMA:
    """FP32 exponential moving average over trainable parameters (LoRA + unlocked DeltaNet)."""

    def __init__(self, model, decay=0.995, warmup_steps=0):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.shadow = {
            name: param.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model, global_step):
        if global_step <= self.warmup_steps:
            # Keep shadow synced during warmup so EMA starts smoothly at post-warmup weights.
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].copy_(param.detach().float().cpu())
            return
        self.num_updates += 1
        eff_decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        one_minus = 1.0 - eff_decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(eff_decay).add_(param.detach().float().cpu(), alpha=one_minus)

    @contextlib.contextmanager
    def apply_to(self, model):
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().cpu().clone()
                    param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name].to(device=param.device, dtype=param.dtype))


class EMACallback(TrainerCallback):
    def __init__(self, ema):
        self.ema = ema
        self._last_step = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step != self._last_step:
            self.ema.update(model, state.global_step)
            self._last_step = state.global_step
        return control


def _row_nll_at_temp(row, temp):
    logits = row["logits"]
    target = row["target"]
    gold_key = str(target).lower() if row["kind"] == "noul" else str(target)
    vals = [v / temp for v in logits.values()]
    m = max(vals)
    lse = m + math.log(sum(math.exp(v - m) for v in vals))
    return -(logits[gold_key] / temp - lse)


def fit_temperature_1d(rows, t_min=0.25, t_max=8.0, iters=40):
    """Exact golden-section 1D minimization of mean NLL over log-temperature."""
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


def _recalibrate_row(row, temp):
    keys = list(row["logits"].keys())
    scaled = [row["logits"][k] / temp for k in keys]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    probs = {k: e / z for k, e in zip(keys, exps)}
    gold_key = str(row["target"]).lower() if row["kind"] == "noul" else str(row["target"])
    out = dict(row)
    out["probabilities"] = probs
    out["nll"] = -math.log(max(probs[gold_key], 1e-15))
    out["brier"] = sum((probs[k] - (1.0 if k == gold_key else 0.0)) ** 2 for k in keys)
    if row["kind"] == "score":
        exp_score = sum(int(k) * probs[k] for k in keys)
        out["expected_score"] = exp_score
        out["score_absolute_error"] = abs(exp_score - int(row["target"]))
    return out


def evaluate_lodo_temperature_calibration(rows, summarize_fn):
    """Leave-one-domain-out (source_family) 3-kind temperature calibration (zero domain leakage)."""
    families = sorted({r["family"] for r in rows})
    if len(families) < 2:
        return None
    cal_rows = []
    fold_temps = {}
    for held_fam in families:
        fit_rows = [r for r in rows if r["family"] != held_fam]
        eval_rows = [r for r in rows if r["family"] == held_fam]
        temps = {}
        for kind in ("choice", "noul", "score"):
            k_fit = [r for r in fit_rows if r["kind"] == kind]
            temps[kind] = fit_temperature_1d(k_fit) if k_fit else 1.0
        fold_temps[held_fam] = temps
        for r in eval_rows:
            cal_rows.append(_recalibrate_row(r, temps.get(r["kind"], 1.0)))
    global_temps = {
        kind: fit_temperature_1d([r for r in rows if r["kind"] == kind])
        for kind in ("choice", "noul", "score")
    }
    return {
        "method": "leave_one_domain_out_3kind_temperature",
        "domains": families,
        "global_fit_temperatures": global_temps,
        "lodo_fold_temperatures": fold_temps,
        "uncalibrated_summary": summarize_fn(rows),
        "lodo_calibrated_summary": summarize_fn(cal_rows),
    }
