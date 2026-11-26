"""Self-contained System One compatible request/response models for Nimble & Kev Playground."""

import json
from typing import Any, Literal, Union
from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "nimble-4b-mm-sota"
    questions: dict[str, Question] = Field(min_length=1)
    calibrated: bool = True
    tta_views: int = 1


class CompareAllRequest(BaseModel):
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)
    calibrated: bool = True
    tta_views: int = 1
    models: list[str] | None = None


def render_flat(v: JSONContent, indent: int = 0) -> str:
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render_flat(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render_flat(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render_flat(x)}"
        for k, x in v.items()
    )


def _normalize(p: list[float]) -> list[float]:
    t = sum(p)
    return [1.0 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p: list[float]) -> float:
    k = len(p)
    return 1.0 if k == 1 else (max(_normalize(p)) - 1.0 / k) / (1.0 - 1.0 / k)


def score_confidence(p: list[float]) -> float:
    n = len(p)
    if n == 1:
        return 1.0
    p = _normalize(p)
    mode = max(range(n), key=p.__getitem__)
    denom = sum(abs(i - (n - 1) / 2.0) for i in range(n)) / n
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / denom)


def round_prob(x: float) -> float:
    return round(float(x), 4)


def format_systemone_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1]),
                            "probabilities": {"false": round_prob(p[0]), "true": round_prob(p[1])}}
        elif m["type"] == "choice":
            dist = {k: round_prob(v) for k, v in zip(m["keys"], p)}
            best_i = max(range(len(p)), key=lambda i: p[i])
            out[m["id"]] = {
                "type": "choice",
                "choice": m["keys"][best_i],
                "confidence": round_prob(choice_confidence(p)),
                "probabilities": dist,
            }
        else:
            score_val = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {
                "type": "score",
                "score": round_prob(score_val),
                "legend": m["legend"],
                "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
                "confidence": round_prob(score_confidence(p)),
            }
    return out


def count_output_tokens(tok, answers: dict) -> int:
    return len(tok(json.dumps(answers, ensure_ascii=False), add_special_tokens=False).input_ids)
