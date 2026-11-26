"""Unit tests for Multimodal Nimble (sandwich_v3, vision modes, collator, FP32 head, calibration)."""

import base64
import io

import torch
from PIL import Image

from nimble.multimodal import (
    IMAGE_BLOCK,
    IMAGE_PAD,
    MMCandidateCollator,
    apply_locked_temperature,
    build_mm_runtime_row,
    cyclic_shifts_for,
    encode_mm_one,
    evaluate_kfold_oof_temperature,
    image_for,
    render_mm_sandwich_v3_content,
    select_lora_targets,
    unlock_vision_merger_parameters,
)
from nimble.training.schema_train import candidate_logits, summarize


class DummyChatTokenizer:
    pad_token_id = 0
    all_special_ids = {0, 1, 2, 3, 4}
    IMAGE_PAD_ID = 4

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        sys_msg = messages[0]["content"]
        usr_msg = messages[1]["content"]
        return f"<|sys|>{sys_msg}<|end|><|usr|>{usr_msg}<|end|><|assistant|>"

    def convert_tokens_to_ids(self, tok):
        if tok == IMAGE_PAD:
            return self.IMAGE_PAD_ID
        raise KeyError(tok)

    def encode(self, text, add_special_tokens=False):
        out = []
        i = 0
        while i < len(text):
            if text.startswith(IMAGE_PAD, i):
                out.append(self.IMAGE_PAD_ID)
                i += len(IMAGE_PAD)
            else:
                out.append(10 + ord(text[i]) % 200)
                i += 1
        # Guarantee single-token distinct codes for A..Z at the very end when text ends with a capital letter
        if text and "A" <= text[-1] <= "Z":
            out[-1] = 300 + (ord(text[-1]) - ord("A"))
        return out


def _png_data_uri(w=256, h=256, color=(40, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_mm_sandwich_v3_layout_and_deescaping():
    fields = [{
        "name": "decision",
        "description": "这条广告属于以下哪个行业类目？",
        "choices": [
            {"code": "A", "value": "汽车零配件零售", "description": "汽车零配件销售"},
            {"code": "B", "value": "以上都不是", "description": "不在候选类目中"},
        ],
    }]
    raw_ctx = '{"title": "燃油宝", "visual_ocr": "强力除积碳"}'
    rendered = render_mm_sandwich_v3_content(raw_ctx, fields, "__TARGET__", has_image=True)
    idx_policy = rendered.index("## Task & Policy")
    idx_ctx = rendered.index("## Context")
    idx_img = rendered.index(IMAGE_BLOCK)
    idx_title = rendered.index("title: 燃油宝")
    idx_choices = rendered.index("## Choices")
    idx_req = rendered.index("Requested field: __TARGET__")
    assert idx_policy < idx_ctx < idx_img < idx_title < idx_choices < idx_req
    assert '\\"' not in rendered


def test_encode_mm_one_and_collator_shapes():
    tok = DummyChatTokenizer()
    schema = {
        "decision": {
            "type": "enum",
            "description": "行业分类",
            "choices": ["餐饮", "服饰", "教育", "以上都不是"],
            "choice_descriptions": {"餐饮": "餐饮服务", "服饰": "服装零售", "教育": "培训", "以上都不是": "其他"},
        }
    }
    img = image_for(_png_data_uri(320, 256))
    enc = encode_mm_one(tok, '{"title": "火锅店"}', schema, image=img, prompt_format="sandwich_v3", max_length=4096)
    assert enc["image_tokens"] > 0
    assert sum(enc["mm"]) == enc["image_tokens"]
    row1 = build_mm_runtime_row(enc, "r1", "r1", "choice", target="服饰")

    enc2 = encode_mm_one(tok, '{"title": "书店长标题测试"}', schema, image=img, prompt_format="sandwich_v3", max_length=4096)
    row2 = build_mm_runtime_row(enc2, "r2", "r2", "choice", target="以上都不是")

    collator = MMCandidateCollator(pad_id=0)
    batch = collator([row1, row2])
    assert batch["input_ids"].shape == batch["mm_token_type_ids"].shape
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["pixel_values"].shape[0] == (row1["grid"][1] * row1["grid"][2] + row2["grid"][1] * row2["grid"][2])
    # Left padding must pad mm_token_type_ids with 0 on the left
    pad_diff = abs(len(row1["input_ids"]) - len(row2["input_ids"]))
    shorter = 0 if len(row1["input_ids"]) < len(row2["input_ids"]) else 1
    if pad_diff > 0:
        assert (batch["mm_token_type_ids"][shorter, :pad_diff] == 0).all()


def test_vision_lora_target_selection_and_merger_unlock():
    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(8, 8)

    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])
            self.merger = torch.nn.ModuleDict({"linear_fc1": torch.nn.Linear(8, 8), "linear_fc2": torch.nn.Linear(8, 8)})

    class FakeInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = FakeVisual()
            self.language_model = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8), "v_proj": torch.nn.Linear(8, 8)})

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeInner()
            self.lm_head = torch.nn.Linear(8, 32, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    m = FakeModel()
    assert len(select_lora_targets(m, "frozen")) == 2
    assert len(select_lora_targets(m, "merger_lora")) == 4
    assert len(select_lora_targets(m, "lora_all")) == 6

    for p in m.parameters():
        p.requires_grad = False
    unlocked, scalars = unlock_vision_merger_parameters(m)
    assert len(unlocked) == 4 and scalars == 2 * (8 * 8 + 8)


def test_fp32_candidate_logits_routes_multimodal_kwargs_to_inner_model():
    captured = {}

    class FakeInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()  # should NOT be called when pixel_values present

        def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
            captured.update(kwargs)
            b, l = input_ids.shape
            h = torch.ones((b, l, 4), dtype=torch.bfloat16)
            return type("Out", (), {"last_hidden_state": h})()

    class FakeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeInner()
            self.lm_head = torch.nn.Linear(4, 16, bias=False).to(torch.bfloat16)

        def get_output_embeddings(self):
            return self.lm_head

    w = FakeWrapper()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "candidate_ids": torch.tensor([[2, 5] + [0] * 24]),
        "candidate_mask": torch.tensor([[True, True] + [False] * 24]),
        "pixel_values": torch.randn(4, 16, dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
    }
    logits = candidate_logits(w, inputs, fp32_candidate_head=True)
    assert logits.dtype == torch.float32
    assert "pixel_values" in captured and "image_grid_thw" in captured and "mm_token_type_ids" in captured
    assert torch.isfinite(logits[0, :2]).all() and torch.isinf(logits[0, 2:]).all()


def test_cyclic_shifts_and_zero_leakage_calibration():
    assert cyclic_shifts_for(26, 5) == [0, 5, 10, 16, 21]
    rows = []
    for i in range(30):
        gold = "A" if i % 3 != 0 else "B"
        # Overconfident logits (scale ~6x)
        logits = {"A": 6.0, "B": 0.0, "C": -3.0} if i % 4 != 0 else {"A": 0.0, "B": 6.0, "C": -3.0}
        exps = {k: torch.exp(torch.tensor(v)).item() for k, v in logits.items()}
        z = sum(exps.values())
        probs = {k: v / z for k, v in exps.items()}
        pred = max(logits, key=logits.get)
        rows.append({
            "id": f"r{i}",
            "family": f"r{i}",
            "kind": "choice",
            "target": gold,
            "prediction": pred,
            "correct": pred == gold,
            "logits": logits,
            "probabilities": probs,
            "nll": -torch.log(torch.tensor(probs[gold])).item(),
            "brier": sum((probs[k] - (1.0 if k == gold else 0.0)) ** 2 for k in logits),
        })
    oof = evaluate_kfold_oof_temperature(rows[:20], summarize, n_folds=5, seed=17)
    assert oof["locked_temperature"] > 1.5
    assert oof["oof_calibrated_summary"]["all"]["nll"] < oof["uncalibrated_summary"]["all"]["nll"]
    test_cal = apply_locked_temperature(rows[20:], oof["locked_temperature"], summarize)
    assert test_cal["calibrated_summary"]["all"]["accuracy"] == test_cal["uncalibrated_summary"]["all"]["accuracy"]
    assert test_cal["calibrated_summary"]["all"]["nll"] < test_cal["uncalibrated_summary"]["all"]["nll"]
