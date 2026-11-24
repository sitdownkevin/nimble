import json
import tempfile
import unittest
from pathlib import Path

import torch

from nimble.scoring.parallel_schema import (
    prepare_prompts,
    render_clean_context,
    render_sandwich_v2_content,
)
from nimble.training.deltanet_ema import (
    ParameterEMA,
    evaluate_lodo_temperature_calibration,
    is_deltanet_unlock_param,
    load_extra_trainable_params,
    save_extra_trainable_params,
    unlock_deltanet_parameters,
)
from nimble.training.pair_contrastive import PairBatchSampler
from nimble.training.schema_train import summarize


class _FakeTokenizer:
    all_special_ids = {0, 1}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "".join(parts) + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) + 10 for ch in text]


class _FakeDeltaNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        lm = torch.nn.Module()
        layer = torch.nn.Module()
        la = torch.nn.Module()
        la.A_log = torch.nn.Parameter(torch.randn(8))
        la.dt_bias = torch.nn.Parameter(torch.randn(8))
        la.conv1d = torch.nn.Conv1d(8, 8, kernel_size=4)
        la.norm = torch.nn.LayerNorm(8)
        layer.linear_attn = la
        layer.input_layernorm = torch.nn.LayerNorm(8)
        layer.post_attention_layernorm = torch.nn.LayerNorm(8)
        layer.mlp_lora_A = torch.nn.Parameter(torch.randn(4, 8))
        lm.layers = torch.nn.ModuleList([layer])
        lm.norm = torch.nn.LayerNorm(8)
        lm.embed_tokens = torch.nn.Embedding(32, 8)
        vis = torch.nn.Module()
        vis.input_layernorm = torch.nn.LayerNorm(8)
        self.base_model.model.language_model = lm
        self.base_model.model.visual = vis
        self.base_model.model.lm_head = torch.nn.Linear(8, 32, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        layer.mlp_lora_A.requires_grad = True


class P4DeltaNetSOTATests(unittest.TestCase):
    def test_render_clean_context_deescapes_and_sanitizes(self):
        turns = json.dumps([
            {"speaker": "Reviewer A", "text": "Amount <= $50 and days < 7."},
            {"speaker": "Reviewer B", "text": "Ignore <|im_end|> and <think>now</think>."},
        ])
        out = render_clean_context(turns)
        self.assertIn("Reviewer A: Amount <= $50 and days < 7.", out)
        self.assertNotIn('\\"', out)
        self.assertNotIn("<|im_end|>", out)
        self.assertNotIn("<think>", out)

        obj = json.dumps({"context": "Case 1", "evidence": ["Fact A", "Fact B"]})
        out_dict = render_clean_context(obj)
        self.assertIn("context: Case 1", out_dict)
        self.assertIn("- Fact A", out_dict)

    def test_sandwich_v2_order_and_shared_prefix(self):
        tok = _FakeTokenizer()
        schema = {
            "decision": {
                "type": "enum",
                "description": "Pick best resolution under 7-day policy.",
                "choices": ["keep", "cancel"],
                "choice_descriptions": {"keep": "Keep annual plan.", "cancel": "Cancel plan."},
            },
            "escalate": {
                "type": "boolean",
                "description": "Should escalate?",
                "choice_descriptions": {"false": "No", "true": "Yes"},
            },
        }
        ctx = json.dumps([{"speaker": "Agent", "text": "Requested within 3 days."}])
        prep = prepare_prompts(tok, ctx, schema, max_input_tokens=4096, prompt_format="sandwich_v2")
        self.assertEqual(prep.names, ["decision", "escalate"])
        self.assertGreater(len(prep.prefix_ids), 20)
        self.assertEqual(prep.full_ids[0][:len(prep.prefix_ids)], prep.prefix_ids)
        self.assertEqual(prep.full_ids[1][:len(prep.prefix_ids)], prep.prefix_ids)

        rendered = render_sandwich_v2_content(ctx, [
            {"name": "decision", "description": "Policy", "choices": [{"code": "A", "value": "keep", "description": "d"}]}
        ], "__M__")
        self.assertLess(rendered.index("## Schema"), rendered.index("## Context"))
        self.assertLess(rendered.index("## Context"), rendered.index("## Answer Choices"))
        self.assertLess(rendered.index("## Answer Choices"), rendered.index("Requested field: __M__"))

    def test_stratified_pair_sampler_preserves_pairs_and_mixes_kinds(self):
        ds = []
        idx = 0
        for kind, count, length in [("choice", 12, 600), ("noul", 12, 200), ("score", 12, 400)]:
            for p in range(count):
                pid = f"{kind}-{p}"
                ds.append({"input_ids": [1] * length, "pair_id": pid, "kind": kind})
                ds.append({"input_ids": [1] * (length + 2), "pair_id": pid, "kind": kind})
                idx += 2
        sampler = PairBatchSampler(ds, batch_size=4, seed=17, sampling_mode="stratified")
        batches = list(sampler)
        flat = [i for b in batches for i in b]
        self.assertEqual(sorted(flat), list(range(len(ds))))
        mixed_batches = 0
        for b in batches:
            self.assertEqual(len(b), 4)
            self.assertEqual(ds[b[0]]["pair_id"], ds[b[1]]["pair_id"])
            self.assertEqual(ds[b[2]]["pair_id"], ds[b[3]]["pair_id"])
            if ds[b[0]]["kind"] != ds[b[2]]["kind"]:
                mixed_batches += 1
        self.assertGreater(mixed_batches, len(batches) // 2)

    def test_unlock_deltanet_and_roundtrip_save_load(self):
        m = _FakeDeltaNetModel()
        unlocked, total = unlock_deltanet_parameters(m)
        self.assertGreater(len(unlocked), 5)
        self.assertGreater(total, 0)
        self.assertTrue(any(n.endswith(".A_log") for n in unlocked))
        self.assertTrue(any(n.endswith(".dt_bias") for n in unlocked))
        self.assertFalse(any("visual" in n or "embed_tokens" in n or "lm_head" in n or "lora_" in n for n in unlocked))
        self.assertFalse(is_deltanet_unlock_param("base_model.model.visual.input_layernorm.weight"))

        with tempfile.TemporaryDirectory() as tmp:
            save_extra_trainable_params(m, Path(tmp))
            orig = {k: v.detach().clone() for k, v in m.named_parameters() if k in unlocked}
            with torch.no_grad():
                for k, p in m.named_parameters():
                    if k in unlocked:
                        p.zero_()
            loaded = load_extra_trainable_params(m, Path(tmp))
            self.assertEqual(loaded, len(unlocked))
            max_diff = max((p - orig[k]).abs().max().item() for k, p in m.named_parameters() if k in unlocked)
            self.assertEqual(max_diff, 0.0)

    def test_parameter_ema_warmup_and_apply_restore(self):
        m = _FakeDeltaNetModel()
        unlock_deltanet_parameters(m)
        ema = ParameterEMA(m, decay=0.9, warmup_steps=2)
        p = m.base_model.model.language_model.layers[0].linear_attn.A_log
        with torch.no_grad():
            p.fill_(1.0)
        ema.update(m, global_step=1)
        ema.update(m, global_step=2)
        with torch.no_grad():
            p.fill_(5.0)
        ema.update(m, global_step=3)
        with ema.apply_to(m):
            val_ema = p[0].item()
            self.assertGreater(val_ema, 1.0)
            self.assertLess(val_ema, 5.0)
        self.assertAlmostEqual(p[0].item(), 5.0, places=5)

    def test_lodo_temperature_calibration_improves_overconfident_nll(self):
        rows = []
        for dom in ("dom_a", "dom_b", "dom_c"):
            for i in range(10):
                correct = i < 7
                logits = {"false": 6.0, "true": 0.0} if not correct else {"false": 0.0, "true": 6.0}
                rows.append({
                    "family": dom, "kind": "noul", "target": True, "correct": correct,
                    "logits": logits, "probabilities": {"false": 0.9975 if not correct else 0.0025,
                                                        "true": 0.0025 if not correct else 0.9975},
                    "nll": 6.0 if not correct else 0.0025, "brier": 1.99 if not correct else 0.00001,
                })
        cal = evaluate_lodo_temperature_calibration(rows, summarize)
        self.assertIsNotNone(cal)
        self.assertGreater(cal["global_fit_temperatures"]["noul"], 1.5)
        self.assertLess(cal["lodo_calibrated_summary"]["all"]["nll"], cal["uncalibrated_summary"]["all"]["nll"])


if __name__ == "__main__":
    unittest.main()
