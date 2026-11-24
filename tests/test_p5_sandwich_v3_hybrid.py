import json
import tempfile
import unittest
from pathlib import Path

import torch

from nimble.scoring.parallel_schema import prepare_prompts, render_sandwich_v3_content
from nimble.training.hybrid_head import HybridGateHead, load_hybrid_head, save_hybrid_head
from nimble.training.schema_data import as_scoring, encode_scoring, runtime_record
from nimble.training.schema_train import CandidateCollator, DynamicChoiceDataset, _blend_centered_logit_reports


class _CharTokenizer:
    all_special_ids = {0, 1}
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "".join(parts) + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) + 10 for ch in text]

    def decode(self, ids):
        return "".join(chr(i - 10) for i in ids if i >= 10)


class P5SandwichV3HybridTests(unittest.TestCase):
    def test_sandwich_v3_layout_spans_and_left_pad_collator(self):
        tok = _CharTokenizer()
        schema = {
            "decision": {
                "type": "enum",
                "description": "Determine adjustment level using the policy.",
                "choices": ["0", "1", "2"],
                "choice_descriptions": {
                    "0": "No adjustment: $0.",
                    "1": "Minimal adjustment: $0.01 to $24.99.",
                    "2": "Moderate adjustment: $25.00 to $59.99.",
                },
            }
        }
        ctx = json.dumps([{"speaker": "Auditor", "text": "Confirmed $35.00 refund."}])
        prep = prepare_prompts(tok, ctx, schema, max_input_tokens=4096, prompt_format="sandwich_v3")
        self.assertIsNotNone(prep.option_spans)
        spans = prep.option_spans[0]
        self.assertEqual(len(spans), 3)
        full_text = tok.decode(prep.full_ids[0])
        self.assertLess(full_text.index("## Task & Policy"), full_text.index("## Context"))
        self.assertLess(full_text.index("## Context"), full_text.index("## Choices"))
        self.assertLess(full_text.index("## Choices"), full_text.index("Requested field:"))
        # Zero duplication: rubric text appears exactly once (in ## Choices)
        self.assertEqual(full_text.count("Moderate adjustment: $25.00 to $59.99."), 1)
        for idx, (s, e) in enumerate(spans):
            span_txt = tok.decode(prep.full_ids[0][s:e])
            self.assertIn(f"({idx}):", span_txt)

        # Verify CandidateCollator shifts option_spans accurately under left-padding.
        short_row = {"input_ids": prep.full_ids[0], "candidate_ids": prep.candidate_ids[0],
                     "labels": 2, "option_spans": spans}
        long_prep = prepare_prompts(tok, ctx + " " + ("padding " * 10), schema, max_input_tokens=4096, prompt_format="sandwich_v3")
        long_row = {"input_ids": long_prep.full_ids[0], "candidate_ids": long_prep.candidate_ids[0],
                    "labels": 2, "option_spans": long_prep.option_spans[0]}
        batch = CandidateCollator(pad_id=0)([short_row, long_row])
        self.assertIn("option_spans", batch)
        for k in range(3):
            s0, e0 = batch["option_spans"][0, k].tolist()
            decoded = tok.decode(batch["input_ids"][0, s0:e0].tolist())
            self.assertIn(f"({k}):", decoded)

    def test_hybrid_gate_head_step0_identity_grad_flow_and_roundtrip(self):
        torch.manual_seed(7)
        head = HybridGateHead(hidden_size=32, head_dim=16)
        h_seq = torch.randn(2, 20, 32)
        letter = torch.tensor([[1.2, -0.5, 0.3, -float("inf")], [0.4, 0.9, -float("inf"), -float("inf")]])
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        spans = torch.tensor([
            [[2, 5], [6, 10], [11, 15], [0, 0]],
            [[3, 8], [9, 14], [0, 0], [0, 0]],
        ])
        out0 = head(h_seq, letter, spans, mask)
        # Exact identity at step 0 (including under bfloat16 autocast)
        self.assertEqual((out0[mask] - letter[mask]).abs().max().item(), 0.0)
        self.assertTrue(torch.isneginf(out0[~mask]).all())
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out_bf16 = head(h_seq.bfloat16(), letter, spans, mask)
            self.assertEqual(out_bf16.dtype, torch.float32)
            self.assertEqual((out_bf16[mask] - letter[mask]).abs().max().item(), 0.0)

        # Verify gradients flow through gate (step 1) and all pointer weights (step 2)
        opt = torch.optim.SGD(head.parameters(), lr=0.1)
        labels = torch.tensor([0, 1])
        for _ in range(2):
            opt.zero_grad()
            logits = head(h_seq, letter, spans, mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
        self.assertNotEqual(head.gate.item(), 0.0)
        self.assertIsNotNone(head.w_q.weight.grad)
        self.assertGreater(head.w_q.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(head.w_k.weight.grad.abs().sum().item(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            save_hybrid_head(head, Path(tmp))
            reloaded = load_hybrid_head(Path(tmp))
            out_a = head(h_seq, letter, spans, mask)
            out_b = reloaded(h_seq, letter, spans, mask)
            self.assertEqual((out_a[mask] - out_b[mask]).abs().max().item(), 0.0)

    def test_cyclic_shift_and_dual_view_dataset(self):
        tok = _CharTokenizer()
        raw_base = {
            "id": "r1_base", "family": "fam1", "source_family": "dom1",
            "input": {
                "state": "Fact base.",
                "questions": {"decision": {
                    "type": "choice", "instructions": "Pick one.",
                    "criteria": {"alpha": "A desc", "beta": "B desc", "gamma": "C desc"},
                }},
            },
            "reference": {"target": "beta"},
        }
        raw_cf = {
            **raw_base, "id": "r1_cf",
            "input": {**raw_base["input"], "state": "Fact cf."},
            "reference": {"target": "gamma"},
        }
        sc = as_scoring(raw_base, False)
        e0 = encode_scoring(sc, tok, 4096, prompt_format="sandwich_v3", cyclic_shift=0)
        e1 = encode_scoring(sc, tok, 4096, prompt_format="sandwich_v3", cyclic_shift=1)
        self.assertEqual(e0["code_to_choice"]["B"], "beta")
        self.assertEqual(e1["code_to_choice"]["A"], "beta")
        self.assertEqual(e1["target_index"], 0)

        rec_b = runtime_record(e0, raw_base)
        rec_c = runtime_record(encode_scoring(as_scoring(raw_cf, False), tok, 4096, prompt_format="sandwich_v3"), raw_cf)
        ds = DynamicChoiceDataset(
            [rec_b, rec_c], [raw_base, raw_cf], tok, 4096, base_seed=17,
            preserve_ordered_kinds=True, sync_pair_shuffle=True,
            prompt_format="sandwich_v3", dual_view_formats=["v1", "sandwich_v3"],
        )
        # Siblings must always share the same view within an epoch and switch next epoch.
        has_span_ep0_b = "option_spans" in ds[0]
        has_span_ep0_c = "option_spans" in ds[1]
        self.assertEqual(has_span_ep0_b, has_span_ep0_c)
        ds.set_epoch(1)
        has_span_ep1_b = "option_spans" in ds[0]
        has_span_ep1_c = "option_spans" in ds[1]
        self.assertEqual(has_span_ep1_b, has_span_ep1_c)
        self.assertNotEqual(has_span_ep0_b, has_span_ep1_b)

    def test_blend_centered_logit_reports(self):
        rep1 = {"rows": [
            {"id": "1", "family": "f", "pair_id": "p", "kind": "noul", "target": True,
             "logits": {"false": 2.0, "true": 1.0}},
            {"id": "2", "family": "f", "pair_id": "p", "kind": "noul", "target": False,
             "logits": {"false": 4.0, "true": 0.0}},
        ]}
        rep2 = {"rows": [
            {"id": "1", "family": "f", "pair_id": "p", "kind": "noul", "target": True,
             "logits": {"false": 0.0, "true": 3.0}},
            {"id": "2", "family": "f", "pair_id": "p", "kind": "noul", "target": False,
             "logits": {"false": 0.0, "true": 1.0}},
        ]}
        blended = _blend_centered_logit_reports([rep1, rep2])
        self.assertEqual(blended["summary"]["all"]["accuracy"], 1.0)
        self.assertEqual(blended["summary"]["pairs"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
