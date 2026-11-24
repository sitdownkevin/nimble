"""Unit tests for P1 candidate order preservation, pair-synchronized dynamic shuffling, and FP32 candidate head."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from nimble.training.schema_data import as_scoring, encode_scoring, is_ordered_or_boolean, read_rows


DATA_DIR = Path(__file__).parents[1] / "data"


class FakeTokenizer:
    all_special_ids = set()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]["content"] + "\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        # Single uppercase letter appended at the end maps to unique token ID 1000+ord(c)
        if text.endswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) and text[:-1].endswith("ASSISTANT:"):
            return self.encode(text[:-1]) + [1000 + ord(text[-1])]
        return [ord(ch) % 256 for ch in text[:64]] + [999]


class P1RecipeTests(unittest.TestCase):
    def test_preserve_ordered_kinds_and_sync_pair_shuffle(self):
        rows = read_rows(DATA_DIR / "train.jsonl")
        tok = FakeTokenizer()

        # Score stays strictly ascending 0..K-1
        score_row = next(r for r in rows if r["input"]["questions"]["decision"]["type"] == "score")
        enc_score = encode_scoring(as_scoring(score_row, True), tok, 4096, shuffle_seed=17, preserve_ordered_kinds=True)
        n_levels = len(score_row["input"]["questions"]["decision"]["criteria"])
        self.assertEqual(list(enc_score["code_to_choice"].values()), [str(i) for i in range(n_levels)])

        # Noul stays strictly [False, True]
        noul_row = next(r for r in rows if r["input"]["questions"]["decision"]["type"] == "noul")
        enc_noul = encode_scoring(as_scoring(noul_row, True), tok, 4096, shuffle_seed=17, preserve_ordered_kinds=True)
        self.assertEqual(list(enc_noul["code_to_choice"].values()), [False, True])

        # Choice pairs share identical choice order under sync_pair_shuffle=True, and change across epochs
        choice_base = next(r for r in rows if r["input"]["questions"]["decision"]["type"] == "choice" and r["variant"] == "base")
        choice_cf = next(r for r in rows if r["family"] == choice_base["family"] and r["variant"] == "counterfactual")
        e0_b = encode_scoring(as_scoring(choice_base, True), tok, 4096, shuffle_seed="17:ep0",
                              preserve_ordered_kinds=True, sync_pair_shuffle=True)
        e0_c = encode_scoring(as_scoring(choice_cf, True), tok, 4096, shuffle_seed="17:ep0",
                              preserve_ordered_kinds=True, sync_pair_shuffle=True)
        self.assertEqual(e0_b["code_to_choice"], e0_c["code_to_choice"])
        self.assertNotEqual(e0_b["target_index"], e0_c["target_index"])

        orders = {
            tuple(encode_scoring(as_scoring(choice_base, True), tok, 4096, shuffle_seed=f"17:ep{ep}",
                                 preserve_ordered_kinds=True, sync_pair_shuffle=True)["code_to_choice"].values())
            for ep in range(5)
        }
        self.assertGreater(len(orders), 1)

    def test_is_ordered_or_boolean_helper(self):
        self.assertTrue(is_ordered_or_boolean({"type": "boolean", "choices": [False, True]}))
        self.assertTrue(is_ordered_or_boolean({"type": "enum", "choices": ["0", "1", "2"]}))
        self.assertFalse(is_ordered_or_boolean({"type": "enum", "choices": ["a", "b", "c"]}))


class P1FP32CandidateHeadTests(unittest.TestCase):
    def test_fp32_candidate_head_matches_full_vocab_and_backprops(self):
        try:
            import torch
            from nimble.training.schema_train import CandidateCollator, candidate_logits
        except ImportError:
            self.skipTest("torch/transformers not installed in this local python")

        torch.manual_seed(0)
        d, v = 16, 32
        head = torch.nn.Linear(d, v, bias=False)
        h_in = torch.randn(2, 3, d, requires_grad=True)

        class FakeBackbone(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache=False):
                return SimpleNamespace(last_hidden_state=h_in)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(language_model=FakeBackbone())
                self.lm_head = head

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=1):
                h = self.model.language_model(input_ids, attention_mask, use_cache=use_cache).last_hidden_state
                return SimpleNamespace(logits=self.lm_head(h[:, -1:, :]))

        model = FakeModel()
        rows = [
            {"input_ids": [1, 2, 3], "candidate_ids": [2, 5, 7], "labels": 1},
            {"input_ids": [4, 5], "candidate_ids": [1, 9], "labels": 0},
        ]
        batch = CandidateCollator(0)(rows)
        out_full = candidate_logits(model, batch, fp32_candidate_head=False)
        out_cand = candidate_logits(model, batch, fp32_candidate_head=True)
        mask = batch["candidate_mask"]
        self.assertTrue(torch.allclose(out_full[mask], out_cand[mask], atol=1e-5))
        loss = torch.nn.functional.cross_entropy(out_cand, batch["labels"])
        loss.backward()
        self.assertGreater(h_in.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
