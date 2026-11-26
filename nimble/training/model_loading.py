"""Load the pinned CUDA Qwen checkpoint shared by schema training and evaluation."""

import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration


def load_base(model_id, revision):
    if not torch.cuda.is_available():
        raise RuntimeError("This trainer requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This trainer requires BF16 support")
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=offline)
    if getattr(cfg, "model_type", "") == "qwen3_5":
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=offline,
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=offline,
        ).to("cuda")
    model.config.use_cache = False
    return model
