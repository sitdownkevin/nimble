"""Images in a record's state (the `image` key of an object state): decoding, the Qwen2-VL preprocessing the Qwen3.5 base
was trained with, and the 3-D M-RoPE positions of a state.

The preprocessing follows transformers' Qwen2VLImageProcessor for the base's preprocessor_config.json (patch 16, merge 2,
temporal patch 2, mean = std = 0.5; smart_resize to multiples of 32 under a token cap; patchify). transformers imports
torchvision for its processor, which this environment does not have, so the same steps run here in torch. The constants
are checked against the snapshot's config by tests/test_vision.py. Positions come from transformers' own
Qwen3_5Model.get_rope_index. Imported lazily by kev.model: text-only code never loads this module.
"""
import base64, functools, io, math
from functools import partial
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PATCH, MERGE, TEMPORAL = 16, 2, 2
IMAGE_MIN_PIXELS = 65536                        # the base's preprocessor size.shortest_edge
IMAGE_TOKENS = 256                              # merged tokens one image may cost (the image budget of kev.model's states)
IMAGE_MAX_PIXELS = IMAGE_TOKENS * (PATCH * MERGE) ** 2
MEAN = STD = 0.5


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """transformers' Qwen2-VL smart_resize: both sides multiples of `factor`, the pixel count within [min, max], the aspect
    ratio kept as closely as possible."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def load_image(source):
    """RGB PIL image from a `data:image/...;base64,` URI or an image file path; transparency is composited onto white."""
    if source.startswith("data:"):
        header, sep, payload = source.partition(",")
        if not sep or ";base64" not in header or not header.startswith("data:image/"):
            raise ValueError("an image data URI must be data:image/<type>;base64,<payload>")
        with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as img:
            return _rgb(img)
    with Image.open(source) as img:
        return _rgb(img)


def _rgb(img):
    rgba = img.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")


def patchify(images):
    """[B, 3, H, W] normalised pixels -> [B * grid_h * grid_w, 3 * temporal * patch * patch] (transformers' patchify). An image is
    one frame, repeated along the temporal patch."""
    B, C, H, W = images.shape
    gh, gw = H // PATCH, W // PATCH
    p = images.reshape(B, C, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).permute(0, 2, 5, 3, 6, 1, 4, 7)
    p = p.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, TEMPORAL, -1, -1).reshape(B, gh * gw, C * TEMPORAL * PATCH * PATCH)
    return p.reshape(-1, p.shape[-1])


@functools.lru_cache(maxsize=64)
def image_inputs(source):
    """-> (pixel_values [N, 1536] float32, image_grid_thw (1, h, w)) for an image source. The grid is in patches; the image
    costs h * w / merge^2 tokens (at most IMAGE_TOKENS). Cached per source: training re-encodes every record each epoch."""
    img = load_image(source)
    width, height = img.size
    rh, rw = smart_resize(height, width, PATCH * MERGE, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    x = F.interpolate(x, size=(rh, rw), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    pixels = patchify((x - MEAN) / STD).contiguous()
    return pixels, (1, rh // PATCH, rw // PATCH)


def image_tokens(grid):
    """Merged tokens an image of this grid occupies in the state."""
    t, h, w = grid
    return t * h * w // (MERGE * MERGE)


def state_positions(ids, image_token_id, grid):
    """3-D M-RoPE positions (3 lists of len(ids)) of a state whose one image occupies `ids`'s image_token_id run, from
    transformers' Qwen3_5Model.get_rope_index (text tokens: one position in all three rows; image tokens: (t, h, w))."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    shim = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=MERGE)),
                           get_vision_position_ids=partial(Qwen3_5Model.get_vision_position_ids, None))
    input_ids = torch.tensor([ids], dtype=torch.long)
    token_type = (input_ids == image_token_id).long()
    grid_thw = torch.tensor([grid], dtype=torch.long)
    position_ids, _ = Qwen3_5Model.get_rope_index(shim, input_ids, mm_token_type_ids=token_type, image_grid_thw=grid_thw)
    return [row.tolist() for row in position_ids[:, 0]]
