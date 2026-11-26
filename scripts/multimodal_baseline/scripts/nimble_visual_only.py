"""Visual-only candidate classification: complete state, question and A-Z options in 20px images."""
import json, random, string, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import nimble_mm as M
from kev import vision as V
from nimble.scoring.parallel_schema import choice_key, choices_for

FONT = '/root/.trae-server/bin/stable-11561ecda36d39c2469f5df9b26bc8ecec59006a-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf'
WIDTH, HEIGHT, PX, MARGIN, LINE_HEIGHT = 768, 768, 20, 12, 26
SYSTEM = 'Select the answer letter shown in the image. Reply with one letter.'
PAD = '<|image_pad|>'
BLOCK = '<|vision_start|>' + PAD + '<|vision_end|>'


def render(context, schema, shuffle_seed=None, row_id=''):
    name, field = next(iter(schema.items()))
    choices = list(choices_for(field))
    if shuffle_seed is not None:
        random.Random(f'{shuffle_seed}:{row_id}').shuffle(choices)
    font = ImageFont.truetype(FONT, PX)
    # Preserve every Unicode character and all source fields, including empty values.
    state = context if isinstance(context, dict) else {'state': context}
    paragraphs = ['STATE'] + [f'{k}: {json.dumps(v, ensure_ascii=False)}' for k, v in state.items() if k != 'image']
    paragraphs += [f'QUESTION {name}: {field["description"]}', 'OPTIONS']
    paragraphs += [f'{code}. {value} — {field.get("choice_descriptions", {}).get(choice_key(value), "")}'
                   for code, value in zip(string.ascii_uppercase, choices)]
    lines = []
    for paragraph in paragraphs:
        line = ''
        for ch in paragraph:
            if line and font.getlength(line + ch) > WIDTH - 2 * MARGIN:
                lines.append(line); line = ''
            line += ch
        lines.append(line)
    per_page = (HEIGHT - 2 * MARGIN) // LINE_HEIGHT
    images = []
    for start in range(0, len(lines), per_page):
        page = lines[start:start + per_page]
        height = math.ceil((2 * MARGIN + len(page) * LINE_HEIGHT) / 32) * 32
        img = Image.new('RGB', (WIDTH, height), 'white')
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(page):
            draw.text((MARGIN, MARGIN + i * LINE_HEIGHT), line, font=font, fill='black')
        images.append(img)
    return images, choices, lines


def encode(tok, context, schema, row_id='', train=False, pixels=True):
    images, choices, lines = render(context, schema, 17 if train else None, row_id)
    prompt = tok.apply_chat_template([{'role':'system','content':SYSTEM},
        {'role':'user','content':' '.join([BLOCK] * len(images)) + '\nAnswer:'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    pad_id = tok.convert_tokens_to_ids(PAD)
    assert ids.count(pad_id) == len(images)
    codes = []
    for code in string.ascii_uppercase[:len(choices)]:
        suffix = tok.encode(prompt + code, add_special_tokens=False)
        assert suffix[:len(ids)] == ids and len(suffix) == len(ids) + 1
        codes.append(suffix[-1])
    assert len(set(codes)) == len(codes)
    grids, tensors, out = [], [], []
    iterator = iter(images)
    for token in ids:
        if token != pad_id:
            out.append(token); continue
        img = next(iterator)
        w, h = img.size
        assert w % 32 == h % 32 == 0
        grid = (1, h // 16, w // 16)
        n = V.image_tokens(grid)
        grids.append(grid)
        out.extend([pad_id] * n)
        if pixels:
            x = torch.from_numpy(np.asarray(img, dtype=np.float32).copy() / 255).permute(2, 0, 1)[None]
            tensors.append(V.patchify((x - V.MEAN) / V.STD).to(torch.bfloat16))
    return {'prompt_token_ids':out, 'candidate_token_ids':codes, 'choices':choices,
            'mm':[int(t == pad_id) for t in out], 'pixels':torch.cat(tensors) if pixels else None,
            'grid':grids, 'image_tokens':sum(V.image_tokens(g) for g in grids),
            'sizes':[i.size for i in images], 'lines':len(lines), 'prompt':prompt}


class Collator(M.MMCollator):
    def __call__(self, rows):
        batch = super().__call__([{**r, 'grid':r['grid'][0]} for r in rows])
        batch['image_grid_thw'] = torch.tensor([g for r in rows for g in r['grid']])
        return batch
