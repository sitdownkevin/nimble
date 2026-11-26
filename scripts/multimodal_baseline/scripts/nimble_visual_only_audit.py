"""Audit every split without materializing pixel tensors or evaluating the test split."""
import json, statistics
from pathlib import Path
from transformers import AutoTokenizer
import nimble_mm as M
import nimble_visual_only as V

tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B-Base',revision='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
out={}
for name,split,train in [('train','train_all_pos_start',True),('validation','validation_pos_start',False),('test','test_pos_start',False)]:
    rows=[]
    for r in M.load_mm_split(split):
        enc=V.encode(tok,*M.schema_for(r),row_id=r['id'],train=train,pixels=False)
        assert r['label'] in enc['choices'] and len(enc['choices'])==len(enc['candidate_token_ids'])
        assert enc['prompt'].count(V.PAD)==len(enc['grid'])
        assert all(str(v) not in enc['prompt'] for v in enc['choices'])
        assert sum(V.V.image_tokens(g) for g in enc['grid'])==enc['image_tokens']==sum(enc['mm'])
        rows.append({'id':r['id'],'pages':len(enc['grid']),'sizes':enc['sizes'],'image_tokens':enc['image_tokens'],
                     'sequence_tokens':len(enc['prompt_token_ids']),'lines':enc['lines']})
    out[name]=rows
    for key in ('pages','image_tokens','sequence_tokens','lines'):
        a=sorted(row[key] for row in rows)
        print(name,key,'min',a[0],'p50',a[len(a)//2],'p95',a[int(.95*(len(a)-1))],'max',a[-1],flush=True)
Path('runs/nimble-0.8b-visual-only/render_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=1))
