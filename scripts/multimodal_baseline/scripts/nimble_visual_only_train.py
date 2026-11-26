#!/usr/bin/env python3
"""Run visual-only Nimble experiment with unchanged candidate loss/optimizer and fixed 700/150/150 splits."""
import argparse, json, os, sys
from pathlib import Path
import swanlab
import nimble_mm as M
import nimble_visual_only as V
import nimble_jev_train as J
import nimble.training.schema_train as st

OUT = Path('runs/nimble-0.8b-visual-only')

def prepare(directory, validation_path, tokenizer, max_length, model_id, revision):
    summary = {}
    prepared = {}
    for name, split, train in [('train','train_all_pos_start',True),('validation','validation_pos_start',False),('test','test_pos_start',False)]:
        result, stats = [], []
        for r in M.load_mm_split(split):
            context, schema = M.schema_for(r)
            enc = V.encode(tokenizer, context, schema, row_id=r['id'], train=train)
            if len(enc['prompt_token_ids']) > max_length:
                raise ValueError(f'{split} {r["id"]} length {len(enc["prompt_token_ids"])} exceeds {max_length}')
            assert enc['prompt'].count(V.PAD) == len(enc['grid'])
            assert r['label'] in enc['choices']
            assert all(str(v) not in enc['prompt'] for v in enc['choices'])
            result.append(M.runtime_row(enc, r['id'], r['id'], 'choice', r['label']))
            stats.append({'id':r['id'], 'pages':len(enc['grid']), 'image_tokens':enc['image_tokens'],
                          'sequence_tokens':len(enc['prompt_token_ids']), 'sizes':enc['sizes'], 'lines':enc['lines']})
        prepared[name] = result
        summary[name] = stats
        print(name, len(stats), 'max_length',max(s['sequence_tokens'] for s in stats), 'max_images',max(s['pages'] for s in stats),flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'render_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1))
    J.STATE.update(val=prepared['validation'],test=prepared['test'],collator=V.Collator(tokenizer.pad_token_id))
    return prepared['train'],prepared['validation'],{'source':'visual-only 20px text; no original advertisement image',
        'training_rows':len(prepared['train']),'validation_rows':len(prepared['validation']),
        'test_rows':len(prepared['test']),'train_max_tokens':max(s['sequence_tokens'] for s in summary['train']),
        'test_split_used':'once after training'}

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default=str(OUT))
    ap.add_argument('--max-length',type=int,default=2048)
    ap.add_argument('--max-steps',type=int,default=264)
    ap.add_argument('--eval-every',type=int,default=20)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    OUT=Path(a.output_dir)
    if a.smoke:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B-Base',revision='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
        r=M.load_mm_split('train_all_pos_start')[0]
        enc=V.encode(tok,*M.schema_for(r),row_id=r['id'],train=True)
        row=M.runtime_row(enc,r['id'],r['id'],'choice',r['label'])
        print('smoke',{'pages':len(enc['grid']),'sizes':enc['sizes'],'tokens':enc['image_tokens'],'length':len(enc['prompt_token_ids']),'label':row['labels'],'candidate':tok.decode([row['candidate_ids'][row['labels']]])},flush=True)
        from peft import get_peft_model,LoraConfig
        model=st.load_base('Qwen/Qwen3.5-0.8B-Base','dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
        batch={k:v.cuda() for k,v in V.Collator(tok.pad_token_id)([row]).items()}
        with __import__('torch').autocast('cuda',dtype=__import__('torch').bfloat16):
            logits=M.candidate_logits_mm(model,batch)
        print('forward',logits.shape,logits[0,row['labels']].item(),flush=True)
        return
    st.prepare_data=prepare
    M.install(st)
    st.CandidateCollator=V.Collator
    J.JevCandidateTrainer.EVAL_EVERY,J.JevCandidateTrainer.OUT=a.eval_every,a.output_dir
    st.CandidateTrainer=J.JevCandidateTrainer
    key=os.environ.get('SWANLAB_API_KEY')
    if key: swanlab.login(api_key=key,save=False)
    swanlab.init(project='jev-real-ad-1000',experiment_name='nimble-0.8b-visual-only-700',
                 mode='online' if key else 'offline',
                 config={'modality':'rendered image only','font_px':20,'max_steps':a.max_steps,'max_length':a.max_length,
                         'lr':1e-4,'lora_rank':16,'batch_size':1,'accumulation':8,'seed':17,'eval_every':a.eval_every})
    sys.argv=['schema_train','train','--model','Qwen/Qwen3.5-0.8B-Base','--revision','dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
       '--data-dir',str(M.DATA),'--validation',str(M.DATA/'validation_pos_start.jsonl'),'--output-dir',a.output_dir,
       '--learning-rate','0.0001','--lora-rank','16','--seed','17','--batch-size','1','--gradient-accumulation','8',
       '--max-length',str(a.max_length),'--max-steps',str(a.max_steps),'--warmup-steps','10','--save-steps','88']
    try: st.main()
    finally: swanlab.finish()

if __name__=='__main__': main()
