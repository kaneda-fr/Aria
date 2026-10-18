"""Paired model-only diagnostic on the current isolated Apollo instance, no audio/actions."""
import argparse
import asyncio,json,time,os
from pathlib import Path
import httpx
fixtures=[
    ('fr-sky','Bonjour Aria. Pourquoi le ciel est-il bleu ? Explique en une phrase simple.'),
    ('en-sky','Hello Aria. Why is the sky blue? Explain in one simple sentence.'),
    ('fr-sunset','Et pourquoi devient-il rouge au coucher du soleil ? Une phrase suffit.'),
    ('en-sunset','And why does it turn red at sunset? One sentence is enough.'),
    ('fr-ice','Pourquoi la glace flotte-t-elle sur l’eau ? Une phrase simple.'),
    ('en-ice','Why does ice float on water? One simple sentence.'),
    ('fr-seasons','Pourquoi y a-t-il des saisons sur Terre ? Une phrase simple.'),
    ('en-seasons','Why does Earth have seasons? One simple sentence.'),
    ('fr-moon','La Lune produit-elle sa propre lumière ? Une phrase simple.'),
    ('en-moon','Does the Moon produce its own light? One simple sentence.'),
    ('fr-vacuum','Le son peut-il voyager dans le vide ? Une phrase simple.'),
    ('en-vacuum','Can sound travel through a vacuum? One simple sentence.'),
    ('fr-plant','De quoi une plante a-t-elle besoin pour la photosynthèse ? Une phrase simple.'),
    ('en-plant','What does a plant need for photosynthesis? One simple sentence.'),
    ('fr-uncertain','Quelle est la couleur de mon vélo ? Si tu ne le sais pas, dis-le.'),
    ('en-uncertain','What colour is my bicycle? If you do not know, say so.'),
]
async def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--server-run',type=Path,required=True)
    args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    socket=str(args.server_run/'aria.sock')
    prefix=f'quality-{time.time_ns()}'
    rows=[]
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),base_url='http://localhost',timeout=300,trust_env=False) as c:
        cfg=(await c.get('/_live/config')).json();assert cfg['tools_mode']=='log_only' and cfg['warmup_ready']
        # Each request starts a fresh session. Alternate ordering to expose prefix costs.
        for idx,(cid,text) in enumerate(fixtures):
            for mode in ([True,False] if idx%2==0 else [False,True]):
                cursor=(await c.get('/_e2e/state')).json()['next'];t=time.monotonic()
                r=await c.post('/api/llm/process',json={'text':text,'session_id':f'{prefix}-{cid.replace("sunset","sky")}-{mode}','speak_response':False,'use_tools':mode});r.raise_for_status()
                elapsed=time.monotonic()-t;events=(await c.get('/_e2e/state',params={'after':cursor})).json()['events']
                row=dict(id=cid,tools=mode,text=text,result=r.json(),elapsed=elapsed,events=events)
                assert not any(e['event']=='Live.Command' for e in events)
                rows.append(row);(root/'ablation.json').write_text(json.dumps({'config':cfg,'rows':rows},ensure_ascii=False,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k!='events'},ensure_ascii=False),flush=True)
if __name__ == '__main__':
    asyncio.run(main())
