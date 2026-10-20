"""Offline FR/EN read evaluation. Requires a synthetic catalog and captured audio."""
import asyncio,json,time,sys
from pathlib import Path
import httpx
from scripts.e2e.conversation_bench import build_cases,measure
from scripts.e2e.queue_bench import scenario


def read_cases():
    pairs=[
      ('caps','Quelles commandes pour la lampe canapé dans le salon essai ?','What controls are available for Lampe Canapé in Salon Essai?','caps','fixture1'),
      ('caps_synonym','Dis-moi ce que permet la lampe divan dans le séjour essai.','What can the couch lamp in the sitting room essai do?','caps','fixture1'),
      ('state','La lampe canapé dans le salon essai est-elle actuellement allumée ?','Is Lampe Canapé in Salon Essai switched on right now?','state','fixture1'),
      ('state_synonym','Quel est le statut de la lampe divan dans le séjour essai ?','Read the state of the sofa lamp in the living room essai.','state','fixture1'),
      ('duplicate','Quel est l’état de la lampe canapé ?','What is the state of Lampe Canapé?','clarify',None),
      ('generic','Quelles capacités pour les lampes ?','What are the lamp capabilities?','clarify',None),
      ('unknown_room','Quel est l’état de la lampe canapé dans la salle inconnue ?','What is the state of Lampe Canapé in the nonexistent room?','missing',None),
      ('unknown_device','Quelles commandes pour la lampe introuvable dans le salon essai ?','What controls are available for the nonexistent lamp in Salon Essai?','missing',None),
      ('multiple','Quel est l’état de Lampe Canapé et Lampe Lecture ?','What is the state of Lampe Canapé and Lampe Lecture?','clarify',None),
      ('negation','Ne lis pas l’état de Lampe Lecture.','Do not read the state of Lampe Lecture.','no_read',None),
      ('ref_start','Quelles fonctionnalités propose Lampe Lecture ?','What functions does Lampe Lecture support?','caps','fixture3'),
      ('ref_follow','Et son état ?','And its state?','state','fixture3'),
      ('unavailable','Quel est l’état de la lampe canapé dans la cuisine essai ?','What is the state of Lampe Canapé in Cuisine Essai?','unavailable','fixture2'),
      ('period','Quel sera l’état de Lampe Lecture demain ?','What will the state of Lampe Lecture be tomorrow?','period',None),
    ]
    rows=[]
    for language in ('fr','en'):
        for name,fr,en,expected,device in pairs:
            text=fr if language=='fr' else en
            rows.append(dict(id=language+'-read-'+name,language=language,kind='read_eval',session=language+'-'+('reference' if name.startswith('ref_') else name),text=text if name=='ref_follow' else 'Aria, '+text,answer=None,expected_commands=[],allow_search=True,expected=expected,target=device))
    return rows


async def run(remote,output,checks_only=False,controls_only=False):
    remote=Path(remote);socket=str(remote/'aria.sock');data={'cases':[],'queues':[]}
    def save():Path(output).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),base_url='http://localhost',timeout=300,trust_env=False) as client:
        deadline=time.monotonic()+300
        while True:
            try:
                cfg=(await client.get('/_live/config')).json()
                if cfg.get('warmup_ready'):break
            except (httpx.HTTPError,ValueError):pass
            assert time.monotonic()<deadline,'startup timeout'
            await asyncio.sleep(1)
        assert cfg['tools_mode']=='log_only' and cfg['history_max_messages']==6
        assert not (await client.get('/_live/playback')).json()['player_connected']
        snapshot=(await client.get('/api/inventory/snapshot')).json()
        assert snapshot['devices']['fixture1']['name']=='Lampe Canapé','Only run against isolated fixture catalog'
        data['config']=cfg;data['inventory']=snapshot
        control_snapshot={**snapshot,'devices':{k:v for k,v in snapshot['devices'].items() if not k.startswith('fixture')}}
        general=build_cases(control_snapshot)
        activated=set()
        for case in general:
            if case['session'] not in activated:
                case['text']='Aria, '+case['text']
                activated.add(case['session'])
        cases=read_cases()+general
        for lang,text in [('fr','Pourquoi le ciel est-il bleu ? Explique en une phrase simple.'),('en','Why is the sky blue? Explain in one simple sentence.'),('fr','Pourquoi y a-t-il des saisons sur Terre ? Une phrase simple.'),('en','Why does Earth have seasons? One simple sentence.')]:
            cases.append(dict(id=lang+'-science-'+str(len(cases)),language=lang,kind='science',session=lang+'-science-'+str(len(cases)),text='Aria, '+text,answer=None,expected_commands=[],allow_search=False))
        if checks_only:
            cases=list(general)
            # Same-session transitions expose deferred prefill cost after local reads.
            for lang in ('fr','en'):
                texts = (['Aria, quelles commandes pour Lampe Lecture ?', 'Quel est son état ?',
                          'Combien font douze fois sept ? Réponds seulement par le nombre.'] if lang=='fr' else
                         ['Aria, what controls are available for Lampe Lecture?', 'What is its state?',
                          'What is twelve times seven? Answer with only the number.'])
                for i,text in enumerate(texts):
                    cases.append(dict(id=lang+'-mixed-'+str(i),language=lang,kind='mixed',session=lang+'-mixed',
                                      text=text,answer='84' if i==2 else None,expected_commands=[],allow_search=i<2))
            # Repeat four representative reads without concurrent unit checks.
            cases += [dict(c,id=c['id']+'-timing',session=c['session']+'-timing')
                      for c in read_cases() if c['id'].endswith(('-caps_synonym','-state_synonym'))]
        if controls_only:
            cases=[c for c in general if c['kind'] in ('on','off_followup')]
        for case in cases:
            while (await client.get('/_live/playback')).json()['input_blocked']:await asyncio.sleep(.05)
            row=await measure(client,socket,case,'read-bench');data['cases'].append(row);save()
            print(case['id'],row['flags'],round(row['latency']['accepted_to_first_audio_ready_s'] or 0,3),flush=True)
        if checks_only:
            for lang in ('fr','en'):
                for repetition in range(2):
                    while (await client.get('/_live/playback')).json()['input_blocked']:await asyncio.sleep(.05)
                    data['queues'].append(await scenario(client,lang,repetition));save()
        data['final_playback']=(await client.get('/_live/playback')).json();save()


if __name__=='__main__':asyncio.run(run(sys.argv[1],sys.argv[2],checks_only="--checks-only" in sys.argv[3:],controls_only="--controls-only" in sys.argv[3:]))
