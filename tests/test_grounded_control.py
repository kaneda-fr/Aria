"""Inventory, execution authority, and truthful-outcome regressions; no real I/O."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.llm.control import resolve, render, ControlMemory, Authorization
from app.llm.tools import execute_tool
from app.plugins.inventory.indexer import InventoryIndexer
from app.plugins.inventory.models import DeviceRecord, CommandRecord
from app.plugins.inventory.provider import InventoryProvider


@pytest.fixture
def inventory(tmp_path):
    provider = InventoryProvider('inventory', {'cache_path':str(tmp_path/'inventory.json'), 'sources':{}})
    provider._mark_started()
    for identifier,name,room,domain in [('lamp','Lampe Bureau','Bureau','light'),('other','Lampe Entrée','Entrée','light'),('shutter','Volet Bureau','Bureau','flap')]:
        commands=[]
        for action in (['on','off','slider'] if domain=='light' else ['up','down','stop','slider']):
            slider=action=='slider'
            command=CommandRecord(identifier+'-'+action,identifier,'action','slider' if slider else 'other',domain+'_'+action,None,(0,99) if slider else None,set(),{'args':{'value_required':slider,'value_type':'int' if slider else None,'range':[0,99] if slider else None,'payload_template':'slider' if slider else 'other'}})
            commands.append(command)
        device=DeviceRecord('fixture',identifier,name,room,domain,True,True,commands,set(),0)
        provider.indexer.add_device(device)
    return provider


@pytest.mark.parametrize('text,expected,value', [
    ('Allume la lumière « Lampe Bureau » dans la pièce « Bureau ».','lamp-on',None),
    ('Please turn off the light named "Lampe Bureau" in room "Bureau".','lamp-off',None),
    ('Règle la lumière « Lampe Bureau » dans « Bureau » à 30 pour cent.','lamp-slider',30),
    ('Set the light "Lampe Bureau" in "Bureau" to 30 percent.','lamp-slider',30),
    ('Ferme le volet « Volet Bureau » dans « Bureau ».','shutter-down',None),
    ('Could you close the shutter "Volet Bureau" in "Bureau"?','shutter-down',None),
    ('Turn the light in Bureau off.','lamp-off',None),
    ('Éteins la lumière du Bureau.','lamp-off',None),
])
def test_exact_grounding(inventory,text,expected,value):
    plan=resolve(text,inventory)
    assert plan.status=='ready'
    assert (plan.authorization.cmd_id,plan.authorization.value)==(expected,value)


@pytest.mark.parametrize('text', [
    'Allume la lumière.', 'Turn on the light.',
    'Turn on the light in room Missing.',
    'Allume la lumière de la pièce « ARIA_TEST_ABSENT_ROOM ».',
    'Turn on Lampe Bureau in room Entrée.',
    'Set Lampe Bureau to 130 percent.',
    'Set Lampe Bureau to -30 percent.',
    'Set Lampe Bureau to 30.5 percent.',
    'Set Lampe Bureau to 30 percent tomorrow.',
    'Turn on Lampe Bureau and turn off Lampe Entrée.',
    'Allume Lampe Bureau si je rentre.',
    'Turn on Lampe Bureau unless I cancel.',
    'Turn on Lampe Bureau not Lampe Entrée.',
])
def test_ambiguous_missing_conflicting_or_conditional_cannot_authorize(inventory,text):
    assert resolve(text,inventory).status!='ready'


@pytest.mark.parametrize('text',['What is the capital of Italy?','Nomme les deux outils de fonction dont tu disposes.',"Don't turn on Lampe Bureau",'Explain how to turn on Lampe Bureau','Say "turn on Lampe Bureau"'])
def test_conversation_cannot_authorize(inventory,text):
    plan = resolve(text,inventory)
    assert plan is None or plan.authorization is None


def test_inventory_lookup_normalizes_accents_exact_names_and_room_filter(inventory):
    ids=lambda text,ctx=None: {x['id'] for x in inventory.get_candidates(text,ctx)['candidates']}
    assert ids('Lampe Entree')=={'other-on','other-off','other-slider'}
    assert ids('lumière',{'room':'Entree'})=={'other-on','other-off','other-slider'}
    assert not ids('lumiere',{'room':'Missing'})
    assert not ids('Lampe Bureau',{'room':'Entree'})
    assert not ids('light MissingRoom')
    assert ids('Lampe Bureau',{'capability':'light_off'})=={'lamp-off'}
    inventory.indexer.get_device('lamp').enabled=False
    assert not ids('Lampe Bureau')


def test_followup_requires_same_session_speaker_recent_device(inventory):
    memory=ControlMemory(limit=2,ttl=120)
    memory.remember(('s','one'),'lamp')
    assert resolve('Now turn that same light off.',inventory,memory.get(('s','one'))).authorization.cmd_id=='lamp-off'
    assert resolve('Maintenant, éteins cette même lumière.',inventory,memory.get(('s','one'))).authorization.cmd_id=='lamp-off'
    assert resolve('Now turn that same light off.',inventory,memory.get(('other','one'))).status=='no_reference'
    assert memory.get(('s','two')) is None
    memory.ttl=-1
    assert memory.get(('s','one')) is None
    memory.remember(('2','one'),'other'); memory.remember(('3','one'),'other'); memory.remember(('4','one'),'other')
    assert len(memory.items)==2


@pytest.mark.asyncio
async def test_dispatch_requires_single_use_grant_and_current_matching_inventory(inventory):
    actions=SimpleNamespace(is_started=True,execute_command=Mock(return_value={'success':True,'dry_run':True,'executed':False}))
    async def execute(args,grant=None):
        return await execute_tool('execute_command',args,inventory,actions,grant)
    assert not (await execute({'cmd_id':'lamp-on'}))['success']
    grant=resolve('Allume Lampe Bureau',inventory).authorization
    assert not (await execute({'cmd_id':'lamp-off'},grant))['success']
    assert not (await execute({'cmd_id':'lamp-on'},grant))['success']
    actions.execute_command.assert_not_called()
    grant=resolve('Allume Lampe Bureau',inventory).authorization
    result=await execute({'cmd_id':'lamp-on'},grant)
    assert result['outcome']=='recorded'
    assert not (await execute({'cmd_id':'lamp-on'},grant))['success']
    actions.execute_command.assert_called_once()
    grant=resolve('Allume Lampe Bureau',inventory).authorization
    inventory.indexer.get_device('lamp').enabled=False
    assert not (await execute({'cmd_id':'lamp-on'},grant))['success']
    actions.execute_command.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['unknown','spec_changed','missing_spec','range','bool','missing_value'])
async def test_invalid_command_or_payload_never_reaches_provider(inventory,mutation):
    actions=SimpleNamespace(is_started=True,execute_command=Mock())
    plan=resolve('Set Lampe Bureau to 30 percent',inventory)
    grant=plan.authorization
    args={'cmd_id':grant.cmd_id,'value':30}
    if mutation=='unknown':args['cmd_id']='invented'
    if mutation=='spec_changed':inventory.indexer.get_command(grant.cmd_id).execution['args']['payload_template']='other'
    if mutation=='missing_spec':inventory.indexer.get_command(grant.cmd_id).execution=None
    if mutation=='range':grant.value=args['value']=100
    if mutation=='bool':grant.value=args['value']=True
    if mutation=='missing_value':args.pop('value')
    result=await execute_tool('execute_command',args,inventory,actions,grant)
    assert not result['success']
    actions.execute_command.assert_not_called()


@pytest.mark.parametrize('text', ['Allume Lampe Bureau','Turn on Lampe Bureau'])
def test_reply_never_promotes_recording_or_publication_to_completion(inventory,text):
    plan=resolve(text,inventory)
    dry=render(plan,{'success':True,'dry_run':True,'message':'The device changed!'})
    submitted=render(plan,{'success':True,'message':'The device changed!'})
    failed=render(plan,{'success':False,'message':'Done!'})
    assert ('aucun appareil modifié' in dry or 'no device changed' in dry)
    assert ('non confirmé' in submitted or 'not confirmed' in submitted)
    assert ('refusée' in failed or 'rejected' in failed)
    assert 'The device changed!' not in dry+submitted


@pytest.mark.parametrize('text', ['Allume Lampe Bureau, éteins Lampe Entrée.', 'Turn on Lampe Bureau then off.', 'Turn on Lampe Bureau at 20:00.'])
def test_multiple_or_scheduled_actions_fail_closed(inventory,text):
    assert resolve(text,inventory).status!='ready'


@pytest.mark.asyncio
async def test_changed_room_invalidates_resolved_authority(inventory):
    actions=SimpleNamespace(is_started=True,execute_command=Mock())
    grant=resolve('Allume Lampe Bureau',inventory).authorization
    inventory.indexer.get_device('lamp').room='Different room'
    result=await execute_tool('execute_command',{'cmd_id':grant.cmd_id},inventory,actions,grant)
    assert result['error']=='inventory_changed'
    actions.execute_command.assert_not_called()


@pytest.mark.parametrize('text,language', [
    ("Don't turn on Lampe Bureau",'en'),
    ('Please do not close Volet Bureau','en'),
    ("N’allume pas Lampe Bureau",'fr'),
    ('Ne pas ouvrir Volet Bureau','fr'),
])
def test_negation_acknowledges_without_model_or_authorization(inventory,text,language):
    plan=resolve(text,inventory)
    assert plan.status=='no_action' and plan.authorization is None and plan.language==language
    assert ('aucune commande' in render(plan) if language=='fr' else 'no command' in render(plan))
