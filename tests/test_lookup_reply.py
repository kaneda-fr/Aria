import pytest

from app.llm.lookup_reply import render_lookup_results


def match(device='one', name='Reading lamp', room='Study', cap='light_on', cmd='private-id'):
    return dict(device_id=device, device_name=name, room=room, capability=cap, cmd_id=cmd)


@pytest.mark.parametrize('lang,on,off,state', [
    ('fr', 'allumer', 'éteindre', 'État actuel inconnu'),
    ('en', 'turn on', 'turn off', 'Current state unknown'),
])
def test_grounded_capabilities_are_not_state_or_actions(lang, on, off, state):
    result = dict(success=True, candidates=[match(), match(cap='light_off'), match()], total_found=3)
    reply = render_lookup_results([result], language=lang)
    assert reply.count('Reading lamp') == 1
    assert on in reply and off in reply and state in reply
    assert 'private-id' not in reply and 'light_on' not in reply
    assert ('aucune action effectuée' if lang == 'fr' else 'no action taken') in reply


def test_same_named_devices_are_distinct_and_searches_deduplicated():
    result = dict(success=True, candidates=[match(), match('two')])
    reply = render_lookup_results([result, result], language='en')
    assert reply.count('Reading lamp') == 2


def test_truncated_results_do_not_claim_exhaustive_devices_or_capabilities():
    result = dict(success=True, candidates=[match()], total_found=8)
    assert 'partial results' in render_lookup_results([result], language='en')
    result = dict(success=True, candidates=[match(str(i), name=f'Lamp {i}') for i in range(7)])
    reply = render_lookup_results([result], language='en')
    assert 'Lamp 2' in reply and 'Lamp 3' not in reply and 'partial results' in reply


def test_unsupported_capability_does_not_speak_internal_identifier():
    reply = render_lookup_results([dict(success=True, candidates=[match(cap='vendor_scene_987')])], language='fr')
    assert 'autres commandes' in reply and 'vendor_scene' not in reply


def test_failure_empty_and_incomplete_results_are_distinct():
    assert 'unavailable' in render_lookup_results([dict(success=False, error='private traceback')], language='en')
    assert 'No device matches this search' in render_lookup_results([dict(success=True, candidates=[])], language='en')
    assert "don't identify" in render_lookup_results([dict(success=True, candidates=[match(device=None)])], language='en')
    reply = render_lookup_results([dict(success=True, candidates=[match()]),
                                   dict(success=True, candidates=[]), dict(success=False)], language='en')
    assert 'One search failed' in reply and 'One search had no matches' in reply


@pytest.mark.parametrize('cap,en,fr', [
    ('light_slider', 'adjust brightness', 'régler la luminosité'),
    ('flap_up', 'open', 'ouvrir'), ('flap_down', 'close', 'fermer'),
    ('flap_stop', 'stop', 'arrêter'), ('flap_slider', 'adjust position', 'régler la position'),
])
def test_provider_capability_vocabulary(cap, en, fr):
    result = dict(success=True, candidates=[match(cap=cap)])
    assert en in render_lookup_results([result], language='en')
    assert fr in render_lookup_results([result], language='fr')
