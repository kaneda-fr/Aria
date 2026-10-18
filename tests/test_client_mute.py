"""Mute timing, intent and confirmation rules, without audio hardware or models."""
from types import SimpleNamespace
import pytest
import app.client_sessions as sessions
from app.llm.client_mute import MuteCommands, intent


@pytest.fixture
def registry(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(sessions, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]))
    r = sessions.ClientSessions()
    r.claim('kitchen', 'one'); r.claim('office', 'two')
    return r, clock


@pytest.mark.parametrize('text,expected', [
    ('Mute yourself.', (300,'en')), ('Could you be quiet for ten minutes please?', (600,'en')),
    ('Stop listening to me for 1 hour and 30 minutes.', (5400,'en')),
    ('Go silent for half an hour', (1800,'en')),
    ('Please mute yourself for twenty-five seconds', (25,'en')),
    ('Mets-toi en sourdine.', (300,'fr')), ('Peux-tu te taire pendant dix minutes ?', (600,'fr')),
    ('Tais-toi pendant vingt et une minutes', (1260,'fr')),
    ('Arrête de m’écouter pendant une demi-heure', (1800,'fr')),
    ('Silence pendant un quart d’heure', (900,'fr')),
    ('Passe en mode silencieux pendant 0,5 minute', (30,'fr')),
    ('Mute yourself for 25 hours', (None,'en')), ('Be quiet for zero minutes', (None,'en')),
    ('Mute yourself until tomorrow', (None,'en')), ('Mute yourself for 2', (None,'en')),
    ("Don't mute yourself", None), ('Ne te mets pas en sourdine', None),
    ('What happens if I say mute yourself?', None), ('Explain silence', None),
    ('Turn off the kitchen light', None), ('Pourquoi le ciel est-il bleu ?', None),
])
def test_explicit_intents_and_full_duration_consumption(text, expected):
    assert intent(text) == expected


def test_expiry_is_monotonic_reconnect_safe_and_isolated(registry):
    r, clock = registry
    r.set_muted('kitchen', True)
    r.release('kitchen','one'); r.claim('kitchen','reconnected')
    assert r.is_muted('kitchen') and not r.is_muted('office')
    clock[0] += 299.99; assert r.is_muted('kitchen')
    clock[0] += .01; assert not r.is_muted('kitchen')
    assert r.snapshot()[0]['mute_remaining_s'] == 0
    r.set_muted('kitchen', True); revision=r.mute_revision('kitchen')
    r.set_muted('kitchen', False)
    assert not r.is_muted('kitchen') and r.mute_revision('kitchen') > revision


def test_retry_does_not_extend_deadline_and_muted_client_not_evicted(registry):
    r, clock=registry; r.limit=2
    r.set_muted('kitchen',True); clock[0] += 10
    assert r.set_muted('kitchen',True)['mute_remaining_s'] == 290
    r.release('kitchen','one')
    with pytest.raises(ValueError): r.claim('third','c')
    for bad in (0, -1, 86401, float('nan'), 2.5, True):
        with pytest.raises(ValueError):r.set_muted('office',True,seconds=bad)


@pytest.mark.parametrize('utterance,confirm,seconds,language', [
    ('Mute yourself','yes',300,'en'), ('Mets-toi en sourdine pendant dix minutes','oui',600,'fr'),
    ('Stop listening for 2 seconds','confirm',2,'en'),
    ('Mets-toi en sourdine pendant vingt secondes','Je le confirme.',20,'fr')])
def test_muting_requires_confirmation_and_only_affects_current_client(registry,utterance,confirm,seconds,language):
    r,clock=registry; commands=MuteCommands(clock=lambda:clock[0])
    first=commands.handle(utterance,'client:kitchen','person',r)
    assert first.status=='confirmation_required' and not r.is_muted('kitchen')
    assert 'kitchen' in first.text and first.language==language
    assert commands.handle(confirm,'client:office','person',r) is None
    second=commands.handle(confirm,'client:kitchen','person',r)
    assert second.status=='muted' and r.is_muted('kitchen') and not r.is_muted('office')
    assert r.mute_status('kitchen')['mute_remaining_s']==seconds
    assert commands.handle(confirm,'client:kitchen','person',r) is None


@pytest.mark.parametrize('interruption', ['timeout','speaker','manual','cancel','topic','clear'])
def test_stale_other_speaker_or_cancelled_confirmation_cannot_mute(registry,interruption):
    r,clock=registry; commands=MuteCommands(clock=lambda:clock[0])
    commands.handle('mute yourself','client:kitchen','person',r)
    speaker='person'
    if interruption=='timeout':clock[0]+=30
    elif interruption=='speaker':speaker='other'
    elif interruption=='manual':r.set_muted('kitchen',True);r.set_muted('kitchen',False)
    elif interruption=='cancel':assert commands.handle('no','client:kitchen',speaker,r).status=='cancelled'
    elif interruption=='topic':assert commands.handle('How are you?','client:kitchen',speaker,r) is None
    elif interruption=='clear':commands.forget_session('client:kitchen')
    commands.handle('yes','client:kitchen',speaker,r)
    assert not r.is_muted('kitchen')


def test_unknown_client_and_invalid_duration_never_claim_success(registry):
    r,clock=registry; commands=MuteCommands(clock=lambda:clock[0])
    assert commands.handle('mute yourself','missing','person',r).status=='unavailable'
    assert commands.handle('mute yourself for two days','client:kitchen','person',r).status=='invalid_duration'
    assert commands.handle('yes','client:kitchen','person',r) is None
    assert not r.is_muted('kitchen')


def test_already_muted_request_does_not_claim_a_different_duration(registry):
    r,clock=registry; commands=MuteCommands(clock=lambda:clock[0])
    r.set_muted('kitchen',True)
    result=commands.handle('mute yourself for two seconds','client:kitchen','person',r)
    assert result.status=='already_muted' and 'unchanged' in result.text
    assert r.mute_status('kitchen')['mute_remaining_s']==300
    assert not commands.pending
