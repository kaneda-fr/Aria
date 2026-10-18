"""Legacy attribution preserves the nearest user and independent confidence."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.speaker_recognition.recognizer import SpeakerRecognizer


@pytest.mark.parametrize('score,known,user', [(0.6, False, 'fixture'), (0.8, True, 'fixture')])
def test_identity_preserves_candidate_and_confidence(score, known, user):
    recognizer = object.__new__(SpeakerRecognizer)
    recognizer._cfg = SimpleNamespace(enabled=True, min_seconds=1.0, threshold_default=0.65)
    recognizer._maybe_reload = lambda: None
    recognizer._lock = threading.Lock()
    recognizer._profiles = [('fixture_en', np.array([score, (1-score**2)**0.5], dtype=np.float32))]
    recognizer._encoder = SimpleNamespace(encode_batch=lambda _: torch.tensor([[1., 0.]]))
    recognizer._debug = False
    result = recognizer.identify(np.zeros(16000, dtype=np.float32), 16000, 1, now_ts=0)
    assert result.name == 'fixture_en'
    assert result.known is known
    assert result.user == user
    assert result.score == pytest.approx(score)


@pytest.mark.parametrize('name,user,score,known,expected,suppress', [
    ('aria_fr','aria',.126,False,'unknown',False),
    ('aria_en','aria',.70,True,'unknown',False),
    ('aria_en','aria',.80,False,'unknown',False),
    ('aria_en','aria',.80,True,'aria',True),
    ('custom_voice','bot',.80,True,'bot',True),
    ('fixture_fr','fixture',.30,False,'fixture',False),
    ('aria_fr','aria',float('nan'),True,'unknown',False),
])
def test_self_suppression_requires_confidence_without_changing_human_attribution(
        name,user,score,known,expected,suppress):
    from app.speaker_recognition.recognizer import SpeakerResult, self_speech_decision
    result = SpeakerResult(name,score,known,user,.1)
    assert self_speech_decision(result, ('aria_fr','aria_en','custom_voice'), .75) == (expected,suppress)
    assert result.name == name and result.known == known  # Diagnostic evidence stays raw.
