"""Cheap post-ASR activation; never infer intent or authorize a device action."""
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
import asyncio
import logging
import math
import os
import re
import time
import unicodedata

from app.speech_output.speech_output import detect_language

log = logging.getLogger('ARIA.conversation')
WAKE = re.compile(r'^\s*(?:(?P<greeting>bonjour|salut|coucou|hello|hey|hi|hé|ok|okay)[\s,!.:;-]+)?aria(?=$|[\s,!.?:;—-])[\s,!.?:;—-]*', re.I)
ENDINGS = {
    'fr': {'au revoir', 'a bientot', 'a plus tard', 'bonne nuit', 'c est tout',
           'terminons la conversation'},
    'en': {'goodbye', 'good bye', 'bye', 'bye bye', 'see you', 'see you later',
           'good night', 'that s all', 'end our conversation'},
}


def closing_language(text):
    words = ' '.join(re.findall(r'\w+', ''.join(c for c in unicodedata.normalize(
        'NFKD', text.lower()) if not unicodedata.combining(c))))
    words = re.sub(r'^(?:merci|thanks|thank you)\s+', '', words)
    words = re.sub(r'\s+aria$', '', words)
    return next((lang for lang, endings in ENDINGS.items() if words in endings), None)


# Name spelling/pronunciation variants, not a dictionary of request sentences.
# "area" is ambiguous English vocabulary and needs an explicit greeting/address.
SPOKEN_VOCATIVE = re.compile(
    r'^\s*(?:(?P<greeting>bonjour|salut|coucou|hello|hey|hi|hé|ok|okay)[\s,!.:;-]+)?'
    r'(?P<name>ariya|ariah|arya|ari|area)(?=$|[\s,!.?:;—-])(?P<tail>.*)$', re.I)
SPOKEN_SUFFIX = re.compile(r'\b(?:ariya|ariah|arya|ari|area)[.!?\s]*$', re.I)


def normalize_spoken_vocative(text: str) -> str:
    """Repair a voice address only; never rewrite words inside the request body."""
    match = SPOKEN_VOCATIVE.match(text)
    if match:
        tail = match['tail']
        if match['name'].lower() == 'area':
            if not match['greeting']:
                return text
            # "Hello area code ..." is not a vocative. Inline ambiguous spelling
            # requires a punctuation boundary, as produced in the voice fixtures.
            if tail.strip(' ,.!?:;—-') and not tail.lstrip().startswith((',', '.', '!', '?', ':', ';', '—', '-')):
                return text
        return text[:match.start('name')] + 'Aria' + tail
    suffix = SPOKEN_SUFFIX.search(text)
    if suffix and closing_language(text[:suffix.start()]) is not None:
        return text[:suffix.start()] + 'Aria'
    return text


@dataclass
class Session:
    language: str
    ready_at: float
    busy: bool = False
    version: int = 0


@dataclass
class Turn:
    kind: str
    text: str
    language: str
    key: str | None
    session: Session | None
    version: int
    reset: list
    expired: bool = False


class ConversationGate:
    def __init__(self, *, enabled=True, idle_s=30, default_language='fr', limit=256,
                 clock=time.monotonic):
        if not math.isfinite(idle_s) or not 1 <= idle_s <= 600:
            raise ValueError('Conversation idle timeout must be 1..600 seconds')
        if default_language not in {'fr', 'en'} or not 1 <= limit <= 1024:
            raise ValueError('Invalid conversation language or session limit')
        self.enabled, self.idle_s = enabled, idle_s
        self.default_language, self.limit, self.clock = default_language, limit, clock
        self.sessions = OrderedDict()

    @classmethod
    def from_env(cls):
        return cls(enabled=os.environ.get('ARIA_WAKE_ENABLED', '1').lower() not in {'0', 'false', 'off'},
                   idle_s=float(os.environ.get('ARIA_CONVERSATION_IDLE_S', '30')),
                   default_language=os.environ.get('ARIA_WAKE_DEFAULT_LANGUAGE', 'fr'))

    def begin(self, key, text, arrived_at):
        # Stateless callers must wake on every request; never share a None session.
        key = key if key and len(key) <= 256 else None
        state = self.sessions.get(key)
        reset, expired = [], False
        if state and not state.busy and arrived_at >= state.ready_at + self.idle_s:
            self.sessions.pop(key)
            state, expired = None, True
            reset.append(key)
        wake = WAKE.match(text)
        body = text[wake.end():].strip() if wake else text.strip()
        language = state.language if state else self.default_language
        if body:
            language = closing_language(body) or detect_language(body)
        elif wake and wake['greeting']:
            greeting = wake['greeting'].lower()
            if greeting in {'bonjour', 'salut', 'coucou', 'hé'}:
                language = 'fr'
            elif greeting in {'hello', 'hey', 'hi'}:
                language = 'en'
        if not state and not wake:
            return Turn('ignored', body, language, key, None, 0, reset, expired)
        if closing_language(body):
            self.sessions.pop(key, None)
            if key and key not in reset:
                reset.append(key)
            return Turn('closed', body, language, key, None, 0, reset, expired)
        if state is None:
            state = Session(language, self.clock())
            if key:
                if key not in reset:
                    reset.append(key)
                self.sessions[key] = state
                while len(self.sessions) > self.limit:
                    evicted, _ = self.sessions.popitem(last=False)
                    reset.append(evicted)
            kind = 'activated'
        else:
            kind = 'followup'
        if key:
            self.sessions.move_to_end(key)
        state.language, state.busy = language, True
        state.version += 1
        return Turn(kind, body, language, key, state, state.version, reset, expired)

    def finish(self, turn, playback=None):
        def complete(_=None):
            state = self.sessions.get(turn.key)
            if state is turn.session and state is not None and state.version == turn.version:
                state.busy, state.ready_at = False, self.clock()
        if isinstance(playback, asyncio.Future) and not playback.done():
            playback.add_done_callback(complete)
        else:
            complete()


async def clear_context(state, keys):
    from app.llm.history_prefill import cancel_prefill
    for key in keys:
        mute_commands = getattr(state, 'mute_commands', None)
        if mute_commands is not None:
            mute_commands.forget_session(key)
        if getattr(state, 'llm_recent_session', None) == key:
            await cancel_prefill(state)
            state.llm_recent_session = None
        lock = getattr(state, 'llm_history_lock', None)
        history = getattr(state, 'llm_history', None)
        if lock is not None and isinstance(history, dict):
            async with lock:
                history.pop(key, None)
        memory = getattr(state, 'control_memory', None)
        if memory is not None:
            memory.forget_session(key)


def conversation_turn(processor):
    """Wrap the shared processor, keeping raw input in admission/private logs."""
    @wraps(processor)
    async def wrapped(app, job):
        gate = getattr(app.state, 'conversation_gate', None)
        if gate is None:
            gate = app.state.conversation_gate = ConversationGate.from_env()
        if not gate.enabled:
            return await processor(app, job)
        arrived_at = (job.conversation_started_at if job.source == "voice"
                      and job.conversation_started_at is not None else job.queued_at)
        address = normalize_spoken_vocative(job.text) if job.source == 'voice' else job.text
        turn = gate.begin(job.session_id, address, arrived_at)
        await clear_context(app.state, turn.reset)
        log.info('ARIA.Conversation', extra={'fields': {
            'request_id': job.request_id, 'session_id': job.session_id,
            'action': turn.kind, 'expired': turn.expired,
            'idle_timeout_s': gate.idle_s, 'source': job.source,
            'spoken_name_normalized': address != job.text}})
        result = None
        try:
            if turn.kind == 'ignored' or turn.kind == 'closed' or not turn.text:
                reply = ('' if turn.kind == 'ignored' else
                         ('Au revoir.' if turn.language == 'fr' else 'Goodbye.') if turn.kind == 'closed' else
                         ('Oui ?' if turn.language == 'fr' else 'Yes?'))
                spoken = False
                speech = getattr(app.state, 'speech_output', None)
                if reply and job.speak_response and speech is not None:
                    try:
                        await speech.speak(reply, language=turn.language)
                        spoken = True
                    except Exception as exc:
                        log.info('ARIA.TTS.Error', extra={'fields': {'error': repr(exc)}})
                result = dict(success=True, response=reply, spoken=spoken,
                              tool_calls_made=0, error=None, ignored=turn.kind == 'ignored')
            else:
                job.conversation_text = turn.text
                result = await processor(app, job)
            return {**result, 'text': job.text, 'conversation_action': turn.kind,
                    'conversation_state': 'active' if turn.session and turn.key else 'waiting'}
        finally:
            speech = getattr(app.state, 'speech_output', None)
            playback = getattr(speech, 'playback_done', None) if result and result.get('spoken') else None
            gate.finish(turn, playback)
    return wrapped
