# Interaction coverage audit — 2026-10-18

## Finding

The current fast control path recognizes a small grammar. A fluent LLM does not
make that grammar general: an utterance that misses `resolve()` gets no execution
authorization, so a later model `execute_command` is rejected. This prevents
wrong-device execution but also prevents valid synonyms from controlling devices.
The existing exact-command benchmark measures that narrow path, not household
language coverage. There is no science-question answer table in the runtime.

`python -m scripts.e2e.phrase_surface --output <private-json>` reproduces the
coverage with a synthetic device and no action publisher. Of 24 FR/EN control
utterances, **9 receive authorization**. Two others request unspecified relative
brightness and need clarification or an explicit relative-value policy; excluding
those leaves 9/22 explicit controls. Of 16 clock/capability paraphrases, **4 use
the grounded local answer**. These counts measure routing/authorization, not the
quality of a generated fallback reply or live hardware reliability.

## Wake name and conversation state

At the original audit, VAD admitted eligible final transcripts without a wake
name. Session history existed, but activation/expiry did not. The narrow control
grammar could strip an optional “Aria” prefix; that was not an activation gate.
Speaker attribution alone did not prove someone was addressing the assistant.

Update: [D-010 wake flow](wake-flow-2026-10-18.md) now implements post-ASR
activation, follow-ups, expiry, and goodbye. The small activation/closing grammar
is an explicit interaction protocol; it does not widen the domain-command grammar
below. Acoustic/name-ASR robustness remains a separate validation limit.
The [spoken-name follow-up](spoken-wake-2026-10-18.md) adds voice-only vocative
spelling normalization. Its finite name aliases are explicitly bounded and tested
against unrelated words; they do not repair arbitrary ASR or broaden control intent.

## Restrictions in code

| Location | Mechanism and consequence |
| --- | --- |
| [control.py](../../app/llm/control.py), `PREFIX`, `ACTION`, `VERBS`, `resolve` | Finite polite prefixes and verb lists; full-string grammar. “Allume Atlas” works; “Mets Atlas en marche”, “Active la lampe Atlas”, “Power up Atlas” and “Shut off Atlas” fall through without a grant. Percentages accept digits, not “trente pour cent” / “thirty percent”. |
| [control.py](../../app/llm/control.py), location extraction | French “dans la cuisine” retains the article in the room name; exact inventory matching then returns `not_found` for room `cuisine`. The English “in the kitchen” equivalent works. |
| [selection.py](../../app/plugins/inventory/selection.py), `DOMAINS`, `FILLER`, `select_commands` | Finite domain/filler vocabulary, exact normalized room/name matching and rejection of unmatched words. No configured room/device aliases or semantic resolution. Normalization is Latin-centric. |
| [local_answers.py](../../app/llm/local_answers.py), `clock_answer`, `capability_answer` | Full-match question templates. “Quelle heure est-il ?” / “What time is it?” work; “Tu as l’heure ?” / “Tell me the current time” do not. “Que peux-tu faire ?” works; “Comment peux-tu m’aider ?” does not. Clock misses can produce an ungrounded model estimate because the fallback has no clock tool. |
| [main.py](../../app/main.py), `_process_request` | Only a server-resolved ready plan grants execution. LLM understanding after a parser miss cannot compensate. Unsupported attempted execution gets a truthful refusal. Inventory lookup remains available to the model. |
| [main.py](../../app/main.py), `_chat_shortcut` | Literal greeting dictionary, gated on the router. Inactive in the current Apollo profile (`ARIA_ROUTER_ENABLED=0`), but another phrase-specific path if enabled. |
| [speech_output.py](../../app/speech_output/speech_output.py), `_detect_lang_heuristic`; [lang_detect.py](../../app/routing/lang_detect.py) | Accent checks and finite function-word sets, with English as fallback. Short French “Coupe Atlas” and “Donne-moi l’heure actuelle” become English in this audit. Interpretation and TTS language should not depend on these few words alone. |

Do not confuse these with useful deterministic code. Outcome templates in
`render()` keep spoken feedback faithful to recorded/published/failed execution.
Command IDs, capability checks, numeric bounds, one-use grants, ambiguity handling,
negation and unsupported multi-action checks are safety boundaries. Echo-filter
patterns and TTS text sanitation are separate audio concerns. Router seed examples
train an optional classifier; they are not literal answer rules and that classifier
currently does not solve control authorization.

## Recommended next implementation

1. Keep the current parser as a cheap fast path, but add a **bounded semantic
   interpretation fallback** for misses. Compare the existing tool-calling model
   with the parser/orchestrator alternative; this audit does not settle D-001.
   Either approach must propose structured intent, device/room reference, action
   and value, then pass through the same server-side inventory and capability
   validation before it can receive a one-use execution grant.
2. Resolve configured device/room aliases against the current inventory. Ask a
   short clarification for ambiguity or an unspecified relative value. Do not
   authorize a fuzzy name match or a generated command ID directly. Correct the
   French article bug independently; it needs no model inference.
3. Expose clock and capabilities as grounded operations to the semantic fallback,
   so paraphrases cannot send these questions to an ungrounded factual model.
   Use request-level ASR language evidence where available, with a tested fallback
   for text; preserve the selected language through all TTS chunks.
4. Expand the held-out benchmark with synonyms, varied word order, accents,
   spelled-out numbers, negation, pronouns, ambiguity and multi-turn clarification.
   Keep examples in evaluation data, not runtime lookup tables. Test unfamiliar
   combinations and independently authored household utterances before claiming
   broad coverage.

Measure direct-path and fallback latency separately on Apollo, including queueing.
Keep one inference worker, bounded context/output, local-only operation, log-only
actions and capture-only audio during this work. A larger vocabulary list alone
would move the boundary without solving the limited interaction surface.

This task identifies these restrictions; it does not silently broaden execution
authority or alter the working echo/voice path.
