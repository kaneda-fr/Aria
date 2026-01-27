# Aria Integration Instructions (Archived)

**⚠️ This file is obsolete.**

The authoritative project specification is now maintained in:
- **[`.copilot-instructions.md`](.copilot-instructions.md)** - Complete architectural specification
- **[`README.md`](README.md)** - User-facing documentation
- **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)** - Configuration reference

This file is kept for historical reference only.

---

## Original Content (Archived)

You are working on the Aria project.

The file `$HOME/dev/aria/.copilot-instructions.md` is the authoritative project specification and MUST be updated before implementing.

Context: The current instructions already include **Echo Suppression v2 (Option A)** and Sonos HTTP TTS in a single-server architecture.
We will KEEP Echo Guard v2 for now, and ADD server-side speaker identification that works on **in-memory finalized VAD audio segments** (NO WAV files in the hot path).
We will also adopt **language-specific speaker profiles**, e.g. `seb_fr`, `seb_en`, and a configurable list of “self” identities (e.g. `aria_fr`, `aria_en`), used to suppress self-speech before the LLM.

DO NOT change existing, working microphone streaming behavior.
DO NOT add recording/enrollment tools to Aria (sandbox stays separate).
DO NOT write WAV files for recognition at runtime (debug-only is fine behind a flag).
All recognition must run on the server.

================================================================
PHASE 1 — UPDATE `.copilot-instructions.md` (SPEC)
================================================================

Add a new section under PART 2 or PART 3 called:

### Speaker Recognition (Server-side, in-memory, before LLM)

Add/clarify these requirements:

1) Sandbox remains separate
- `record_samples.py` and `enroll.py` remain standalone tools and do NOT ship in the Aria server repo.
- Enrollment output is JSON profile files copied onto the server.

2) Profile format and naming (language-specific)
- Profiles live in a server directory configured by env: `ARIA_SPK_PROFILES_DIR=/var/aria/profiles`
- Each profile is produced by the sandbox `enroll.py` and contains:
  `{ "name": "<speaker_id>", "embedding": [float32...] }`
- Speaker IDs may be language-specific:
  - `seb_fr`, `seb_en`, `alice_fr`, etc.
- “Self” identities are configured by env and used for suppression:
  - `ARIA_SPK_SELF_NAMES=aria_fr,aria_en`  (comma-separated)
- (Optional) Map speaker_id → base user:
  - `seb_fr` and `seb_en` both map to `seb` (derive by stripping suffix `_fr/_en` unless overridden).

3) Runtime inputs (NO WAV)
- Speaker recognition must operate on the **in-memory finalized VAD segment audio buffer**, not on a WAV file.
- The ASR/VAD pipeline must expose the finalized segment as:
  - PCM float32 (preferred) or pcm_s16le plus sample rate and channels
  - same segment used for ASR FINAL transcript
- Speaker recognition converts:
  - to mono
  - resamples to 16k
  - runs embedding extraction

4) Recognition timing and gating
- Recognition happens **before calling the LLM**.
- On each FINAL transcript:
  a) run speaker recognition on the SAME finalized audio segment
  b) attach `speaker_name`, `speaker_score`, `speaker_known`, and `speaker_user` (base) to the transcript event
  c) if `speaker_name` is in `ARIA_SPK_SELF_NAMES` AND `speaker_known=True` → DROP transcript (do not call LLM)
  d) else call LLM with `speaker_user` (or `unknown`) available for routing

5) Keep Echo Guard v2
- Echo Guard v2 remains enabled as a safety backstop to prevent feedback loops.
- Suppression rule precedence:
  - First: self-speech suppression via speaker recognition (if confident)
  - Second: Echo Guard v2 scoring suppression (Option A)
  - Otherwise: allow to LLM

6) Performance/latency controls
Add env vars:
ARIA_SPK_ENABLED=1
ARIA_SPK_PROFILES_DIR=/var/aria/profiles
ARIA_SPK_MODEL=speechbrain/spkrec-ecapa-voxceleb
ARIA_SPK_DEVICE=cpu
ARIA_SPK_NUM_THREADS=4
ARIA_SPK_MIN_SECONDS=1.0
ARIA_SPK_THRESHOLD_DEFAULT=0.65
ARIA_SPK_SELF_MIN_SCORE=0.75
ARIA_SPK_TIMEOUT_SEC=1.0
ARIA_SPK_HOT_RELOAD=0   # v1 loads at startup only
ARIA_SPK_DEBUG_DUMP_WAV=0  # if 1, write debug wavs; default 0
ARIA_SPK_SELF_NAMES=aria_fr,aria_en

7) Logging
Log at minimum:
- `ARIA.SPK: name=<name> score=<score> known=<bool> user=<user>`
- If suppressed as self speech:
  - `ARIA.SPK: suppressed_self_speech name=<name> score=<score>`
- Keep existing transcript/LLM logs.

================================================================
PHASE 2 — IMPLEMENTATION (CODE)
================================================================

Implement server-side speaker recognition as specified above.

A) Create module:
`app/speaker_recognition/recognizer.py`

Responsibilities:
- Load profiles from `ARIA_SPK_PROFILES_DIR` at startup
- Load and warm the encoder model at startup (avoid first-utterance latency)
- Provide an API that accepts **in-memory audio**:
  - `identify(audio: np.ndarray | bytes, sample_rate: int, channels: int, *, now_ts: float) -> SpeakerResult`
- Convert input to:
  - mono
  - 16k
  - float32
- Extract embedding with SpeechBrain ECAPA (default model)
- Compare with cosine similarity to all loaded profiles
- Return:
  - best_name, best_score
  - second_best_score (for debugging)
  - known: best_score >= threshold (per env)
  - user: base user derived from speaker_id (e.g. `seb_fr` → `seb`)
- Must support language-specific profiles naturally (just compare against all profiles)

B) Threading/async
- Recognition must not block the audio ingest loop.
- Run embedding extraction in a thread executor or background task, but the FINAL transcript handling must **await** the result up to `ARIA_SPK_TIMEOUT_SEC`.
- On timeout:
  - set `speaker_name=unknown`, `speaker_known=False`, and proceed to LLM (do not block forever).

C) Integration point (pre-LLM)
- At the handler that receives FINAL transcript (where you currently do Echo Guard v2 + LLM):
  1) call speaker recognizer on the finalized VAD segment buffer (in-memory)
  2) if recognized as self (in ARIA_SPK_SELF_NAMES) and score >= ARIA_SPK_SELF_MIN_SCORE:
     - log suppressed_self_speech
     - drop transcript (no LLM)
     - return
  3) else run Echo Guard v2 suppression as backstop (existing)
  4) else call LLM with speaker label available

D) Audio segment contract
- Ensure the FINAL transcript pipeline passes both:
  - final transcript text
  - finalized segment audio buffer + sample rate + channels
to the downstream processing stage (speaker + echo guard + LLM).

E) Do NOT add enrollment tools
- Do not copy `record_samples.py` or `enroll.py` into Aria.
- Profiles are managed operationally by copying JSONs to `ARIA_SPK_PROFILES_DIR`.

Deliverables:
- Updated `.copilot-instructions.md` with the new Speaker Recognition section
- `app/speaker_recognition/recognizer.py` implemented
- Integration into FINAL transcript handler (pre-LLM) using in-memory audio
- Uses language-specific profiles (`seb_fr`, `seb_en`) and self names list
- Echo Guard v2 kept as safety backstop

Then stop.# Aria Integration Instructions (Archived)

**⚠️ This file is obsolete.**

The authoritative project specification is now maintained in:
- **[`.copilot-instructions.md`](.copilot-instructions.md)** - Complete architectural specification
- **[`README.md`](README.md)** - User-facing documentation  
- **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)** - Configuration reference

This file is kept for historical reference only.

---

## Original Content (Archived)

You are working on the Aria project.

The file `$HOME/dev/aria/.copilot-instructions.md` is the authoritative project specification and MUST be updated before implementing.

Context: The current instructions already include **Echo Suppression v2 (Option A)** and Sonos HTTP TTS in a single-server architecture.
We will KEEP Echo Guard v2 for now, and ADD server-side speaker identification that works on **in-memory finalized VAD audio segments** (NO WAV files in the hot path).
We will also adopt **language-specific speaker profiles**, e.g. `seb_fr`, `seb_en`, and a configurable list of “self” identities (e.g. `aria_fr`, `aria_en`), used to suppress self-speech before the LLM.

DO NOT change existing, working microphone streaming behavior.
DO NOT add recording/enrollment tools to Aria (sandbox stays separate).
DO NOT write WAV files for recognition at runtime (debug-only is fine behind a flag).
All recognition must run on the server.

================================================================
PHASE 1 — UPDATE `.copilot-instructions.md` (SPEC)
================================================================

Add a new section under PART 2 or PART 3 called:

### Speaker Recognition (Server-side, in-memory, before LLM)

Add/clarify these requirements:

1) Sandbox remains separate
- `record_samples.py` and `enroll.py` remain standalone tools and do NOT ship in the Aria server repo.
- Enrollment output is JSON profile files copied onto the server.

2) Profile format and naming (language-specific)
- Profiles live in a server directory configured by env: `ARIA_SPK_PROFILES_DIR=/var/aria/profiles`
- Each profile is produced by the sandbox `enroll.py` and contains:
  `{ "name": "<speaker_id>", "embedding": [float32...] }`
- Speaker IDs may be language-specific:
  - `seb_fr`, `seb_en`, `alice_fr`, etc.
- “Self” identities are configured by env and used for suppression:
  - `ARIA_SPK_SELF_NAMES=aria_fr,aria_en`  (comma-separated)
- (Optional) Map speaker_id → base user:
  - `seb_fr` and `seb_en` both map to `seb` (derive by stripping suffix `_fr/_en` unless overridden).

3) Runtime inputs (NO WAV)
- Speaker recognition must operate on the **in-memory finalized VAD segment audio buffer**, not on a WAV file.
- The ASR/VAD pipeline must expose the finalized segment as:
  - PCM float32 (preferred) or pcm_s16le plus sample rate and channels
  - same segment used for ASR FINAL transcript
- Speaker recognition converts:
  - to mono
  - resamples to 16k
  - runs embedding extraction

4) Recognition timing and gating
- Recognition happens **before calling the LLM**.
- On each FINAL transcript:
  a) run speaker recognition on the SAME finalized audio segment
  b) attach `speaker_name`, `speaker_score`, `speaker_known`, and `speaker_user` (base) to the transcript event
  c) if `speaker_name` is in `ARIA_SPK_SELF_NAMES` AND `speaker_known=True` → DROP transcript (do not call LLM)
  d) else call LLM with `speaker_user` (or `unknown`) available for routing

5) Keep Echo Guard v2
- Echo Guard v2 remains enabled as a safety backstop to prevent feedback loops.
- Suppression rule precedence:
  - First: self-speech suppression via speaker recognition (if confident)
  - Second: Echo Guard v2 scoring suppression (Option A)
  - Otherwise: allow to LLM

6) Performance/latency controls
Add env vars:
ARIA_SPK_ENABLED=1
ARIA_SPK_PROFILES_DIR=/var/aria/profiles
ARIA_SPK_MODEL=speechbrain/spkrec-ecapa-voxceleb
ARIA_SPK_DEVICE=cpu
ARIA_SPK_NUM_THREADS=4
ARIA_SPK_MIN_SECONDS=1.0
ARIA_SPK_THRESHOLD_DEFAULT=0.65
ARIA_SPK_SELF_MIN_SCORE=0.75
ARIA_SPK_TIMEOUT_SEC=1.0
ARIA_SPK_HOT_RELOAD=0   # v1 loads at startup only
ARIA_SPK_DEBUG_DUMP_WAV=0  # if 1, write debug wavs; default 0
ARIA_SPK_SELF_NAMES=aria_fr,aria_en

7) Logging
Log at minimum:
- `ARIA.SPK: name=<name> score=<score> known=<bool> user=<user>`
- If suppressed as self speech:
  - `ARIA.SPK: suppressed_self_speech name=<name> score=<score>`
- Keep existing transcript/LLM logs.

================================================================
PHASE 2 — IMPLEMENTATION (CODE)
================================================================

Implement server-side speaker recognition as specified above.

A) Create module:
`app/speaker_recognition/recognizer.py`

Responsibilities:
- Load profiles from `ARIA_SPK_PROFILES_DIR` at startup
- Load and warm the encoder model at startup (avoid first-utterance latency)
- Provide an API that accepts **in-memory audio**:
  - `identify(audio: np.ndarray | bytes, sample_rate: int, channels: int, *, now_ts: float) -> SpeakerResult`
- Convert input to:
  - mono
  - 16k
  - float32
- Extract embedding with SpeechBrain ECAPA (default model)
- Compare with cosine similarity to all loaded profiles
- Return:
  - best_name, best_score
  - second_best_score (for debugging)
  - known: best_score >= threshold (per env)
  - user: base user derived from speaker_id (e.g. `seb_fr` → `seb`)
- Must support language-specific profiles naturally (just compare against all profiles)

B) Threading/async
- Recognition must not block the audio ingest loop.
- Run embedding extraction in a thread executor or background task, but the FINAL transcript handling must **await** the result up to `ARIA_SPK_TIMEOUT_SEC`.
- On timeout:
  - set `speaker_name=unknown`, `speaker_known=False`, and proceed to LLM (do not block forever).

C) Integration point (pre-LLM)
- At the handler that receives FINAL transcript (where you currently do Echo Guard v2 + LLM):
  1) call speaker recognizer on the finalized VAD segment buffer (in-memory)
  2) if recognized as self (in ARIA_SPK_SELF_NAMES) and score >= ARIA_SPK_SELF_MIN_SCORE:
     - log suppressed_self_speech
     - drop transcript (no LLM)
     - return
  3) else run Echo Guard v2 suppression as backstop (existing)
  4) else call LLM with speaker label available

D) Audio segment contract
- Ensure the FINAL transcript pipeline passes both:
  - final transcript text
  - finalized segment audio buffer + sample rate + channels
to the downstream processing stage (speaker + echo guard + LLM).

E) Do NOT add enrollment tools
- Do not copy `record_samples.py` or `enroll.py` into Aria.
- Profiles are managed operationally by copying JSONs to `ARIA_SPK_PROFILES_DIR`.

Deliverables:
- Updated `.copilot-instructions.md` with the new Speaker Recognition section
- `app/speaker_recognition/recognizer.py` implemented
- Integration into FINAL transcript handler (pre-LLM) using in-memory audio
- Uses language-specific profiles (`seb_fr`, `seb_en`) and self names list
- Echo Guard v2 kept as safety backstop

Then stop.
