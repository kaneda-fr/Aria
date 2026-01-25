from __future__ import annotations

import re
from dataclasses import dataclass


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    enabled: bool = True
    max_chars_per_chunk: int = 220
    min_chunk_chars: int = 60


def chunk_text(text: str, cfg: ChunkingConfig) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []

    # Replace code blocks with a short spoken summary.
    t = _CODE_BLOCK_RE.sub("[code omitted]", t)

    # Avoid reading long URLs.
    t = _URL_RE.sub("[link]", t)

    t = _WS_RE.sub(" ", t).strip()

    if not cfg.enabled:
        return [t]

    # Sentence-ish splitting.
    parts = _split_sentences(t)

    chunks: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
            continue
        # If adding keeps us under max, append.
        if len(cur) + 1 + len(p) <= cfg.max_chars_per_chunk:
            cur = cur + " " + p
            continue
        chunks.append(cur)
        cur = p

    if cur:
        chunks.append(cur)

    # Merge very small chunks forward when possible.
    merged: list[str] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if len(c) < cfg.min_chunk_chars and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            if len(c) + 1 + len(nxt) <= cfg.max_chars_per_chunk:
                merged.append(c + " " + nxt)
                i += 2
                continue
        merged.append(c)
        i += 1

    return merged


def _split_sentences(text: str) -> list[str]:
    # Keep punctuation with the sentence.
    out: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+\s+", text):
        end = m.end()
        out.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
