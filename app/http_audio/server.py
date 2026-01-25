from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def build_tts_router(*, tts_cache_dir: Path) -> APIRouter:
    """Create routes that serve normalized TTS WAV files.

    Sonos fetches audio via HTTP GET (pull model).
    """

    router = APIRouter()
    tts_dir = (tts_cache_dir / "sonos").resolve()

    @router.get("/tts/{key}.wav")
    async def get_tts_wav(key: str) -> FileResponse:
        # Key is expected to be a short hex-ish hash. Keep the path safe.
        safe = "".join(ch for ch in key if ch.isalnum())
        if safe != key or len(key) > 128:
            raise HTTPException(status_code=400, detail="invalid key")

        p = (tts_dir / f"{key}.wav").resolve()
        if tts_dir not in p.parents:
            raise HTTPException(status_code=400, detail="invalid path")
        if not p.exists() or p.stat().st_size == 0:
            raise HTTPException(status_code=404, detail="not found")

        return FileResponse(
            path=str(p),
            media_type="audio/wav",
            filename=p.name,
        )

    return router
