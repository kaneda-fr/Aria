from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="aria-server", description="Run the ARIA FastAPI server")
    parser.add_argument("--host", default=os.environ.get("ARIA_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ARIA_PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("ARIA_WORKERS", "1")))
    parser.add_argument("--log-level", default=os.environ.get("ARIA_LOG_LEVEL", "info"))
    args = parser.parse_args()

    # Note: keep import string so uvicorn can manage lifespan correctly.
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=args.workers,
    )
