"""Start the FastAPI server.

    python scripts/run_api.py
    python scripts/run_api.py --reload --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the API server")
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on edits")
    args = parser.parse_args()

    import uvicorn

    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"\n  Web UI    http://{display_host}:{args.port}")
    print(f"  API docs  http://{display_host}:{args.port}/docs")
    print("  First request may be slow while the models load.\n")

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
