"""Point the application at a throwaway SQLite database before anything imports it.

`config` reads the environment once, at import time, and `load_dotenv()` would
otherwise pick up a developer's `.env` (typically the XAMPP MySQL database). An
explicit `DATABASE_URL` wins over every other database setting, so setting it
here guarantees the evaluation can never read or write a real database.

Import this module first, before `config` or anything under `src/`.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Generated files (database, FAISS index) stay out of the repository.
SCRATCH_DIR = Path(
    os.getenv("EVAL_SCRATCH_DIR") or Path(tempfile.gettempdir()) / "fs" / "pa_scratch" / "eval"
).resolve()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = SCRATCH_DIR / "eval_phones.db"

os.environ["DB_BACKEND"] = "sqlite"
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"
os.environ["VECTOR_INDEX_PATH"] = str(SCRATCH_DIR / "vector_index")
# The caller decides; default to the deterministic path.
os.environ.setdefault("USE_LLM", "false")


def seed_database() -> int:
    """Rebuild the scratch database from `data/scraped_phones.json`.

    The file is recreated on every run so row ids — and therefore the index
    fingerprint — are identical from run to run.
    """
    from src.database.db import get_engine
    from src.database.models import Base
    from src.scraper.pipeline import load_from_snapshot

    engine = get_engine()
    Base.metadata.drop_all(engine)
    result = load_from_snapshot()
    return int(result["stats"]["phones"])
