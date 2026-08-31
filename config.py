"""Central configuration for the Samsung Phone Query and Review System.

Every tunable lives here and is overridable through environment variables or a
`.env` file, so the same code runs against XAMPP MySQL locally and PostgreSQL
in production without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
# Supported out of the box:
#   MySQL       mysql+pymysql://root:@127.0.0.1:3306/samsung_phones
#   PostgreSQL  postgresql+psycopg2://postgres:pass@127.0.0.1:5432/samsung_phones
#   SQLite      sqlite:///data/samsung_phones.db      (zero-setup fallback)
DB_BACKEND = _env("DB_BACKEND", "mysql").lower()
DB_HOST = _env("DB_HOST", "127.0.0.1")
DB_PORT = int(_env("DB_PORT", "3306" if DB_BACKEND == "mysql" else "5432"))
DB_USER = _env("DB_USER", "root")
DB_PASSWORD = _env("DB_PASSWORD", "")
DB_NAME = _env("DB_NAME", "samsung_phones")


def build_database_url(include_database: bool = True) -> str:
    """Assemble the SQLAlchemy URL for the configured backend.

    `include_database=False` yields a server-level URL used once at bootstrap to
    issue `CREATE DATABASE`, which cannot run while connected to a database that
    does not exist yet.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    name = DB_NAME if include_database else ""

    if DB_BACKEND == "sqlite":
        return f"sqlite:///{DATA_DIR / (DB_NAME + '.db')}"
    if DB_BACKEND == "mysql":
        password = f":{DB_PASSWORD}" if DB_PASSWORD else ""
        return f"mysql+pymysql://{DB_USER}{password}@{DB_HOST}:{DB_PORT}/{name}?charset=utf8mb4"
    if DB_BACKEND in ("postgres", "postgresql"):
        password = f":{DB_PASSWORD}" if DB_PASSWORD else ""
        # A server-level PostgreSQL connection still needs a database to attach
        # to; the always-present maintenance database is the conventional pick.
        target = name or "postgres"
        return f"postgresql+psycopg2://{DB_USER}{password}@{DB_HOST}:{DB_PORT}/{target}"

    raise ValueError(f"Unsupported DB_BACKEND: {DB_BACKEND!r}")


DATABASE_URL = build_database_url()

# --------------------------------------------------------------------------
# Scraper
# --------------------------------------------------------------------------
GSMARENA_BASE_URL = "https://www.gsmarena.com/"
GSMARENA_SAMSUNG_LISTING = "samsung-phones-9.php"

# GSMArena rate-limits aggressively; this delay keeps the crawl polite and is
# the main reason a full scrape takes a couple of minutes rather than seconds.
SCRAPER_DELAY_SECONDS = float(_env("SCRAPER_DELAY_SECONDS", "1.5"))
SCRAPER_TIMEOUT = int(_env("SCRAPER_TIMEOUT", "30"))
SCRAPER_MAX_RETRIES = int(_env("SCRAPER_MAX_RETRIES", "3"))
SCRAPER_MAX_LISTING_PAGES = int(_env("SCRAPER_MAX_LISTING_PAGES", "16"))
SCRAPER_USER_AGENT = _env(
    "SCRAPER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

# The 15 models the system is built around: five flagship generations, the
# foldables, and two mid-rangers so comparison queries have range to work with.
TARGET_MODELS = [
    "Galaxy S21 5G",
    "Galaxy S21 Ultra 5G",
    "Galaxy S22 5G",
    "Galaxy S22 Ultra 5G",
    "Galaxy S23",
    "Galaxy S23 Ultra",
    "Galaxy S23 FE",
    "Galaxy S24",
    "Galaxy S24 Ultra",
    "Galaxy S25",
    "Galaxy S25 Ultra",
    "Galaxy Z Fold5",
    "Galaxy Z Flip5",
    "Galaxy Z Fold6",
    "Galaxy A54",
]

# --------------------------------------------------------------------------
# RAG / LLM
# --------------------------------------------------------------------------
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_MAX_NEW_TOKENS = int(_env("LLM_MAX_NEW_TOKENS", "512"))
# Low by design: the model is summarising retrieved specs, so faithfulness to
# the numbers matters far more than variety of phrasing.
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.2"))
LLM_DEVICE = _env("LLM_DEVICE", "auto")  # auto | cuda | cpu

# With LLM_DEVICE=auto, fall back to CPU when less than this much VRAM is free.
# The default 1.5B model needs ~3.1 GB in fp16 plus room for the KV cache, so
# 4 GB is the point below which a load is likely to fail partway through.
# Raise it if you switch to a larger LLM_MODEL.
LLM_MIN_FREE_VRAM_GB = float(_env("LLM_MIN_FREE_VRAM_GB", "4.0"))

# When False the system answers from retrieved specs using deterministic
# templates instead of a generative model. Useful on machines without a GPU.
USE_LLM = _env("USE_LLM", "true").lower() in ("1", "true", "yes")

RAG_TOP_K = int(_env("RAG_TOP_K", "5"))
VECTOR_INDEX_PATH = DATA_DIR / "vector_index"

# --------------------------------------------------------------------------
# Multi-agent system
# --------------------------------------------------------------------------
# Which orchestration backend runs the agent crews:
#   crewai  — CrewAI Agent/Task/Crew, driven by the local model
#   native  — the built-in orchestrator in src/agents/base.py
#   auto    — CrewAI when it is installed and working, native otherwise
AGENT_FRAMEWORK = _env("AGENT_FRAMEWORK", "auto").lower()

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = int(_env("API_PORT", "8000"))
