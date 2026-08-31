# Samsung Phone Query and Review System

An intelligent system for querying Samsung smartphone data: specifications are
scraped from GSMArena into a relational database, a retrieval-augmented chatbot
answers questions about them using open-source models, a multi-agent pipeline
writes product reviews and comparisons, and everything is exposed through a
FastAPI service with a browser UI.

Everything runs locally. No paid API keys are required at any point.

---

## What it does

| Sample question | How it is answered |
| --- | --- |
| *"What are the camera specs of the Samsung Galaxy S23?"* | Model resolved, camera passage retrieved, answer generated from it |
| *"Which Samsung phone has the best battery life?"* | SQL ranking over `battery_capacity_mah`, then summarised |
| *"How does the Galaxy S23 compare to the S22 in terms of performance?"* | Both spec sheets fetched, difference table built, verdict written |
| *"Review the Galaxy S24 Ultra for a photographer"* | CrewAI crew: Data Specialist → Reviewer |

**Current dataset:** 15 Samsung models, 858 specification rows, 53 price
listings, 120 indexed passages.

---

## Architecture

```
                       ┌──────────────────────────────┐
   GSMArena  ────────► │  Scraper (requests + BS4)    │
                       │  discovery → parse → typed   │
                       └───────────────┬──────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │  MySQL / PostgreSQL          │
                       │  phones · specifications ·   │
                       │  prices                      │
                       └───────┬──────────────┬───────┘
                               │              │
              ┌────────────────▼───┐    ┌─────▼─────────────────┐
              │  RAG chatbot       │    │  CrewAI multi-agent   │
              │  MiniLM embeddings │    │  Data Specialist      │
              │  FAISS + intent    │    │  Reviewer             │
              │  routing → Qwen    │    │  Comparison Analyst   │
              └────────────────┬───┘    └─────┬─────────────────┘
                               │              │
                       ┌───────▼──────────────▼───────┐
                       │  FastAPI  ·  Web UI  ·  CLI  │
                       └──────────────────────────────┘
```

### Layout

```
config.py                    All settings, env-overridable
src/
  scraper/
    gsmarena_scraper.py      Discovery + spec-page extraction
    parsers.py               Free text → typed values
    pipeline.py              Scrape → validate → store
  database/
    models.py                SQLAlchemy schema (3 tables)
    db.py                    Engine, sessions, bootstrap
    repository.py            Queries and model-name resolution
  rag/
    documents.py             Rows → per-aspect passages
    vector_store.py          FAISS index + cosine search
    query_analysis.py        Intent classification
    chatbot.py               Retrieval → context → answer
  agents/
    crewai_backend.py        CrewAI agents, tasks, crews + local-model adapter
    base.py                  Fallback orchestrator (same Agent/Task/Crew shape)
    specialists.py           Specialist agents for the fallback backend
    crew.py                  Workflows + backend selection
  llm/provider.py            Shared Hugging Face model
  api/
    main.py                  FastAPI endpoints
    schemas.py               Pydantic contracts
    web.py                   Browser UI
scripts/                     scrape · demo · chat · run_api
tests/test_system.py         34 tests
```

---

## Setup

### 1. Requirements

- Python 3.10+ (developed on 3.13)
- MySQL or PostgreSQL running locally — XAMPP's MySQL is fine
- ~4 GB disk for model weights; a GPU is optional but much faster

### 2. Install

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit if your DB credentials differ
```

CrewAI is the largest item in that file (~70 transitive packages). If it fails
to install, everything still works — the agent layer falls back to the built-in
orchestrator and `/health` reports which backend is live.

### 3. Configure the database

`.env` defaults to MySQL on `127.0.0.1:3306` as `root` with no password, which
matches a stock XAMPP install. The database itself is created automatically —
you only need the server running.

<details>
<summary>PostgreSQL instead</summary>

```ini
DB_BACKEND=postgresql
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=yourpassword
```
Then `pip install psycopg2-binary`.
</details>

<details>
<summary>No database server at all</summary>

Set `DB_BACKEND=sqlite` in `.env`. Everything else works unchanged.
</details>

### 4. Scrape

```bash
python scripts/scrape.py
```

Takes about two minutes: it walks the paginated Samsung listing to find each
model's URL, then parses its spec page, pausing 1.5 s between requests. It also
writes `data/scraped_phones.json`, so the database can later be rebuilt without
touching the network:

```bash
python scripts/scrape.py --from-snapshot
```

### 5. Run

```bash
python scripts/run_api.py      # http://127.0.0.1:8000
```

The first request downloads the models (~2 GB embeddings + LLM) and is slow;
everything after that is fast. Also available:

```bash
python scripts/demo.py         # full walkthrough of all four subsystems
python scripts/chat.py         # interactive terminal chatbot
python tests/test_system.py    # test suite
```

---

## API

Interactive docs at `/docs`. Browser UI at `/`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | Ask a question, get a grounded answer with sources |
| `GET` | `/chat?q=...` | Same, convenient from a browser |
| `POST` | `/agents/review` | Generate a product review |
| `POST` | `/agents/compare` | Head-to-head comparison |
| `POST` | `/agents/query` | Free text, auto-routed to the right crew |
| `GET` | `/phones` | All phones |
| `GET` | `/phones/{id}` | One phone, with complete raw specs |
| `GET` | `/phones/search?q=` | Fuzzy model-name search |
| `GET` | `/phones/by-name/{name}` | Look up by name |
| `GET` | `/health` | Database, index and model status |
| `GET` | `/stats` | Row counts and configured models |

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Which Samsung phone has the best battery life?"}'
```

```json
{
  "answer": "The Samsung phone with the best battery life is the Samsung Galaxy S25 Ultra, with a battery capacity of 5000 mAh...",
  "intent": "superlative",
  "phones": ["Samsung Galaxy S25 Ultra", "Samsung Galaxy S24 Ultra"],
  "sources": [{"phone": "Samsung Galaxy S25 Ultra", "aspect": "ranking:battery"}],
  "generated_by": "llm:Qwen/Qwen2.5-1.5B-Instruct"
}
```

---

## How each part works

### 1. Scraping

Two stages. **Discovery** walks `samsung-phones-9.php` and its paginated
continuations, building a name→URL index and stopping as soon as all target
models are found. **Extraction** parses `#specs-list`, which GSMArena renders as
one table per category with `td.ttl`/`td.nfo` label-value pairs.

Because GSMArena writes for humans (`"Li-Ion 5000 mAh, non-removable"`), every
numeric column goes through a parser in `parsers.py`. Each returns `None` rather
than guessing, so unparseable entries stay out of the rankings instead of
corrupting them. Two cases worth noting:

- **Camera resolution** takes the *first* sensor listed, not the largest. On the
  S23 Ultra the string reads "200 MP wide, 10 MP periscope, 10 MP telephoto,
  12 MP ultrawide" — the headline sensor is the one that leads.
- **RAM vs storage** share the `NNGB` shape and are told apart by the `RAM`
  suffix, so `"128GB 8GB RAM, 512GB 12GB RAM"` yields 12 GB RAM / 512 GB storage.

The crawl reuses one TCP session, waits between requests, and backs off
exponentially on HTTP 429/503.

### 2. Database

Three tables, keeping two views of the same scrape:

- **`phones`** flattens the specs you filter and sort on, with numeric columns
  parsed out of the free text — so "best battery life" is plain SQL, not
  string matching.
- **`specifications`** preserves every key/value GSMArena publishes, unmodified,
  so nothing is lost to that flattening.
- **`prices`** is separate because one phone carries several regional listings.

Re-scraping a phone replaces its child rows rather than appending, so the data
never accumulates duplicates.

### 3. RAG chatbot

Each phone becomes several **aspect documents** — display, camera, performance,
battery, design, connectivity, pricing — rather than one blob. A camera question
should retrieve the camera paragraph, not a 60-line spec dump where the relevant
sentence is diluted. Embeddings are L2-normalised in a FAISS inner-product
index, making the returned score exact cosine similarity.

Retrieval alone is not enough, though. Embeddings have no notion of *ranking*,
so "which phone has the best battery life?" would return passages that merely
mention batteries rather than the one with the largest number. And a comparison
question names two models, so a single nearest-neighbour lookup returns whichever
the wording happens to favour. So questions are **routed by intent first**:

| Intent | Context strategy |
| --- | --- |
| `superlative` | SQL `ORDER BY` on the matching numeric column |
| `comparison` | Both spec sheets, plus aspect-focused retrieval |
| `spec_lookup` | Vector search restricted to the named phone and aspect |
| `price` | Price table for the named model |
| `list` | Full catalogue |
| `general` | Unrestricted vector search |

Retrieval is also **aspect-filtered**. Padding context out to `top_k` with
whatever ranked next turned out to be actively harmful: asked for the S23's
camera specs, the model received the display passage too and reported the
screen's 120 Hz refresh rate as a selfie-camera capability. Off-topic passages
are now dropped rather than merely demoted, and a test guards the regression.

### 4. Multi-agent system

Built on **CrewAI** (`src/agents/crewai_backend.py`): real `Agent`, `Task` and
`Crew` objects running `Process.sequential`, with a database `BaseTool` and
CrewAI's task-to-task context passing.

| Agent | Responsibility |
| --- | --- |
| Samsung Phone Data Specialist | Holds the database tool. Reports facts, no prose. |
| Senior Smartphone Reviewer | Writes a seven-section review from those facts. |
| Smartphone Comparison Analyst | Delivers a verdict from the difference table. |

The data agent always runs first and the writers consume its output through
`Task(context=[...])`. That ordering is the point of the split: the writers have
no database access, so they have nothing to work from except retrieved facts.

**Running CrewAI on a local model.** CrewAI routes model calls through LiteLLM,
which expects a hosted provider, while the brief also asks for open-source
models. `LocalCrewLLM` subclasses CrewAI's `BaseLLM` so the crew drives the same
local Qwen model as the rest of the system — no API key, no separate inference
server.

**Why the specification data is injected rather than looked up by the agent.**
The database tool is attached to the specification agent, but the authoritative
spec sheet is also placed directly in the task description. This is not
belt-and-braces for its own sake — it is a measured response to how the small
model behaves. Asked to look up the Galaxy S23 Ultra through the tool, Qwen-1.5B
skipped the tool and answered from memory, reporting **108 MP** (the S22 Ultra's
figure) instead of the correct 200 MP. A review built on a failed lookup is
fiction, so the crew is given the verified record and the tool remains for models
capable of calling it.

The same reasoning applies one step further down the chain. In an early version
only task 1 saw the database record and task 2 worked purely from task 1's
paraphrase of it; the errors compounded, and a comparison came back citing an
"Exynos 2300" chipset and a 4500 mAh battery that no phone in the database has.
Both writing tasks now receive the authoritative record alongside the upstream
context, with instructions that the record wins on any disagreement.

**Two backends.** `src/agents/base.py` also contains a self-contained
orchestrator with the same Agent/Task/Crew shape. It runs when CrewAI is not
installed or no LLM can be loaded, and unlike CrewAI it has deterministic
template fallbacks — so the system still answers on hardware that cannot run a
generative model at all. `AGENT_FRAMEWORK` selects between them; `auto` (the
default) prefers CrewAI. `/health` reports which one is live.

### 5. Models

| Role | Model | Why |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 22M params, fast, strong on short factual passages |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | Fits in 6 GB VRAM, good instruction-following |

Both are open source and downloaded from Hugging Face on first run. Swap either
in `.env`; a larger `LLM_MODEL` gives noticeably more precise prose if you have
the VRAM.

**Graceful degradation.** If the LLM cannot be loaded — no GPU, low memory, no
network, or `USE_LLM=false` — every component falls back to deterministic
templates built from the same retrieved facts. Answers stay correct; only the
prose quality changes. The load is attempted once per process, and a failure is
cached so later requests skip the slow retry.

---

## Testing

```bash
python tests/test_system.py
```

37 tests across six areas: spec parsers, intent routing, database queries and
name resolution, RAG retrieval, agent-backend selection, agent workflows, and
the HTTP contract. The LLM is disabled throughout — the tests check the
deterministic machinery, not the wording a generative model happens to produce.
With no LLM, CrewAI reports itself unavailable and the suite exercises the
native fallback path.

```
Ran 37 tests in 31.762s

OK
```

---

## Configuration

Every setting in `config.py` is overridable from `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DB_BACKEND` | `mysql` | `mysql` · `postgresql` · `sqlite` |
| `DATABASE_URL` | — | Full SQLAlchemy URL; overrides the parts above |
| `SCRAPER_DELAY_SECONDS` | `1.5` | Politeness delay between requests |
| `SCRAPER_MAX_LISTING_PAGES` | `16` | Discovery crawl depth |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for retrieval |
| `LLM_MODEL` | `Qwen2.5-1.5B-Instruct` | Generative model |
| `LLM_DEVICE` | `auto` | `auto` · `cuda` · `cpu` |
| `USE_LLM` | `true` | `false` forces template answers |
| `RAG_TOP_K` | `5` | Passages retrieved per query |
| `AGENT_FRAMEWORK` | `auto` | `crewai` · `native` · `auto` |

Tracked phone models are listed in `TARGET_MODELS` in `config.py`.

---

## Troubleshooting

**`CUDA error: out of memory`**

The model needs about 3.1 GB of VRAM, so on a 6 GB card only one copy fits.
Running the API server and the demo script (or the tests, or `task.py chat`) at
the same time means two processes each trying to load their own copy, and the
second one fails.

Either stop the other process, or give the second one its own device:

```bash
# Windows
set LLM_DEVICE=cpu && python task.py demo
# bash
LLM_DEVICE=cpu python task.py demo
```

With `LLM_DEVICE=auto` (the default) this is now detected before loading:
if less than `LLM_MIN_FREE_VRAM_GB` is free the model loads on CPU with a
warning instead of crashing. CPU generation works but is several times slower.
`USE_LLM=false` skips the model entirely and answers from templates.

Check what is holding the GPU with `nvidia-smi`, and what the app thinks it is
using at `/health` under `llm`.

**Agent endpoints feel slow.** Expected — see Known limitations below.
`AGENT_FRAMEWORK=native` is roughly twice as fast.

**`Can't connect to MySQL server`.** Start MySQL from the XAMPP Control Panel.
Or set `DB_BACKEND=sqlite` in `.env` to run with no database server at all.

---

## Known limitations

- **Small-model precision.** At 1.5B parameters the generator occasionally
  misreads a figure when a passage lists four camera lenses in one string, and
  in comparisons it sometimes declares a winner on a spec where the two phones
  are identical. The retrieved context and the difference table are correct in
  both cases — it is the prose that slips. A larger `LLM_MODEL` fixes it; this
  is the main reason the numeric comparison table is rendered directly from the
  database rather than left to the model.
- **Agent latency.** A CrewAI review takes roughly 75–135 s and a comparison
  65–90 s on a 6 GB laptop GPU. Two agents each generate several hundred tokens,
  and generation slows noticeably as the prompt grows — which the injected spec
  sheets make unavoidable. `AGENT_FRAMEWORK=native` is about three times faster
  (one LLM call per agent) if you want a snappier demo. The chatbot endpoints
  are unaffected and answer in a few seconds.
- **Prices are point-in-time.** GSMArena lists current street prices, captured
  at scrape time, not launch MSRP.
- **Battery ties.** Six models share a 5000 mAh cell, so "best battery life"
  ranks by capacity alone — GSMArena's endurance rating is not scraped.
- **Layout coupling.** The scraper targets GSMArena's current markup. If the
  site restructures, `parse_spec_page` needs updating; it logs a warning and
  skips rather than storing garbage.
