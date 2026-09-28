# Evaluation

A fixed question set with gold labels computed from the database, and a harness
that runs the real pipeline (`SamsungChatbot.chat()`: router, MiniLM
embeddings, FAISS, SQL rankings, template or Qwen generation) over it and scores
each stage separately. Nothing is mocked.

```
eval/
  _env.py            points the app at a scratch SQLite DB before config loads
  build_dataset.py   generates + verifies dataset.jsonl (deterministic)
  dataset.jsonl      188 questions, dev/test split
  run_eval.py        runs the pipeline, writes <out>.json and <out>.md
  results/           baseline runs (JSON has every answer and score)
```

## Running

```bash
python -m eval.build_dataset             # regenerate dataset.jsonl
python -m eval.build_dataset --check     # exit 1 if the file is stale
python -m eval.run_eval --mode template --split dev
python -m eval.run_eval --mode llm --split test --out eval/results/baseline_llm_test.json
python -m eval.run_eval --mode template --split all --limit 20   # quick smoke run
```

`--split` is `dev`, `test` or `all`; `--limit N` keeps the first N questions of
the split (the file is ordered by category, so a small limit only covers the
first categories). `--out` defaults to `eval/results/<mode>_<split>.json`; the
Markdown summary is written beside it.

**Isolation.** `eval/_env.py` sets `DATABASE_URL` to a SQLite file under
`$TMP/fs/pa_scratch/eval/` (override with `EVAL_SCRATCH_DIR`) and
`VECTOR_INDEX_PATH` next to it, before `config` is imported. An explicit
`DATABASE_URL` beats every other database setting, including a `.env`, so an
evaluation never touches a real MySQL/PostgreSQL database. The scratch database
is dropped and reloaded from `data/scraped_phones.json` on every run, so row ids
and the index fingerprint are identical between runs. `HF_HUB_OFFLINE=1` is
set by default; the embedding model (and Qwen for `--mode llm`) must already
be in the Hugging Face cache.

`--mode llm` refuses to run if the model cannot be loaded, rather than silently
reporting template answers; the `generated_by` field of each record shows what
actually produced it.

## How the dataset is built

`build_dataset.py` loads the snapshot into the scratch DB and generates every
question from hand-written templates. Which template, which phone and which
spelling of the name is used comes from `random.Random(20260928)`, so the output
is byte-identical on every run (`--check` verifies this).

| Category | n (dev/test) | What it is | Gold |
|---|---|---|---|
| spec_lookup | 45 (23/22) | Each of the 15 phones x 3 of the 10 query aspects (display, camera, performance, storage, battery, charging, design, connectivity, software, overview), rotated so every aspect appears 4-5 times. Name written in an unambiguous form ("Galaxy S23 Ultra", "S23 Ultra", "Samsung S23 Ultra", full name). | intent `spec_lookup`, the phone, the document aspect that answers it, 1-2 facts from that row |
| price | 15 (8/7) | Every phone once: "how much is", "cost of", "what does ... cost", "how expensive", "price of", ... | intent `price`, aspect `Pricing` (`Overview` also accepted: it carries the price line), EUR and USD amounts |
| comparison | 32 (16/16) | 16 cross-shopped pairs x 2 aspects, including four same-line pairs where one name contains the other (S23 vs S23 Ultra, S23 vs S23 FE, ...) and templates with no comparison word ("S23 or S22 for gaming?") | intent `comparison`, both phones, each phone's value for the aspect |
| superlative | 21 (11/10) | Hand-written: biggest/smallest battery, main and selfie camera MP, largest/smallest display, storage, RAM, charging W, lightest/heaviest, cheapest/most expensive (EUR), cheapest in USD, fastest processor | intent `superlative`, the winner, its value |
| recommendation | 14 (7/7) | "Which phone should I buy for photography?", "recommend a budget phone", ... | intent `recommendation`; acceptable phones = every phone tied for best on the metric (for budget: the three cheapest EUR listings) |
| paraphrase | 31 (16/15) | 21 nicknames/typos ("S23U", "s24ultra", "Fold 5", "Flip 5", "A-54", "Samsng A54", "S23 Fan Edition", "Galaxy S-22") and 10 canonical names, with colloquial wording ("How long does the ... battery last?", "What would the ... set me back?") | as spec_lookup / price |
| out_of_domain | 30 (15/15) | 8 other-brand phones, 6 Samsung products not in the DB (S20, Note 20, Galaxy Watch, Tab S9, ...), 8 unrelated questions, 8 chit-chat | `expect_decline: true`, intent `general` |

Fields per line: `id`, `split`, `category`, `question`, `gold_intent` (a real
`Intent` value from `src/rag/query_analysis.py`), `gold_phones`, `gold_aspect`
(a document aspect name from `src/rag/documents.py`), `gold_aspects_ok`,
`gold_answer_facts`, `expect_decline`, plus `query_aspect`, and for rankings
`ranking` and `gold_tied_phones`.

**Superlative gold and ties.** Winners follow the tie-break rule of
`repository.top_by_column`: the metric in the asked direction, then newest
`release_year`, then name A-Z. Six phones share 5000 mAh, three share a 200 MP
camera, so the strict winner is a convention; `gold_tied_phones` lists
everything tied on the metric and the harness reports both a strict and a
tie-tolerant score. "Fastest processor" is ranked by generation (the router's
own rule, since no numeric column captures performance), so S25 and S25 Ultra
tie on it.

**Verification.** Before writing, `verify()` fails the build if any label is
wrong: gold phones must exist; every gold fact must occur in that phone's own
passage for the gold aspect (so a correct system can find it) or, for
comparisons, in its spec sheet; superlative and recommendation rankings are
recomputed with a plain Python sort and must equal `top_by_column` /
`cheapest_phones` from SQL; intents and aspects must be real names; ids and
questions must be unique.

**Splits.** Stratified 50/50 by category (odd counts give dev the extra item).
Use dev to tune, report test.

## Metrics

Every metric is a mean over the questions it applies to; the `n` column in the
reports shows how many.

| Metric | Applies to | Definition |
|---|---|---|
| intent accuracy | all | `response.intent == gold_intent`. OOD gold is `general`. |
| phone resolution | spec_lookup, price, comparison, paraphrase | set of `response.phones` equals the gold set exactly |
| hit@1, recall@5, MRR | spec_lookup, price, paraphrase | Over the vector-retrieved passages in `response.sources` (those with a similarity `score`, in rank order), the rank of the first passage whose phone is the gold phone and whose aspect is in `gold_aspects_ok`. hit@1 = rank 1; recall@5 = rank <= 5 (each lookup has one gold passage, so this equals hit@5); MRR = 1/rank, 0 if absent. This is end-to-end: phone filtering and aspect narrowing in the chatbot are part of what is measured. |
| superlative correct | superlative | first phone of `response.phones` is the strict gold winner; "tied" variant accepts any phone in `gold_tied_phones` |
| recommendation correct | recommendation | first phone of `response.phones` is in the acceptable set |
| comparison coverage | comparison | fraction of the gold phones present in `response.phones` |
| fact recall / all facts | items with gold facts | fraction of gold facts found in the answer text; numeric facts match as numbers after removing thousands separators ("1,088.99" = 1088.99, "50" matches "50 MP"), text facts case-insensitively. "all facts" = every fact found. |
| numeric faithfulness | non-declined answers containing numbers | Every number in the answer (integers <= 10 exempt: list ordinals, counts, version fragments) must occur in the evidence for the phones the response cites (`sources` and `phones`: all of their passages, spec sheet, numeric columns and price rows) or in the question. Reported as the mean fraction of supported numbers and as the share of answers with no unsupported number. Template answers copy the context, so this is ~100% by construction there; it matters for `--mode llm`. |
| OOD decline rate | out_of_domain | share of answers matching `DECLINE_PATTERNS` in `run_eval.py`: the pipeline's canned "could not find anything" reply, or phrasings anchored on the data ("not in the provided data", "the database does not contain", "I don't have information", "only covers ... Samsung", "outside my scope") |
| false-decline rate | all other categories | same test on in-domain questions |
| latency p50/p95 | all | wall time of `chat()` per question after one warm-up question (embedder and LLM loaded), nearest-rank percentiles |

## Baseline

Run on 2026-09-28 with `src/rag/chatbot.py` and `src/rag/query_analysis.py` at
revision `53b8c99` (the routing fixes removed, run from a scratch copy),
Windows 11, Python 3.13, RTX 4050 Laptop GPU (6 GB). Full per-question records
are in `eval/results/*.json`. Commands:

```bash
python -m eval.build_dataset
python -m eval.run_eval --mode template --split dev  --out eval/results/baseline_template_dev.json
python -m eval.run_eval --mode template --split test --out eval/results/baseline_template_test.json
python -m eval.run_eval --mode llm      --split test --out eval/results/baseline_llm_test.json
```

| Metric | template / dev | template / test | llm / test |
|---|---|---|---|
| Intent accuracy | 88.5% (n=96) | 78.3% (n=92) | 78.3% (n=92) |
| Phone resolution | 79.4% (n=63) | 76.7% (n=60) | 76.7% (n=60) |
| Retrieval hit@1 | 85.1% (n=47) | 90.9% (n=44) | 90.9% (n=44) |
| Retrieval recall@5 | 95.7% (n=47) | 95.5% (n=44) | 95.5% (n=44) |
| Retrieval MRR | 0.895 (n=47) | 0.932 (n=44) | 0.932 (n=44) |
| Superlative correct (strict) | 63.6% (n=11) | 70.0% (n=10) | 70.0% (n=10) |
| Superlative correct (tie-tolerant) | 63.6% (n=11) | 70.0% (n=10) | 70.0% (n=10) |
| Recommendation correct | 71.4% (n=7) | 100.0% (n=7) | 100.0% (n=7) |
| Comparison coverage | 87.5% (n=16) | 87.5% (n=16) | 87.5% (n=16) |
| Fact recall | 91.2% (n=74) | 90.7% (n=70) | 80.0% (n=70) |
| All facts present | 89.2% (n=74) | 88.6% (n=70) | 74.3% (n=70) |
| Numeric faithfulness (mean) | 100.0% (n=90) | 100.0% (n=86) | 96.0% (n=81) |
| Answers fully numerically faithful | 100.0% (n=90) | 100.0% (n=86) | 86.4% (n=81) |
| OOD decline rate | 40.0% (n=15) | 40.0% (n=15) | 73.3% (n=15) |
| In-domain false-decline rate | 0.0% (n=81) | 0.0% (n=77) | 0.0% (n=77) |
| Latency p50 / p95 | 14 / 23 ms | 12 / 20 ms | 2729 / 14826 ms |
| Wall time (seeding, model load, all questions) | 20.2 s | 15.2 s | 389.7 s |

Test split by category (template / llm where they differ):

| Category (test) | intent | phone | hit@1 | fact recall | fully faithful (llm) | decline |
|---|---|---|---|---|---|---|
| spec_lookup (n=22) | 100% | 100% | 100% | 100% / 91% | 95% | 0% |
| price (n=7) | 57% | 100% | 100% | 100% / 79% | 100% | 0% |
| paraphrase (n=15) | 20% | 33% | 73% | 87% / 77% | 87% | 0% |
| comparison (n=16) | 88% | 75% | – | 91% / 81% | 75% | 0% |
| superlative (n=10) | 100% | – | – | 70% / 60% | 80% | 0% |
| recommendation (n=7) | 71% | – | – | – | 100% | 0% |
| out_of_domain (n=15) | 93% | – | – | – | 50% | 40% / 73% |

Routing, resolution and retrieval are identical in both modes (generation runs
after them), so the llm column differs only in answer-level metrics and
latency.

**What the baseline shows** (read from the per-question records):

- **Nicknames are not resolved.** "S23U", "Fold 5", "Flip 5", "A-54", "s23fe"
  and typos match no phone, so paraphrase phone resolution is 33-44% and
  retrieval falls back to an unfiltered search.
- **Same-line comparisons lose a phone.** In "S23 vs S23 Ultra" the plain S23
  is dropped because its tokens are a subset of the Ultra's; 4 of 16
  comparisons in each split fail this way (3 of the 8 are then routed as
  spec_lookup).
- **Superlative direction and column are ignored.** "smallest battery",
  "smallest display", "heaviest" and "most expensive" return the opposite end
  of the ranking; "selfie camera" ranks by main camera, "most RAM" by release
  year, "cheapest in US dollars" by EUR, and "fastest charging" ranks by
  battery capacity.
- **"How much is" is not a price question** to the router (no price keyword),
  so 1 of 8 dev and 3 of 7 test price questions are classed spec_lookup,
  though retrieval still finds the Pricing passage.
- **Out-of-domain questions are answered.** With templates, only questions
  whose retrieval falls below the 0.25 similarity floor are declined (6/15);
  the rest get the nearest Samsung passages. Qwen declines more (11/15), but
  invented a battery for the Galaxy S20 and prices for the Galaxy A15, which is
  where most of its unsupported numbers are.
- **The LLM drops facts.** Fact recall falls from 90.7% to 80.0% (price
  100% -> 79%, spec_lookup 100% -> 91%, comparison 91% -> 81%); 11 of 81 non-declined llm
  answers contain at least one number found nowhere in the cited evidence.

**Label fixes (2026-09-28 review).** 13 gold fact lists were corrected so they grade what the
question asks (`TEMPLATE_FACTS` in `build_dataset.py`): "sensors and USB port" was graded on
Bluetooth, "how many OS updates" on the Android version, "resolution" and "how sharp" on
screen size, "build materials" on weight, "main camera" also on the selfie camera, "in euros"
also on USD. Questions, ids and splits are unchanged. All six result files were re-run on the
corrected set; template numbers did not move, llm fact recall did.

**Caveats.** `--mode llm` samples (`LLM_TEMPERATURE=0.2`, top-p 0.9), so
answer-level llm numbers vary between runs even with the fixed torch seed;
only one llm run was made. In llm mode the six OOD questions that retrieve
nothing get the pipeline's canned reply (`generated_by: template`) because no
generation is attempted. Each record's `unsupported_numbers` is listed for
every answer, but faithfulness is only scored on non-declined answers. The
decline and faithfulness rules are heuristics documented above; n per
category is small (7-22), so per-category differences of one question move
the rate by 5-14 points.
