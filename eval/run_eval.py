"""Run the real chatbot pipeline over the evaluation set and score it.

    python -m eval.run_eval --mode template --split dev
    python -m eval.run_eval --mode llm --split test --out eval/results/baseline_llm_test.json
    python -m eval.run_eval --mode template --split all --limit 20

Nothing is mocked: questions go through `SamsungChatbot.chat()` with the real
MiniLM embeddings and FAISS index, against a scratch SQLite database seeded
from `data/scraped_phones.json` (see `eval/_env.py`). `--mode llm` generates
with the configured local model (`LLM_MODEL`); `--mode template` uses the
deterministic template answers.

Writes `<out>.json` (every metric plus a per-question record) and `<out>.md`
(the summary tables). Metric definitions live in `eval/README.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Categories whose question names the phone(s) outright.
NAMED_PHONE_CATEGORIES = ("spec_lookup", "price", "comparison", "paraphrase")
# Single-phone lookups with one gold passage, scored on retrieval.
LOOKUP_CATEGORIES = ("spec_lookup", "price", "paraphrase")

# An answer counts as a decline when it matches any of these (lower-cased).
# The first is the pipeline's own canned reply; the rest catch an LLM saying
# the data does not cover the question. They are anchored on the *data* or
# *database* so "the S23 does not have a headphone jack" is not a decline.
DECLINE_PATTERNS = [
    r"could not find anything",
    r"\b(?:not|no)\b[^.]{0,60}\b(?:in|from|within) the (?:provided |reference |specification )?(?:database|data|reference data|information)",
    r"(?:database|reference data|data provided|provided data|provided information)[^.]{0,40}\b(?:does not|doesn't|do not|don't) (?:contain|include|cover|have|mention|list)",
    r"\b(?:i|we) (?:don't|do not|cannot|can't) (?:have|find|provide)[^.]{0,40}\b(?:information|data|details)",
    r"\bno (?:information|data|details|specifications) (?:about|on|for|regarding)",
    r"\bonly (?:covers?|includes?|contains?|has)[^.]{0,40}samsung",
    r"\b(?:outside|beyond) (?:the|my) (?:scope|knowledge)",
]
_DECLINE = [re.compile(p) for p in DECLINE_PATTERNS]

_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# Small integers are exempt from the faithfulness check: they are list
# ordinals ("1."), counts ("two phones" written as 2) and version fragments far
# more often than claims about a spec.
FAITHFULNESS_EXEMPT_MAX = 10


def is_decline(answer: str) -> bool:
    lowered = answer.lower()
    return any(pattern.search(lowered) for pattern in _DECLINE)


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in _NUMBER.findall(text)]


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil without math
    return ordered[int(rank) - 1]


# --------------------------------------------------------------------------
# Scoring one response
# --------------------------------------------------------------------------
def retrieval_rank(item: dict, sources: list[dict]) -> int | None:
    """1-based rank of the first acceptable (phone, aspect) passage, else None.

    Only vector-retrieved passages (the ones carrying a similarity `score`)
    are ranked; ranking tables and spec sheets are not retrieval results.
    """
    retrieved = [s for s in sources if "score" in s]
    gold_phone = item["gold_phones"][0]
    for rank, source in enumerate(retrieved, start=1):
        if source["phone"] == gold_phone and source["aspect"] in item["gold_aspects_ok"]:
            return rank
    return None


def score_item(item: dict, response, latency_ms: float, evidence: dict[str, str]) -> dict:
    from eval.build_dataset import fact_in_text

    answer = response.answer
    pred_phones = list(response.phones)
    declined = is_decline(answer)
    scores: dict[str, float | None] = {
        "intent_correct": float(response.intent == item["gold_intent"]),
        "declined": float(declined),
    }

    category = item["category"]
    if category in NAMED_PHONE_CATEGORIES:
        scores["phone_correct"] = float(set(pred_phones) == set(item["gold_phones"]))

    if category in LOOKUP_CATEGORIES:
        rank = retrieval_rank(item, response.sources)
        scores["hit@1"] = float(rank == 1)
        scores["recall@5"] = float(rank is not None and rank <= 5)
        scores["mrr"] = 1.0 / rank if rank else 0.0

    if category == "superlative":
        top = pred_phones[0] if pred_phones else None
        scores["superlative_correct"] = float(top == item["gold_phones"][0])
        scores["superlative_correct_tied"] = float(top in item["gold_tied_phones"])

    if category == "recommendation":
        top = pred_phones[0] if pred_phones else None
        scores["recommendation_correct"] = float(top in item["gold_phones"])

    if category == "comparison":
        gold = set(item["gold_phones"])
        scores["comparison_coverage"] = len(gold & set(pred_phones)) / len(gold)

    facts = item["gold_answer_facts"]
    if facts and not item["expect_decline"]:
        found = [fact_in_text(fact, answer) for fact in facts]
        scores["fact_recall"] = sum(found) / len(found)
        scores["facts_all_present"] = float(all(found))

    # Numeric faithfulness: every number in the answer must occur in the
    # evidence for the phones the response cites (their passages, spec sheet,
    # numeric columns and prices) or in the question itself.
    cited = {s.get("phone") for s in response.sources} | set(pred_phones)
    support = item["question"] + "\n" + "\n".join(evidence.get(name, "") for name in cited)
    supported_numbers = set(numbers_in(support))
    claimed = [n for n in numbers_in(answer) if not (n.is_integer() and n <= FAITHFULNESS_EXEMPT_MAX)]
    unsupported = sorted({n for n in claimed if n not in supported_numbers})
    if claimed and not declined:
        scores["numeric_faithfulness"] = 1 - sum(n in unsupported for n in claimed) / len(claimed)
        scores["numerically_faithful"] = float(not unsupported)

    return {
        "id": item["id"],
        "split": item["split"],
        "category": category,
        "question": item["question"],
        "gold_intent": item["gold_intent"],
        "gold_phones": item["gold_phones"],
        "pred_intent": response.intent,
        "pred_phones": pred_phones,
        "pred_aspects": response.aspects,
        "top_sources": response.sources[:5],
        "generated_by": response.generated_by,
        "unsupported_numbers": unsupported,
        "latency_ms": round(latency_ms, 1),
        "scores": scores,
        "answer": answer,
    }


# --------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------
METRIC_ORDER = [
    "intent_correct",
    "phone_correct",
    "hit@1",
    "recall@5",
    "mrr",
    "superlative_correct",
    "superlative_correct_tied",
    "recommendation_correct",
    "comparison_coverage",
    "fact_recall",
    "facts_all_present",
    "numeric_faithfulness",
    "numerically_faithful",
]


def aggregate(records: list[dict]) -> dict:
    out: dict[str, dict] = {"n": len(records)}
    for metric in METRIC_ORDER:
        values = [r["scores"][metric] for r in records if r["scores"].get(metric) is not None]
        if values:
            out[metric] = {"mean": round(sum(values) / len(values), 4), "n": len(values)}

    ood = [r for r in records if r["category"] == "out_of_domain"]
    ind = [r for r in records if r["category"] != "out_of_domain"]
    if ood:
        out["ood_decline_rate"] = {
            "mean": round(sum(r["scores"]["declined"] for r in ood) / len(ood), 4),
            "n": len(ood),
        }
    if ind:
        out["false_decline_rate"] = {
            "mean": round(sum(r["scores"]["declined"] for r in ind) / len(ind), 4),
            "n": len(ind),
        }

    latencies = [r["latency_ms"] for r in records]
    out["latency_ms"] = {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "mean": round(statistics.fmean(latencies), 1) if latencies else None,
    }
    return out


def _git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=EVAL_DIR.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=EVAL_DIR.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{rev}{'+dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _fmt(entry: dict | None, pct: bool = True) -> str:
    if not entry:
        return "–"
    value = entry["mean"]
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def render_markdown(result: dict) -> str:
    meta, overall, per_cat = result["meta"], result["overall"], result["per_category"]
    lines = [
        f"# Evaluation — {meta['mode']} mode, {meta['split']} split",
        "",
        f"- Questions: {overall['n']}  ·  generated by: {meta['generated_by']}",
        f"- Revision: `{meta['revision']}`  ·  run at {meta['started_at']} UTC  ·  wall time {meta['wall_seconds']} s",
        f"- Embeddings: `{meta['embedding_model']}`  ·  LLM: `{meta['llm_model'] if meta['mode'] == 'llm' else 'none (templates)'}`",
        f"- Command: `{meta['command']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value | n |",
        "|---|---|---|",
    ]
    rows = [(m, overall.get(m)) for m in METRIC_ORDER] + [
        ("ood_decline_rate", overall.get("ood_decline_rate")),
        ("false_decline_rate", overall.get("false_decline_rate")),
    ]
    for metric, entry in rows:
        if entry:
            lines.append(f"| {metric} | {_fmt(entry, metric != 'mrr')} | {entry['n']} |")
    lat = overall["latency_ms"]
    lines += [f"| latency p50 / p95 | {lat['p50']:.0f} / {lat['p95']:.0f} ms | {overall['n']} |", ""]

    columns = [
        ("intent", "intent_correct"), ("phone", "phone_correct"), ("hit@1", "hit@1"),
        ("R@5", "recall@5"), ("MRR", "mrr"), ("super", "superlative_correct"),
        ("super(tied)", "superlative_correct_tied"), ("rec", "recommendation_correct"),
        ("cmp cov", "comparison_coverage"), ("facts", "fact_recall"),
        ("num faithful", "numerically_faithful"), ("decline", "decline"),
    ]
    lines += [
        "## Per category",
        "",
        "| Category | n | " + " | ".join(c for c, _ in columns) + " | p50 ms |",
        "|---|---|" + "---|" * len(columns) + "---|",
    ]
    for category, agg in per_cat.items():
        cells = []
        for _, metric in columns:
            if metric == "decline":
                entry = agg.get("ood_decline_rate") or agg.get("false_decline_rate")
                cells.append(_fmt(entry))
            else:
                cells.append(_fmt(agg.get(metric), metric != "mrr"))
        lines.append(f"| {category} | {agg['n']} | " + " | ".join(cells) + f" | {agg['latency_ms']['p50']:.0f} |")
    lines += [
        "",
        "`decline` is the decline rate for out_of_domain and the false-decline rate elsewhere.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def build_evidence() -> dict[str, str]:
    """Every number the database holds about each phone, as searchable text."""
    from src.database.db import session_scope
    from src.database.repository import get_all_phones
    from src.rag.documents import build_documents_for_phone

    evidence: dict[str, str] = {}
    with session_scope() as session:
        for phone in get_all_phones(session):
            parts = [d.text for d in build_documents_for_phone(phone)]
            parts.append(phone.spec_summary())
            parts.append(" ".join(str(v) for v in phone.to_dict().values() if isinstance(v, (int, float))))
            parts.append(" ".join(f"{p.amount:.2f}" for p in phone.prices))
            evidence[phone.name] = "\n".join(parts)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Samsung phone chatbot")
    parser.add_argument("--mode", choices=("template", "llm"), default="template")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--out", type=Path, default=None, help="JSON path; .md is written beside it")
    args = parser.parse_args()

    # Must happen before config is imported (eval._env imports nothing of ours).
    os.environ["USE_LLM"] = "true" if args.mode == "llm" else "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from eval import _env

    import config
    from src.rag.chatbot import SamsungChatbot

    items = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if args.split != "all":
        items = [item for item in items if item["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    out = args.out or RESULTS_DIR / f"{args.mode}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    wall_start = time.perf_counter()
    phones = _env.seed_database()
    print(f"Seeded {phones} phones into {_env.DB_PATH}")

    chatbot = SamsungChatbot(use_llm=args.mode == "llm")
    chatbot.prepare()
    evidence = build_evidence()

    if args.mode == "llm":
        import torch

        torch.manual_seed(20260928)
        from src.llm.provider import get_llm

        if get_llm() is None:
            print("LLM could not be loaded; refusing to report template answers as llm mode")
            return 1

    # Warm-up: loads the embedder (and the LLM) so latency measures answering.
    chatbot.chat("What is the battery capacity of the Galaxy S23?")

    records = []
    for number, item in enumerate(items, start=1):
        began = time.perf_counter()
        response = chatbot.chat(item["question"])
        latency_ms = (time.perf_counter() - began) * 1000
        record = score_item(item, response, latency_ms, evidence)
        records.append(record)
        flags = " ".join(k for k, v in record["scores"].items() if v == 0.0 and k != "declined")
        print(f"[{number:3d}/{len(items)}] {item['id']:20s} {latency_ms:7.0f} ms  {flags}")

    categories = sorted({r["category"] for r in records})
    generated_by = sorted({r["generated_by"] for r in records})
    result = {
        "meta": {
            "mode": args.mode,
            "split": args.split,
            "limit": args.limit,
            "dataset": str(args.dataset.relative_to(EVAL_DIR.parent)) if args.dataset.is_relative_to(EVAL_DIR.parent) else str(args.dataset),
            "command": "python -m eval.run_eval " + " ".join(sys.argv[1:]),
            "revision": _git_revision(),
            "started_at": started_at,
            "wall_seconds": round(time.perf_counter() - wall_start, 1),
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "llm_temperature": config.LLM_TEMPERATURE,
            "rag_top_k": config.RAG_TOP_K,
            "generated_by": ", ".join(generated_by),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "overall": aggregate(records),
        "per_category": {c: aggregate([r for r in records if r["category"] == c]) for c in categories},
        "items": records,
    }
    if args.mode == "llm":
        import torch

        result["meta"]["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
