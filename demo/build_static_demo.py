"""Build the static demo page from recorded evaluation outputs.

Live Python Spaces on Hugging Face need a paid plan, so the public demo shows
what the real pipeline answered for every evaluation question, recorded by
eval/run_eval.py. Rebuild after re-running the evaluation:

    python -m demo.build_static_demo

Output: demo/site/index.html and demo/site/data.json (a static site).
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
SITE = ROOT / "demo" / "site"

KEEP_SCORES = ("intent_correct", "phone_correct", "fact_recall", "declined", "numerically_faithful")


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf8"))["items"] if path.exists() else []


def main():
    llm = {item["id"]: item for item in load("after_llm_test.json")}
    records = []
    for item in load("after_template_dev.json") + load("after_template_test.json"):
        other = llm.get(item["id"])
        records.append({
            "id": item["id"],
            "split": item["split"],
            "category": item["category"],
            "question": item["question"],
            "intent": item["pred_intent"],
            "phones": item.get("pred_phones") or [],
            "sources": item.get("top_sources") or [],
            "latency_ms": item.get("latency_ms"),
            "scores": {k: item["scores"][k] for k in KEEP_SCORES if k in item.get("scores", {})},
            "answer": item["answer"],
            "llm_answer": other["answer"] if other else None,
            "llm_generated": other.get("generated_by") == "llm" if other else False,
        })
    records.sort(key=lambda r: (r["category"], r["id"]))
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf8")
    (SITE / "index.html").write_text((ROOT / "demo" / "index.html").read_text(encoding="utf8"), encoding="utf8")
    print(f"wrote {len(records)} recorded answers to {SITE}")


if __name__ == "__main__":
    main()
