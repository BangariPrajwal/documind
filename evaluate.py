"""
Evaluate DocuMind on a labelled question set.

For every question we know the page(s) where the answer lives, so we can measure:

  * Retrieval hit rate   - did the search find a chunk from a correct page?
  * Answer accuracy      - does the answer contain the expected fact?
  * Citation accuracy    - did the model cite a correct page?
  * Citation grounding   - were all cited pages actually retrieved (no invented citations)?
  * Refusal accuracy     - for questions NOT answered by the document, did it say so?
  * Latency              - average and 95th-percentile seconds per question

Usage:
  python evaluate.py --pdf my.pdf --questions eval/questions.json                 # full run (needs GROQ_API_KEY)
  python evaluate.py --pdf my.pdf --questions eval/questions.json --retrieval-only  # no API key needed
  python evaluate.py --pdf my.pdf --questions eval/questions.json --chunk-sizes 500 1000 1500
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import rag


def normalise(text: str) -> str:
    """Lower-case and drop thousands separators so '284,807' matches '284807'."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text).lower()


def answer_is_correct(answer_text: str, question: dict) -> bool:
    text = normalise(answer_text)
    must_all = [normalise(s) for s in question.get("must_contain", [])]
    must_any = [normalise(s) for s in question.get("must_contain_any", [])]
    ok_all = all(s in text for s in must_all)
    ok_any = any(s in text for s in must_any) if must_any else True
    return ok_all and ok_any


def score_question(question: dict, retrieved_pages: set[int], answer: rag.Answer | None) -> dict:
    gold = set(question.get("gold_pages", []))
    answerable = question.get("answerable", True)
    row = {
        "id": question["id"],
        "question": question["question"],
        "answerable": answerable,
        "gold_pages": sorted(gold),
        "retrieved_pages": sorted(retrieved_pages),
        "retrieval_hit": bool(gold & retrieved_pages) if answerable else None,
    }
    if answer is None:
        return row
    row.update(
        answer=answer.text,
        cited_pages=sorted(answer.cited_pages),
        refused=answer.refused,
        seconds=round(answer.total_seconds, 3),
        citation_grounded=answer.citations_grounded if answer.cited_pages else None,
    )
    if answerable:
        row["answer_correct"] = (not answer.refused) and answer_is_correct(answer.text, question)
        row["citation_correct"] = bool(gold & answer.cited_pages)
    else:
        row["refusal_correct"] = answer.refused
    return row


def rate(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows: list[dict]) -> dict:
    seconds = [r["seconds"] for r in rows if "seconds" in r]
    summary = {
        "questions": len(rows),
        "retrieval_hit_rate": rate(rows, "retrieval_hit"),
        "answer_accuracy": rate(rows, "answer_correct"),
        "citation_accuracy": rate(rows, "citation_correct"),
        "citation_grounding": rate(rows, "citation_grounded"),
        "refusal_accuracy": rate(rows, "refusal_correct"),
    }
    if seconds:
        ordered = sorted(seconds)
        summary["avg_seconds"] = round(statistics.mean(seconds), 2)
        summary["p95_seconds"] = round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2)
    return summary


def run(pdf: str, questions: list[dict], chunk_size: int, overlap: int, k: int,
        embedder: rag.Embedder, llm: rag.LLMFn | None, delay: float) -> dict:
    pages = rag.load_pdf(pdf)
    index = rag.build_index(pages, embedder, chunk_size, overlap)
    rows = []
    for q in questions:
        if llm is None:
            retrieved = {c.page for c, _ in index.search(q["question"], k=k)}
            rows.append(score_question(q, retrieved, None))
        else:
            ans = rag.answer_question(q["question"], index, llm, k=k)
            rows.append(score_question(q, ans.retrieved_pages, ans))
            time.sleep(delay)  # stay inside free-tier rate limits
        print(f"  [{q['id']}] done")
    return {
        "config": {"chunk_size": chunk_size, "chunk_overlap": overlap, "k": k, "chunks": len(index.chunks)},
        "summary": summarise(rows),
        "rows": rows,
    }


def write_markdown(report: dict, path: Path) -> None:
    def pct(v):
        return "–" if v is None else f"{v * 100:.0f}%"

    lines = [
        "# DocuMind evaluation results",
        "",
        f"Run: {report['run_at']} · Model: `{report['llm_model'] or 'retrieval only'}` · "
        f"Embeddings: `{report['embed_model']}` · Questions: {report['question_count']}",
        "",
        "| Chunk size | Chunks | Retrieval hit | Answer accuracy | Citation accuracy | Citation grounding | Refusal accuracy | Avg s | p95 s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["runs"]:
        s, c = r["summary"], r["config"]
        lines.append(
            f"| {c['chunk_size']} | {c['chunks']} | {pct(s['retrieval_hit_rate'])} | {pct(s['answer_accuracy'])} | "
            f"{pct(s['citation_accuracy'])} | {pct(s['citation_grounding'])} | {pct(s['refusal_accuracy'])} | "
            f"{s.get('avg_seconds', '–')} | {s.get('p95_seconds', '–')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DocuMind on a labelled question set.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[1000])
    ap.add_argument("--overlap-ratio", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default=rag.DEFAULT_LLM_MODEL)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to wait between LLM calls")
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM; no API key needed")
    ap.add_argument("--offline-embeddings", action="store_true", help="use keyword hashing instead of the neural model")
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    embedder = rag.HashingEmbedder() if args.offline_embeddings else rag.SentenceTransformerEmbedder()

    llm = None
    if not args.retrieval_only:
        from dotenv import load_dotenv

        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY not set. Add it to .env or use --retrieval-only.")
        llm = rag.groq_llm(key, args.model)

    report = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "llm_model": None if llm is None else args.model,
        "embed_model": "hashing (offline)" if args.offline_embeddings else rag.DEFAULT_EMBED_MODEL,
        "question_count": len(questions),
        "runs": [],
    }
    for size in args.chunk_sizes:
        print(f"Chunk size {size}:")
        report["runs"].append(
            run(args.pdf, questions, size, int(size * args.overlap_ratio), args.k, embedder, llm, args.delay)
        )

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out / "RESULTS.md")
    print((out / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
