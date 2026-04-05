"""
RAG retrieval eval harness.

Runs a list of eval queries through the hybrid search endpoint, scores each
against its expected-keyword set, and writes per-query JSON + a summary CSV.

Configure your eval queries in eval_queries.json (a list of objects with
`query`, `expected_keywords`, optional `domain`, optional `notes`). See
eval_queries.example.json for the schema.

Usage:
  python rag_eval.py --server http://localhost:9000 --queries eval_queries.json
"""

import argparse
import csv
import json
import os
from datetime import datetime

import requests


def load_queries(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def score_result(result_text: str, expected_keywords: list[str]) -> dict:
    """Case-insensitive keyword presence scoring. Returns hits, total, fraction."""
    text_lower = result_text.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in text_lower]
    total = len(expected_keywords)
    return {
        "hits": hits,
        "miss": [kw for kw in expected_keywords if kw not in hits],
        "fraction": len(hits) / total if total else 0.0,
    }


def run_eval(server: str, queries: list[dict], top_k: int, dense_n: int, bm25_n: int, rrf_k: int, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q['query'][:80]}")
        resp = requests.post(
            f"{server}/search",
            json={
                "query": q["query"],
                "top_k": top_k,
                "dense_top_n": dense_n,
                "bm25_top_n": bm25_n,
                "rrf_k": rrf_k,
                "domain": q.get("domain"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        combined = " ".join(r.get("text", "") for r in data["stages"]["final"])
        scoring = score_result(combined, q.get("expected_keywords", []))

        record = {
            "query": q["query"],
            "domain": q.get("domain"),
            "notes": q.get("notes", ""),
            "expected_keywords": q.get("expected_keywords", []),
            "hits": scoring["hits"],
            "miss": scoring["miss"],
            "fraction": scoring["fraction"],
            "timings": data["timings"],
            "final": data["stages"]["final"],
        }
        results.append(record)

        safe_name = "".join(c if c.isalnum() else "_" for c in q["query"][:60])
        with open(f"{out_dir}/{safe_name}.json", "w") as f:
            json.dump(record, f, indent=2)

    return results


def write_summary(results: list[dict], out_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary_path = f"{out_dir}/eval_summary_{ts}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "domain", "fraction", "hits_count", "miss_count", "total_s", "rerank_s"])
        for r in results:
            w.writerow([
                r["query"],
                r["domain"] or "",
                f"{r['fraction']:.3f}",
                len(r["hits"]),
                len(r["miss"]),
                r["timings"].get("total", ""),
                r["timings"].get("rerank", ""),
            ])
    avg_frac = sum(r["fraction"] for r in results) / len(results) if results else 0.0
    avg_total = sum(r["timings"].get("total", 0) for r in results) / len(results) if results else 0.0
    print(f"\nMean keyword-hit fraction: {avg_frac:.3f}")
    print(f"Mean total latency:        {avg_total:.2f}s")
    print(f"Summary: {summary_path}")
    return summary_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.environ.get("RAG_SERVER", "http://localhost:9000"))
    p.add_argument("--queries", default="eval_queries.json", help="JSON file with eval queries")
    p.add_argument("--out-dir", default="eval_results", help="Directory for per-query JSON + summary CSV")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dense-n", type=int, default=20)
    p.add_argument("--bm25-n", type=int, default=20)
    p.add_argument("--rrf-k", type=int, default=60)
    args = p.parse_args()

    queries = load_queries(args.queries)
    print(f"Running {len(queries)} eval queries against {args.server}")
    results = run_eval(args.server, queries, args.top_k, args.dense_n, args.bm25_n, args.rrf_k, args.out_dir)
    write_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
