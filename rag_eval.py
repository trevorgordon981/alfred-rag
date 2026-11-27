#!/usr/bin/env python3
"""
RAG Retrieval Eval Harness

Evaluates the served hybrid retrieval pipeline via `/search/debug` (which shares
the exact ranking implementation with `/search`) against labeled query files.

Supports:
  - Exact document matching (`expected_doc_ids` -> Hit@1, Hit@k, MRR)
  - Quoted source-span verification (`expected_spans`)
  - Out-of-domain / no-result negative controls (`expect_no_results` + `min_score`)
  - Keyword-hit diagnostics across stages (`expected_keywords`)
  - Provenance metadata (git commit, query file hash, server models, timestamp)

Usage:
    python rag_eval.py --server http://127.0.0.1:9000 --queries fixtures/demo_eval_queries.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = os.environ.get("RAG_SERVER", "http://127.0.0.1:9000")
DEFAULT_QUERIES = str(REPO_ROOT / "fixtures" / "demo_eval_queries.json")
DEFAULT_OUTPUT_DIR = "eval_results"
DEFAULT_DENSE_TOP_N = 30
DEFAULT_BM25_TOP_N = 30
DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 5
DEFAULT_MAX_PER_SOURCE = 2
DEFAULT_MIN_SCORE = 0.45


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_eval_queries(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Query file must contain a JSON list: {path}")
    return data


def _auth_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def run_debug_search(
    query: str,
    domain: str | None = None,
    *,
    server: str = DEFAULT_SERVER,
    dense_top_n: int = DEFAULT_DENSE_TOP_N,
    bm25_top_n: int = DEFAULT_BM25_TOP_N,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_TOP_K,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
    rerank: bool = True,
    api_key: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "dense_top_n": dense_top_n,
        "bm25_top_n": bm25_top_n,
        "rrf_k": rrf_k,
        "top_k": top_k,
        "max_per_source": max_per_source,
        "rerank": rerank,
    }
    if domain:
        payload["domain"] = domain
    resp = requests.post(
        f"{server.rstrip('/')}/search/debug",
        json=payload,
        headers=_auth_headers(api_key),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def compute_diagnostics(sr: dict[str, Any], eq: dict[str, Any]) -> dict[str, Any]:
    """Compute per-stage keyword hits, exact doc_id retrieval, span recall, and OOD checks."""
    diag: dict[str, Any] = {}
    stages = sr.get("stages", {})
    expected_kw = [kw.lower() for kw in eq.get("expected_keywords", [])]
    expected_docs = set(eq.get("expected_doc_ids", []))
    expected_spans = list(eq.get("expected_spans", []))
    expect_no_results = bool(eq.get("expect_no_results", False))
    min_score = float(eq.get("min_score", DEFAULT_MIN_SCORE))
    query_domain = eq.get("domain")

    for stage_name, chunks in stages.items():
        keyword_hits = []
        for chunk in chunks:
            text_lower = chunk.get("text", "").lower()
            matched = [kw for kw in expected_kw if kw in text_lower]
            if matched:
                keyword_hits.append(
                    {
                        "id": chunk.get("id", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "rank": chunk.get("rank", 0),
                        "domain": chunk.get("domain", ""),
                        "matched_keywords": matched,
                        "rerank_score": chunk.get("rerank_score"),
                        "rrf_score": chunk.get("rrf_score"),
                        "bm25_score": chunk.get("bm25_score"),
                    }
                )
        diag[stage_name] = {
            "total_chunks": len(chunks),
            "keyword_hits": keyword_hits,
            "keyword_hit_count": len(keyword_hits),
        }

    final = stages.get("final", [])
    if query_domain and final:
        wrong = [c for c in final if c.get("domain") != query_domain]
        diag["cross_domain"] = {
            "wrong_count": len(wrong),
            "total": len(final),
            "details": [
                {
                    "id": c["id"],
                    "doc_id": c.get("doc_id", ""),
                    "domain": c.get("domain"),
                    "rank": c["rank"],
                    "rerank_score": c.get("rerank_score"),
                }
                for c in wrong
            ],
        }

    # Exact document matching (Hit@1, Hit@k, MRR)
    first_doc_rank: int | None = None
    if expected_docs:
        for idx, chunk in enumerate(final, start=1):
            cid_doc = chunk.get("doc_id") or chunk.get("source") or ""
            if cid_doc in expected_docs:
                first_doc_rank = idx
                break
    diag["doc_match"] = {
        "expected_doc_ids": sorted(expected_docs),
        "hit_at_1": first_doc_rank == 1 if expected_docs else None,
        "hit_at_k": first_doc_rank is not None if expected_docs else None,
        "first_rank": first_doc_rank,
        "mrr": round(1.0 / first_doc_rank, 4) if first_doc_rank else (0.0 if expected_docs else None),
    }

    # Verbatim source-span verification
    if expected_spans:
        combined_final_text = "\n".join(c.get("text", "") for c in final)
        matched_spans = [sp for sp in expected_spans if sp in combined_final_text]
        diag["span_match"] = {
            "expected": len(expected_spans),
            "matched": len(matched_spans),
            "all_matched": len(matched_spans) == len(expected_spans),
            "missing": [sp for sp in expected_spans if sp not in combined_final_text],
        }
    else:
        diag["span_match"] = {"expected": 0, "matched": 0, "all_matched": None, "missing": []}

    # Out-of-domain / no-result threshold check
    if expect_no_results:
        above_threshold = [
            c for c in final if float(c.get("rerank_score", 0.0)) >= min_score
        ]
        diag["no_result_check"] = {
            "expect_no_results": True,
            "min_score": min_score,
            "above_threshold_count": len(above_threshold),
            "passed": len(above_threshold) == 0,
            "max_score": max((float(c.get("rerank_score", 0.0)) for c in final), default=0.0),
        }
    else:
        diag["no_result_check"] = {"expect_no_results": False, "passed": None}

    reranked = stages.get("reranked", [])
    if reranked:
        scores = [c.get("rerank_score", 0.0) for c in reranked]
        diag["reranker_scores"] = {
            "max": max(scores),
            "min": min(scores),
            "spread": round(max(scores) - min(scores), 4),
            "top5": scores[:5],
            "bottom5": scores[-5:],
            "top1_vs_top5_gap": round(scores[0] - scores[4], 4) if len(scores) >= 5 else None,
        }

    dense_hits = {h["id"] for h in diag.get("dense", {}).get("keyword_hits", [])}
    bm25_hits = {h["id"] for h in diag.get("bm25", {}).get("keyword_hits", [])}
    rrf_hits = {h["id"] for h in diag.get("rrf_merged", {}).get("keyword_hits", [])}
    final_hits = {h["id"] for h in diag.get("final", {}).get("keyword_hits", [])}
    retrieval_hits = dense_hits | bm25_hits
    diag["dropout_analysis"] = {
        "in_dense": len(dense_hits),
        "in_bm25": len(bm25_hits),
        "in_rrf": len(rrf_hits),
        "in_final": len(final_hits),
        "dropped_by_reranker": sorted(retrieval_hits - final_hits),
        "never_retrieved": not bool(retrieval_hits) and bool(expected_kw),
    }

    # Overall query pass status
    if expect_no_results:
        passed = bool(diag["no_result_check"]["passed"])
    elif expected_docs:
        span_ok = True if not expected_spans else bool(diag["span_match"]["all_matched"])
        passed = bool(diag["doc_match"]["hit_at_k"]) and span_ok
    else:
        passed = len(final_hits) > 0
    diag["passed"] = passed
    return diag


def summarize_eval_run(
    results: list[dict[str, Any]],
    *,
    queries_path: str | None = None,
    server: str | None = None,
    health_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate Hit@1, Hit@k, MRR, span recall, OOD rejection, and provenance."""
    doc_items = [r for r in results if r["diagnostics"]["doc_match"]["hit_at_k"] is not None]
    span_items = [r for r in results if r["diagnostics"]["span_match"]["expected"] > 0]
    ood_items = [r for r in results if r["diagnostics"]["no_result_check"]["expect_no_results"]]

    hit1 = sum(1 for r in doc_items if r["diagnostics"]["doc_match"]["hit_at_1"])
    hitk = sum(1 for r in doc_items if r["diagnostics"]["doc_match"]["hit_at_k"])
    mrr = (
        round(
            sum(r["diagnostics"]["doc_match"]["mrr"] or 0.0 for r in doc_items) / len(doc_items),
            4,
        )
        if doc_items
        else None
    )
    span_pass = sum(1 for r in span_items if r["diagnostics"]["span_match"]["all_matched"])
    ood_pass = sum(1 for r in ood_items if r["diagnostics"]["no_result_check"]["passed"])
    total_pass = sum(1 for r in results if r["diagnostics"]["passed"])

    return {
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "queries_file": queries_path,
            "queries_sha256": _file_sha256(queries_path) if queries_path and os.path.isfile(queries_path) else None,
            "server": server,
            "models": (health_payload or {}).get("models"),
            "indexes": (health_payload or {}).get("indexes"),
            "index_generation": (health_payload or {}).get("generation"),
        },
        "metrics": {
            "total_queries": len(results),
            "passed_queries": total_pass,
            "pass_rate": round(total_pass / len(results), 4) if results else 0.0,
            "doc_labeled_queries": len(doc_items),
            "doc_hit_at_1": hit1,
            "doc_hit_at_k": hitk,
            "doc_mrr": mrr,
            "span_labeled_queries": len(span_items),
            "span_all_matched": span_pass,
            "ood_queries": len(ood_items),
            "ood_rejected": ood_pass,
        },
    }


def truncate(text: str, max_len: int = 120) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def print_summary(eq: dict[str, Any], sr: dict[str, Any], diag: dict[str, Any]) -> None:
    status = "PASS" if diag.get("passed") else "FAIL"
    print(f"\n{'=' * 80}")
    print(f"[{status}] QUERY: {eq['query']}")
    if eq.get("notes"):
        print(f"NOTES: {eq['notes']}")
    print(f"TIMINGS: {sr['timings']}")

    final = sr["stages"].get("final", [])
    print(f"\n--- FINAL ({len(final)} chunks) ---")
    for chunk in final:
        domain_tag = f"[{chunk.get('domain', '')}]"
        doc_tag = chunk.get("doc_id") or chunk.get("source") or ""
        score = chunk.get("rerank_score")
        score_str = f"rerank={score:.4f}" if score is not None else ""
        print(f"  #{chunk.get('rank', '?'):2}  {score_str:14s}  {domain_tag:14s}  {doc_tag}")
        print(f"       {truncate(chunk.get('text', ''))}")

    dm = diag.get("doc_match", {})
    sm = diag.get("span_match", {})
    nrc = diag.get("no_result_check", {})
    if dm.get("expected_doc_ids"):
        print(f"  DOC MATCH: hit@1={dm['hit_at_1']} hit@k={dm['hit_at_k']} rank={dm['first_rank']}")
    if sm.get("expected"):
        print(f"  SPAN MATCH: {sm['matched']}/{sm['expected']}")
    if nrc.get("expect_no_results"):
        print(
            f"  OOD CHECK: passed={nrc['passed']} max_score={nrc['max_score']:.4f} "
            f"(threshold={nrc['min_score']})"
        )


def save_result(
    eq: dict[str, Any], sr: dict[str, Any], diag: dict[str, Any], output_dir: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe = (
        "".join(c if c.isalnum() or c == " " else "_" for c in eq["query"])[:60]
        .strip()
        .replace(" ", "_")
    )
    path = os.path.join(output_dir, f"{safe}.json")
    data = {
        "id": eq.get("id"),
        "query": eq["query"],
        "domain": eq.get("domain"),
        "expected_doc_ids": eq.get("expected_doc_ids", []),
        "expected_spans": eq.get("expected_spans", []),
        "expected_keywords": eq.get("expected_keywords", []),
        "expect_no_results": bool(eq.get("expect_no_results", False)),
        "notes": eq.get("notes", ""),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timings": sr["timings"],
        "counts": sr["counts"],
        "diagnostics": diag,
        "stages": {
            sn: [
                {
                    **{k: v for k, v in c.items() if k != "text"},
                    "text_preview": truncate(c.get("text", ""), 200),
                    "text_full": c.get("text", ""),
                }
                for c in chunks
            ]
            for sn, chunks in sr["stages"].items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def write_summary_csv(results: list[dict[str, Any]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_summary.csv")
    rows = []
    for item in results:
        eq, sr, diag = item["eval_query"], item["server_response"], item["diagnostics"]
        dropout = diag.get("dropout_analysis", {})
        dm = diag.get("doc_match", {})
        sm = diag.get("span_match", {})
        rs = diag.get("reranker_scores", {})
        t = sr.get("timings", {})
        final = sr["stages"].get("final", [])
        top_doc = (final[0].get("doc_id") or final[0].get("source") or "") if final else ""
        top_domain = final[0].get("domain", "") if final else ""
        top_score = final[0].get("rerank_score", 0.0) if final else 0.0
        rows.append(
            {
                "id": eq.get("id", ""),
                "query": eq["query"],
                "passed": diag.get("passed", False),
                "top_doc_id": top_doc,
                "top_domain": top_domain,
                "top_score": round(float(top_score), 4),
                "doc_hit_at_1": dm.get("hit_at_1"),
                "doc_first_rank": dm.get("first_rank"),
                "span_matched": f"{sm.get('matched', 0)}/{sm.get('expected', 0)}",
                "kw_in_final": dropout.get("in_final", 0),
                "dropped": len(dropout.get("dropped_by_reranker", [])),
                "spread": rs.get("spread", ""),
                "rerank_s": t.get("rerank", 0),
                "total_s": t.get("total", 0),
                "notes": eq.get("notes", ""),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alfred-rag retrieval against labeled queries.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Base URL of rag_server.py")
    parser.add_argument("--queries", default=DEFAULT_QUERIES, help="Path to JSON query file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/CSV eval outputs")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-top-n", type=int, default=DEFAULT_DENSE_TOP_N)
    parser.add_argument("--bm25-top-n", type=int, default=DEFAULT_BM25_TOP_N)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    parser.add_argument("--no-rerank", action="store_true", help="Evaluate fast path (rerank=False)")
    parser.add_argument("--api-key", default=os.environ.get("RAG_API_KEY", ""), help="Optional API key")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        health_resp = requests.get(f"{args.server.rstrip('/')}/health", timeout=5)
        health_payload = health_resp.json()
    except Exception as exc:
        print(f"Cannot reach {args.server}: {exc}", file=sys.stderr)
        return 1

    queries = load_eval_queries(args.queries)
    print(f"Running {len(queries)} eval queries from {args.queries} against {args.server}\n")

    results = []
    for i, eq in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] {eq['query']}")
        try:
            sr = run_debug_search(
                eq["query"],
                eq.get("domain"),
                server=args.server,
                dense_top_n=args.dense_top_n,
                bm25_top_n=args.bm25_top_n,
                rrf_k=args.rrf_k,
                top_k=args.top_k,
                max_per_source=args.max_per_source,
                rerank=not args.no_rerank,
                api_key=args.api_key,
            )
            diag = compute_diagnostics(sr, eq)
            results.append({"eval_query": eq, "server_response": sr, "diagnostics": diag})
            print_summary(eq, sr, diag)
            save_result(eq, sr, diag, args.output_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not results:
        return 1

    csv_path = write_summary_csv(results, args.output_dir)
    report = summarize_eval_run(
        results,
        queries_path=args.queries,
        server=args.server,
        health_payload=health_payload,
    )
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    m = report["metrics"]
    print(f"\n{'=' * 80}")
    print(f"Summary CSV:  {csv_path}")
    print(f"Report JSON:  {report_path}")
    print(
        f"Passed:       {m['passed_queries']}/{m['total_queries']} "
        f"(Hit@1={m['doc_hit_at_1']}/{m['doc_labeled_queries']}, "
        f"MRR={m['doc_mrr']}, Spans={m['span_all_matched']}/{m['span_labeled_queries']}, "
        f"OOD rejected={m['ood_rejected']}/{m['ood_queries']})"
    )
    print(f"{'=' * 80}")
    return 0 if m["passed_queries"] == m["total_queries"] else 2


if __name__ == "__main__":
    sys.exit(main())
