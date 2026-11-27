#!/usr/bin/env python3
"""
5-minute (CPU-only, ~2s) offline end-to-end demo for alfred-rag.

Runs the real LanceDB + Tantivy + FastAPI stack against the public synthetic
fixture corpus (`fixtures/demo_corpus/`) with zero GPUs, zero HuggingFace
downloads, and zero private data:
  1. Indexes the synthetic corpus into real LanceDB and Tantivy stores.
  2. Verifies `/health` and `/ready` report HTTP 200 with matching counts and generation.
  3. Runs all labeled queries in `fixtures/demo_eval_queries.json` through BOTH
     `/search` and `/search/debug`, asserting 100% ID/order/score parity.
  4. Verifies deletion and emptying reconciliation across both stores.
  5. Verifies Prometheus `/metrics` exporter contract.

Usage:
    python scripts/offline_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
import rag_eval  # noqa: E402
import rag_server  # noqa: E402
from scripts import ingest as ingest_mod  # noqa: E402


def run_demo(work_dir: str | None = None) -> dict:
    fixture_corpus = REPO_ROOT / "fixtures" / "demo_corpus"
    fixture_queries = REPO_ROOT / "fixtures" / "demo_eval_queries.json"

    owns_tmp = work_dir is None
    base_dir = Path(tempfile.mkdtemp(prefix="alfred_rag_demo_")) if owns_tmp else Path(work_dir)
    try:
        corpus_copy = base_dir / "data"
        if corpus_copy.exists():
            shutil.rmtree(corpus_copy)
        shutil.copytree(fixture_corpus, corpus_copy)

        lancedb_dir = str(base_dir / "indexes" / "lancedb")
        tantivy_dir = str(base_dir / "indexes" / "tantivy")

        # 1. Build real LanceDB + Tantivy indexes offline on CPU
        build_summary = ingest_mod.ingest(
            data_dir=str(corpus_copy),
            lancedb_path=lancedb_dir,
            tantivy_path=tantivy_dir,
            lance_table_name="alfred_rag_demo",
            embed_backend="offline",
            reset=True,
        )

        # 2. Configure FastAPI server against those real indexes
        rag_server.configure_offline_runtime(
            lancedb_path=lancedb_dir,
            tantivy_path=tantivy_dir,
            lance_table_name="alfred_rag_demo",
        )
        client = TestClient(rag_server.app)

        health = client.get("/health")
        ready = client.get("/ready")
        assert health.status_code == 200, f"/health failed: {health.text}"
        assert ready.status_code == 200, f"/ready failed: {ready.text}"
        health_body = health.json()
        assert (
            health_body["indexes"]["lancedb"]
            == health_body["indexes"]["tantivy"]
            == build_summary["total_chunks"]
        )

        # 3. Evaluate labeled queries & verify /search == /search/debug parity
        queries = rag_eval.load_eval_queries(str(fixture_queries))
        eval_results = []
        parity_checks = 0
        for eq in queries:
            payload = {
                "query": eq["query"],
                "top_k": 5,
                "dense_top_n": 20,
                "bm25_top_n": 20,
                "rrf_k": 60,
                "max_per_source": 2,
                "rerank": True,
            }
            if eq.get("domain"):
                payload["domain"] = eq["domain"]

            s_resp = client.post("/search", json=payload)
            d_resp = client.post("/search/debug", json=payload)
            assert s_resp.status_code == 200, s_resp.text
            assert d_resp.status_code == 200, d_resp.text
            s_body = s_resp.json()
            d_body = d_resp.json()

            s_pairs = [(r["id"], r["rerank_score"]) for r in s_body["results"]]
            d_pairs = [(r["id"], r["rerank_score"]) for r in d_body["stages"]["final"]]
            assert s_pairs == d_pairs, (
                f"/search and /search/debug diverged for {eq['id']}: {s_pairs} vs {d_pairs}"
            )
            parity_checks += 1

            diag = rag_eval.compute_diagnostics(d_body, eq)
            eval_results.append(
                {"eval_query": eq, "server_response": d_body, "diagnostics": diag}
            )

        report = rag_eval.summarize_eval_run(
            eval_results,
            queries_path=str(fixture_queries),
            server="inprocess://offline-demo",
            health_payload=health_body,
        )

        # 4. Deletion + emptying reconciliation demonstration across both stores
        deleted_doc = corpus_copy / "ml" / "embedding_finetuning.md"
        emptied_doc = corpus_copy / "retrieval" / "2020-02-14-chunking-guidelines.md"
        deleted_doc.unlink()
        emptied_doc.write_text("", encoding="utf-8")

        reconcile_summary = ingest_mod.ingest(
            data_dir=str(corpus_copy),
            lancedb_path=lancedb_dir,
            tantivy_path=tantivy_dir,
            lance_table_name="alfred_rag_demo",
            embed_backend="offline",
            reset=False,
        )
        rag_server._refresh_indexes()

        post_health = client.get("/health").json()
        assert post_health["ready"] is True
        assert (
            post_health["indexes"]["lancedb"]
            == post_health["indexes"]["tantivy"]
            == build_summary["total_chunks"] - 2
        )

        check_deleted = client.post(
            "/search",
            json={"query": "MultipleNegativesRankingLoss InformationRetrievalEvaluator", "top_k": 5},
        ).json()
        returned_doc_ids = {r["doc_id"] for r in check_deleted["results"]}
        assert "ml/embedding_finetuning.md" not in returned_doc_ids
        assert "retrieval/2020-02-14-chunking-guidelines.md" not in returned_doc_ids

        # 5. Prometheus metrics contract check
        metrics_text = client.get("/metrics").text
        assert "llm_llm_" not in metrics_text
        for required_metric in (
            "llm_tokens_input_total",
            "llm_rerank_scored_docs_total",
            "llm_requests_total",
            "llm_tool_calls_total",
        ):
            assert required_metric in metrics_text, f"Missing metric {required_metric}"

        return {
            "build": build_summary,
            "eval": report["metrics"],
            "parity_checks_passed": parity_checks,
            "reconcile": {
                "files_deleted": reconcile_summary["files_deleted"],
                "chunks_deleted": reconcile_summary["chunks_deleted"],
                "remaining_lancedb": post_health["indexes"]["lancedb"],
                "remaining_tantivy": post_health["indexes"]["tantivy"],
                "generation": post_health["generation"],
            },
            "metrics_contract": "ok",
        }
    finally:
        if owns_tmp:
            shutil.rmtree(base_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline CPU demo for alfred-rag.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional persistent working directory (default: ephemeral temp dir)",
    )
    args = parser.parse_args(argv)

    result = run_demo(work_dir=args.work_dir)
    print(json.dumps(result, indent=2))
    m = result["eval"]
    if m["passed_queries"] != m["total_queries"]:
        print(
            f"Demo evaluation did not pass all queries: {m['passed_queries']}/{m['total_queries']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
