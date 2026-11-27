"""Comprehensive regression tests covering RAG-01 through RAG-08 against real
CPU LanceDB and Tantivy indexes.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lancedb  # noqa: E402
import tantivy  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import rag_common  # noqa: E402
import rag_eval  # noqa: E402
import rag_server  # noqa: E402
from scripts import ingest as ingest_mod  # noqa: E402
from scripts import offline_demo  # noqa: E402


class _ServerStateGuard:
    """Save and restore rag_server module globals around tests."""

    _ATTRS = (
        "embed_model",
        "reranker_tokenizer",
        "reranker_model",
        "lance_table",
        "tantivy_index",
        "loaded_generation",
        "active_device",
        "active_models",
        "LANCEDB_PATH",
        "TANTIVY_PATH",
        "LANCE_TABLE",
        "RAG_API_KEY",
        "RAG_ADMIN_API_KEY",
        "INGEST_SCRIPT",
        "INGEST_PYTHON",
        "INGEST_LOCKFILE",
        "INGEST_LOG",
        "LABEL_EMBED",
        "LABEL_RERANK",
        "MAX_REQUEST_BODY_BYTES",
    )

    def __enter__(self):
        self._saved = {k: getattr(rag_server, k) for k in self._ATTRS}
        rag_server.RAG_API_KEY = ""
        rag_server.RAG_ADMIN_API_KEY = ""
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            setattr(rag_server, k, v)
        rag_server._cached_query_embedding.cache_clear()


class TestRAG01ReconciliationAndCrashRecovery(unittest.TestCase):
    """RAG-01: deleted/emptied doc sync, empty reset parity, journal crash recovery."""

    def test_incremental_delete_and_empty_purge_both_stores(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            doc_keep = data_dir / "notes" / "keep.md"
            doc_delete = data_dir / "notes" / "delete_me.md"
            doc_empty = data_dir / "notes" / "blank_me.md"

            doc_keep.write_text(
                "Alpha keeper document covering vector retrieval invariants.",
                encoding="utf-8",
            )
            doc_delete.write_text(
                "Beta transient zeppelin document scheduled for deletion.",
                encoding="utf-8",
            )
            doc_empty.write_text(
                "Gamma ephemeral narwhal document scheduled to be emptied.",
                encoding="utf-8",
            )

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")

            first = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=True,
            )
            self.assertEqual(first["total_chunks"], 3)

            # Delete one source file and blank another (zero new chunks produced).
            doc_delete.unlink()
            doc_empty.write_text("   \n\n  ", encoding="utf-8")

            second = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=False,
            )
            self.assertEqual(second["files_deleted"], 1)
            self.assertEqual(second["chunks_written"], 0)
            self.assertEqual(second["chunks_deleted"], 2)
            self.assertEqual(second["total_chunks"], 1)

            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
            )
            client = TestClient(rag_server.app)

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["indexes"], {"lancedb": 1, "tantivy": 1})

            # Neither deleted nor emptied doc may appear in dense, BM25, or /search.
            for q in ("zeppelin deletion", "narwhal emptied"):
                res = client.post("/search/debug", json={"query": q, "top_k": 5}).json()
                all_doc_ids = {
                    c["doc_id"]
                    for stage in ("dense", "bm25", "final")
                    for c in res["stages"][stage]
                }
                self.assertNotIn("notes/delete_me.md", all_doc_ids)
                self.assertNotIn("notes/blank_me.md", all_doc_ids)

    def test_empty_reset_clears_both_lancedb_and_tantivy(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            for idx in range(3):
                (data_dir / "notes" / f"doc_{idx}.md").write_text(
                    f"Document {idx} with searchable content.", encoding="utf-8"
                )

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")

            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=True,
            )

            # Empty all source documents, then run reset.
            for p in (data_dir / "notes").glob("*.md"):
                p.write_text("", encoding="utf-8")

            summary = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=True,
            )
            self.assertEqual(summary["chunks_written"], 0)
            self.assertEqual(summary["total_chunks"], 0)

            db = lancedb.connect(lancedb_dir)
            self.assertNotIn("test_rag", rag_common.lance_table_names(db))

            schema = rag_common.build_tantivy_schema(tantivy)
            t_idx = tantivy.Index(schema, path=tantivy_dir)
            t_idx.reload()
            self.assertEqual(t_idx.searcher().num_docs, 0)

            lance_gen = rag_common.read_json_optional(rag_common.generation_marker_path(lancedb_dir))
            tantivy_gen = rag_common.read_json_optional(rag_common.generation_marker_path(tantivy_dir))
            self.assertIsNotNone(lance_gen)
            self.assertEqual(lance_gen["generation"], tantivy_gen["generation"])
            self.assertEqual(lance_gen["chunk_count"], 0)

            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
            )
            client = TestClient(rag_server.app)
            self.assertEqual(client.get("/health").status_code, 503)
            self.assertEqual(client.get("/ready").status_code, 503)

    def test_interrupted_incremental_update_detected_and_replayed(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            doc_a = data_dir / "notes" / "a.md"
            doc_b = data_dir / "notes" / "b.md"
            doc_a.write_text("Original content of document A.", encoding="utf-8")
            doc_b.write_text("Original content of document B.", encoding="utf-8")

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")

            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=True,
            )

            # Modify A and delete B, but simulate a crash after LanceDB commits
            # and before Tantivy commits.
            doc_a.write_text("Updated content of document A with phoenix keyword.", encoding="utf-8")
            doc_b.unlink()

            with mock.patch.object(
                ingest_mod,
                "_commit_and_close_writer",
                side_effect=RuntimeError("simulated tantivy crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated tantivy crash"):
                    ingest_mod.ingest(
                        data_dir=str(data_dir),
                        lancedb_path=lancedb_dir,
                        tantivy_path=tantivy_dir,
                        lance_table_name="test_rag",
                        embed_backend="offline",
                        reset=False,
                    )

            # Journal must remain on disk and fail readiness AND search endpoints closed (503)
            # so mixed new-dense + old-sparse generations are NEVER served.
            self.assertTrue(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
            )
            rag_server._refresh_indexes()  # must refuse to refresh while journal is pending
            client = TestClient(rag_server.app)
            unhealthy = client.get("/health")
            self.assertEqual(unhealthy.status_code, 503)
            self.assertIn("journal", unhealthy.json()["reason"].lower())
            self.assertEqual(client.get("/ready").status_code, 503)

            for endpoint in ("/search", "/search/debug"):
                blocked = client.post(endpoint, json={"query": "phoenix keyword", "top_k": 5})
                self.assertEqual(
                    blocked.status_code,
                    503,
                    f"{endpoint} must fail closed with 503 while journal is pending",
                )
                self.assertIn("journal", blocked.json()["detail"].lower())

            # Re-running ingest replays the interrupted transaction cleanly with zero duplicates.
            recovered = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=False,
            )
            self.assertTrue(recovered["recovered_from_journal"])
            self.assertFalse(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))

            rag_server._refresh_indexes()
            healthy = client.get("/health")
            self.assertEqual(healthy.status_code, 200)
            self.assertEqual(healthy.json()["indexes"], {"lancedb": 1, "tantivy": 1})

            s_resp = client.post("/search", json={"query": "phoenix keyword", "top_k": 5})
            d_resp = client.post("/search/debug", json={"query": "phoenix keyword", "top_k": 5})
            self.assertEqual(s_resp.status_code, 200)
            self.assertEqual(d_resp.status_code, 200)
            self.assertEqual([r["doc_id"] for r in s_resp.json()["results"]], ["notes/a.md"])
            self.assertEqual([r["doc_id"] for r in d_resp.json()["results"]], ["notes/a.md"])

    def test_generation_and_chunk_count_mismatch_fail_search_and_recover(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            (data_dir / "notes" / "one.md").write_text("Single test doc.", encoding="utf-8")

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")

            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=True,
            )
            # Case A: Corrupt Tantivy generation marker to simulate generation divergence.
            rag_common.write_json_atomic(
                rag_common.generation_marker_path(tantivy_dir),
                {"generation": 999, "chunk_count": 1, "lance_table": "test_rag"},
            )

            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
            )
            client = TestClient(rag_server.app)
            resp = client.get("/ready")
            self.assertEqual(resp.status_code, 503)
            self.assertIn("mismatch", resp.json()["reason"].lower())
            for endpoint in ("/search", "/search/debug"):
                blocked = client.post(endpoint, json={"query": "single test doc", "top_k": 5})
                self.assertEqual(blocked.status_code, 503)
                self.assertIn("mismatch", blocked.json()["detail"].lower())

            # Case B: Same generation number but disagreeing chunk_count markers.
            rag_common.write_json_atomic(
                rag_common.generation_marker_path(tantivy_dir),
                {"generation": 1, "chunk_count": 7, "lance_table": "test_rag"},
            )
            self.assertEqual(client.get("/ready").status_code, 503)
            for endpoint in ("/search", "/search/debug"):
                blocked = client.post(endpoint, json={"query": "single test doc", "top_k": 5})
                self.assertEqual(blocked.status_code, 503)
                self.assertIn("chunk_count", blocked.json()["detail"].lower())

            recovered = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
                embed_backend="offline",
                reset=False,
            )
            self.assertTrue(recovered["recovered_from_journal"])
            rag_server._refresh_indexes()
            self.assertEqual(client.get("/ready").status_code, 200)
            self.assertEqual(
                client.post("/search", json={"query": "single test doc", "top_k": 5}).status_code,
                200,
            )
            self.assertEqual(
                client.post("/search/debug", json={"query": "single test doc", "top_k": 5}).status_code,
                200,
            )

    def test_short_and_mismatched_embedding_batches_rejected_preserving_journal(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            (data_dir / "notes" / "first.md").write_text(
                "First document for embedding cardinality validation.", encoding="utf-8"
            )
            (data_dir / "notes" / "second.md").write_text(
                "Second document for embedding cardinality validation.", encoding="utf-8"
            )

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")

            # Short batch: 1 vector returned for 2 chunks must fail BEFORE touching either store,
            # leaving .ingest_journal.json on disk and zero partial dense/sparse rows.
            with mock.patch.object(
                ingest_mod,
                "embed_texts",
                return_value=[[0.1, 0.2, 0.3, 0.4]],
            ):
                with self.assertRaisesRegex(ValueError, "Embedding count mismatch"):
                    ingest_mod.ingest(
                        data_dir=str(data_dir),
                        lancedb_path=lancedb_dir,
                        tantivy_path=tantivy_dir,
                        lance_table_name="card_test",
                        embed_backend="offline",
                        reset=True,
                    )

            self.assertTrue(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))
            db = lancedb.connect(lancedb_dir)
            self.assertNotIn("card_test", rag_common.lance_table_names(db))
            self.assertFalse(os.path.exists(tantivy_dir))

            # Non-finite vector elements are also rejected before mutating stores.
            with mock.patch.object(
                ingest_mod,
                "embed_texts",
                return_value=[[0.1, float("nan")], [0.2, 0.3]],
            ):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    ingest_mod.ingest(
                        data_dir=str(data_dir),
                        lancedb_path=lancedb_dir,
                        tantivy_path=tantivy_dir,
                        lance_table_name="card_test",
                        embed_backend="offline",
                        reset=True,
                    )
            self.assertTrue(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))

            # Recovery with a valid offline embedder reconciles cleanly and clears the journal.
            recovered = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="card_test",
                embed_backend="offline",
                reset=False,
            )
            self.assertTrue(recovered["recovered_from_journal"])
            self.assertEqual(recovered["total_chunks"], 2)
            self.assertFalse(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))

            # Dimension mismatch on incremental append must fail before mutating either store,
            # preserve the journal, and keep existing dense/sparse counts equal (2 vs 2).
            (data_dir / "notes" / "third.md").write_text(
                "Third document whose embedding has wrong vector dimension.", encoding="utf-8"
            )
            with mock.patch.object(
                ingest_mod,
                "embed_texts",
                return_value=[[0.1, 0.2, 0.3]],  # 3d instead of existing 256d table
            ):
                with self.assertRaisesRegex(ValueError, "dimension mismatch"):
                    ingest_mod.ingest(
                        data_dir=str(data_dir),
                        lancedb_path=lancedb_dir,
                        tantivy_path=tantivy_dir,
                        lance_table_name="card_test",
                        embed_backend="offline",
                        reset=False,
                    )

            self.assertTrue(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))
            self.assertEqual(db.open_table("card_test").count_rows(), 2)
            t_idx = tantivy.Index(rag_common.build_tantivy_schema(tantivy), path=tantivy_dir)
            t_idx.reload()
            self.assertEqual(t_idx.searcher().num_docs, 2)


class TestRAG02UnifiedSearchAndEval(unittest.TestCase):
    """RAG-02: /search and /search/debug share one pipeline and rank identically."""

    def test_search_and_search_debug_identical_across_modes_and_policies(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            fixture_dir = str(REPO_ROOT / "fixtures" / "demo_corpus")

            ingest_mod.ingest(
                data_dir=fixture_dir,
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="parity_test",
                embed_backend="offline",
                reset=True,
            )
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="parity_test",
            )
            client = TestClient(rag_server.app)

            queries = [
                {
                    "query": "notes security reconciliation readiness",
                    "top_k": 5,
                    "max_per_source": 1,
                    "domain_boost": {"operations": 1.25},
                    "rerank": True,
                },
                {
                    "query": "notes security reconciliation readiness",
                    "top_k": 5,
                    "max_per_source": 1,
                    "domain_boost": {"operations": 1.25},
                    "rerank": False,
                },
                {
                    "query": "chunking guidelines source diversification",
                    "top_k": 4,
                    "max_per_source": 1,
                    "rerank": True,
                },
            ]

            for payload in queries:
                s_body = client.post("/search", json=payload).json()
                d_body = client.post("/search/debug", json=payload).json()
                s_ids = [(r["id"], r["rerank_score"]) for r in s_body["results"]]
                d_ids = [(r["id"], r["rerank_score"]) for r in d_body["stages"]["final"]]
                d_top = [(r["id"], r["rerank_score"]) for r in d_body["results"]]
                self.assertEqual(s_ids, d_ids)
                self.assertEqual(s_ids, d_top)
                self.assertEqual(s_body["counts"], d_body["counts"])

    def test_eval_diagnostics_exact_doc_span_and_ood(self) -> None:
        sr = {
            "stages": {
                "dense": [],
                "bm25": [],
                "rrf_merged": [],
                "reranked": [
                    {
                        "id": "c1",
                        "doc_id": "retrieval/hybrid_rrf_notes.md",
                        "domain": "retrieval",
                        "text": "RRF uses score(d) = sum(1 / (k + rank_i)) with k=60.",
                        "rerank_score": 0.91,
                        "rank": 1,
                    }
                ],
                "final": [
                    {
                        "id": "c1",
                        "doc_id": "retrieval/hybrid_rrf_notes.md",
                        "domain": "retrieval",
                        "text": "RRF uses score(d) = sum(1 / (k + rank_i)) with k=60.",
                        "rerank_score": 0.91,
                        "rank": 1,
                    }
                ],
            },
            "timings": {"total": 0.01},
            "counts": {"returned": 1},
        }
        eq = {
            "id": "q1",
            "query": "RRF formula",
            "expected_doc_ids": ["retrieval/hybrid_rrf_notes.md"],
            "expected_spans": ["score(d) = sum(1 / (k + rank_i))"],
            "expected_keywords": ["k=60"],
        }
        diag = rag_eval.compute_diagnostics(sr, eq)
        self.assertTrue(diag["passed"])
        self.assertTrue(diag["doc_match"]["hit_at_1"])
        self.assertTrue(diag["span_match"]["all_matched"])

        # OOD negative control fails if top score exceeds min_score, passes if below.
        ood_eq = {
            "id": "q-ood",
            "query": "unrelated",
            "expect_no_results": True,
            "min_score": 0.5,
        }
        self.assertFalse(rag_eval.compute_diagnostics(sr, ood_eq)["passed"])
        sr_low = json.loads(json.dumps(sr))
        sr_low["stages"]["final"][0]["rerank_score"] = 0.12
        self.assertTrue(rag_eval.compute_diagnostics(sr_low, ood_eq)["passed"])


class TestRAG03SecurityBindAuthAndLimits(unittest.TestCase):
    """RAG-03: loopback bind default, API/admin auth, and request limits."""

    def test_bind_defaults_loopback_and_blocks_unsafe_external(self) -> None:
        self.assertEqual(
            rag_common.resolve_bind_config(host=None, port=None, allow_external_bind=False, api_key=""),
            ("127.0.0.1", 9000),
        )
        # Non-loopback without opt-in flag fails.
        with self.assertRaisesRegex(ValueError, "ALLOW_EXTERNAL_BIND"):
            rag_common.resolve_bind_config(
                host="0.0.0.0", port=9000, allow_external_bind=False, api_key="secret"
            )
        # Non-loopback with opt-in flag but without API key fails.
        with self.assertRaisesRegex(ValueError, "RAG_API_KEY"):
            rag_common.resolve_bind_config(
                host="0.0.0.0", port=9000, allow_external_bind=True, api_key=""
            )
        # Non-loopback with BOTH opt-in and API key succeeds.
        self.assertEqual(
            rag_common.resolve_bind_config(
                host="0.0.0.0", port=9123, allow_external_bind=True, api_key="secret-token"
            ),
            ("0.0.0.0", 9123),
        )

    def test_api_key_and_admin_reset_authorization(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            data_dir = Path(tmpdir) / "data"
            (data_dir / "ops").mkdir(parents=True)
            (data_dir / "ops" / "a.md").write_text("Auth verification passage.", encoding="utf-8")

            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="auth_test",
                embed_backend="offline",
                reset=True,
            )
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="auth_test",
            )
            rag_server.RAG_API_KEY = "reader-key"
            rag_server.RAG_ADMIN_API_KEY = "admin-key"
            rag_server.INGEST_LOCKFILE = os.path.join(tmpdir, "ingest.lock")
            rag_server.INGEST_LOG = os.path.join(tmpdir, "ingest.log")
            client = TestClient(rag_server.app)

            # Probes stay accessible without auth.
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)

            # Unauthenticated corpus/admin routes fail with 401.
            self.assertEqual(client.post("/search", json={"query": "auth"}).status_code, 401)
            self.assertEqual(client.post("/search/debug", json={"query": "auth"}).status_code, 401)
            self.assertEqual(client.post("/embed", json={"texts": ["hi"]}).status_code, 401)
            self.assertEqual(
                client.post("/rerank", json={"query": "a", "documents": ["b"]}).status_code, 401
            )
            self.assertEqual(client.get("/metrics").status_code, 401)
            self.assertEqual(client.post("/admin/ingest/reset").status_code, 401)

            # Standard API key allows reads and incremental ingest, but NOT reset (403).
            reader_headers = {"Authorization": "Bearer reader-key"}
            self.assertEqual(
                client.post("/search", json={"query": "auth"}, headers=reader_headers).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/ingest", json={"incremental": True, "reset": True}, headers=reader_headers
                ).status_code,
                403,
            )
            self.assertEqual(
                client.post("/admin/ingest/reset", headers=reader_headers).status_code, 403
            )

            # Admin key succeeds on /admin/ingest/reset.
            admin_headers = {"X-API-Key": "admin-key"}
            fake_proc = mock.MagicMock(pid=51515)
            with mock.patch.object(rag_server.subprocess, "Popen", return_value=fake_proc):
                resp = client.post("/admin/ingest/reset", headers=admin_headers)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["mode"], "reset")

    def test_request_body_and_parameter_bounds(self) -> None:
        with _ServerStateGuard():
            rag_server.MAX_REQUEST_BODY_BYTES = 256
            client = TestClient(rag_server.app)

            oversized = client.post(
                "/search",
                content=json.dumps({"query": "x" * 512}),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(oversized.status_code, 413)

            rag_server.MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
            self.assertEqual(
                client.post("/search", json={"query": "ok", "top_k": 500}).status_code, 422
            )
            self.assertEqual(client.post("/embed", json={"texts": []}).status_code, 422)


class TestRAG04IdentityAndDiversification(unittest.TestCase):
    """RAG-04: collision-free doc_id, same-basename diversification, deduplicated RRF."""

    def test_shared_upstream_source_records_distinct_and_delete_independently(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "team_a").mkdir(parents=True)
            (data_dir / "team_b").mkdir(parents=True)

            file_a = data_dir / "team_a" / "feed.json"
            file_b = data_dir / "team_b" / "feed.json"
            file_a.write_text(
                json.dumps(
                    {
                        "text": "Cobalt turbine telemetry report from team alpha.",
                        "source": "upstream://shared-feed",
                    }
                ),
                encoding="utf-8",
            )
            file_b.write_text(
                json.dumps(
                    {
                        "text": "Obsidian capacitor telemetry report from team beta.",
                        "source": "upstream://shared-feed",
                    }
                ),
                encoding="utf-8",
            )

            chunks_a = ingest_mod.build_chunks_for_file(str(file_a), str(data_dir))
            chunks_b = ingest_mod.build_chunks_for_file(str(file_b), str(data_dir))
            self.assertNotEqual(chunks_a[0]["id"], chunks_b[0]["id"])
            self.assertNotEqual(chunks_a[0]["doc_id"], chunks_b[0]["doc_id"])
            self.assertEqual(chunks_a[0]["source"], "upstream://shared-feed")
            self.assertEqual(chunks_b[0]["source"], "upstream://shared-feed")

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="shared_src",
                embed_backend="offline",
                reset=True,
            )

            # Deleting file_a must keep file_b intact in both stores.
            file_a.unlink()
            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="shared_src",
                embed_backend="offline",
                reset=False,
            )

            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="shared_src",
            )
            client = TestClient(rag_server.app)
            self.assertEqual(client.get("/health").json()["indexes"], {"lancedb": 1, "tantivy": 1})
            res = client.post("/search", json={"query": "obsidian capacitor", "top_k": 5}).json()
            self.assertEqual([r["doc_id"] for r in res["results"]], ["team_b/feed.json#0"])

    def test_same_basename_files_not_collapsed_by_max_per_source(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "alpha").mkdir(parents=True)
            (data_dir / "beta").mkdir(parents=True)

            (data_dir / "alpha" / "notes.md").write_text(
                "Shared quartz signal calibration checklist for alpha module.", encoding="utf-8"
            )
            (data_dir / "beta" / "notes.md").write_text(
                "Shared quartz signal calibration checklist for beta module.", encoding="utf-8"
            )

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="basename_test",
                embed_backend="offline",
                reset=True,
            )
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="basename_test",
            )
            client = TestClient(rag_server.app)

            for rerank_flag in (True, False):
                res = client.post(
                    "/search",
                    json={
                        "query": "quartz signal calibration checklist",
                        "top_k": 5,
                        "max_per_source": 1,
                        "rerank": rerank_flag,
                    },
                ).json()
                doc_ids = {r["doc_id"] for r in res["results"]}
                self.assertEqual(doc_ids, {"alpha/notes.md", "beta/notes.md"})

    def test_rrf_merge_deduplicates_within_single_leg(self) -> None:
        merged, scores = rag_server.rrf_merge_with_scores(
            dense_ids=["dup", "dup", "dup", "other"],
            bm25_ids=["other"],
            k=60,
        )
        # 'dup' only appears in dense (rank 0 -> 1/61), while 'other' appears in
        # both dense (rank 1 -> 1/62) and bm25 (rank 0 -> 1/61), so 'other' MUST win!
        self.assertEqual(merged[0], "other")
        self.assertAlmostEqual(scores["dup"], 1.0 / 61.0, places=9)
        self.assertAlmostEqual(scores["other"], 1.0 / 62.0 + 1.0 / 61.0, places=9)

    def test_legacy_v1_schema_migrates_cleanly_on_incremental_edit_without_reset(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            (data_dir / "docs").mkdir(parents=True)
            doc_a = data_dir / "notes" / "alpha.md"
            doc_b = data_dir / "docs" / "vendor_guide.md"
            doc_a.write_text("Legacy baseline alpha passage.", encoding="utf-8")
            doc_b.write_text("Legacy baseline vendor reference guide.", encoding="utf-8")

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            os.makedirs(lancedb_dir, exist_ok=True)
            os.makedirs(tantivy_dir, exist_ok=True)

            # Construct an authentic legacy v1 index (missing doc_id & source_class,
            # default Tantivy tokenizers on id/domain, flat v1 manifest, no .generation.json).
            embedder = rag_common.OfflineHashEmbedder()
            vecs = embedder.encode(
                ["Legacy baseline alpha passage.", "Legacy baseline vendor reference guide."]
            ).tolist()
            db = lancedb.connect(lancedb_dir)
            db.create_table(
                "legacy_rag",
                data=[
                    {
                        "id": "legacy_id_a",
                        "text": "Legacy baseline alpha passage.",
                        "domain": "notes",
                        "filename": "alpha.md",
                        "url": "",
                        "source": "notes/alpha.md",
                        "format": "md",
                        "chunk_index": 0,
                        "vector": vecs[0],
                    },
                    {
                        "id": "legacy_id_b",
                        "text": "Legacy baseline vendor reference guide.",
                        "domain": "docs",
                        "filename": "vendor_guide.md",
                        "url": "",
                        "source": "docs/vendor_guide.md",
                        "format": "md",
                        "chunk_index": 0,
                        "vector": vecs[1],
                    },
                ],
                mode="overwrite",
            )

            sb = tantivy.SchemaBuilder()
            sb.add_text_field("id", stored=True)
            sb.add_text_field("text", stored=True, tokenizer_name="en_stem")
            sb.add_text_field("domain", stored=True)
            sb.add_text_field("filename", stored=True)
            sb.add_text_field("url", stored=True)
            sb.add_text_field("source", stored=True)
            legacy_t_idx = tantivy.Index(sb.build(), path=tantivy_dir)
            writer = legacy_t_idx.writer()
            writer.add_document(
                tantivy.Document(
                    id="legacy_id_a",
                    text="Legacy baseline alpha passage.",
                    domain="notes",
                    filename="alpha.md",
                    url="",
                    source="notes/alpha.md",
                )
            )
            writer.add_document(
                tantivy.Document(
                    id="legacy_id_b",
                    text="Legacy baseline vendor reference guide.",
                    domain="docs",
                    filename="vendor_guide.md",
                    url="",
                    source="docs/vendor_guide.md",
                )
            )
            writer.commit()
            writer.wait_merging_threads()
            del writer, legacy_t_idx

            # Write legacy v1 flat manifest.
            fp_a = ingest_mod._file_fingerprint(str(doc_a))
            fp_b = ingest_mod._file_fingerprint(str(doc_b))
            rag_common.write_json_atomic(
                os.path.join(lancedb_dir, rag_common.MANIFEST_FILENAME),
                {
                    "notes/alpha.md": {**fp_a, "chunk_ids": ["legacy_id_a"]},
                    "docs/vendor_guide.md": {**fp_b, "chunk_ids": ["legacy_id_b"]},
                },
            )

            # Edit one file on top of the legacy index.
            doc_a.write_text(
                "Updated alpha passage with migrated zenith telemetry.", encoding="utf-8"
            )

            # Dry run detects required schema migration without mutating stores.
            dry = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="legacy_rag",
                embed_backend="offline",
                reset=False,
                dry_run=True,
            )
            self.assertTrue(dry["schema_migration_required"])
            self.assertFalse(dry["schema_migrated"])
            self.assertIn("doc_id", " ".join(dry["schema_migration_reasons"]))

            # Incremental ingest (reset=False) safely migrates both stores without crashing.
            migrated = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="legacy_rag",
                embed_backend="offline",
                reset=False,
            )
            self.assertTrue(migrated["schema_migrated"])
            self.assertEqual(migrated["total_chunks"], 2)
            self.assertFalse(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))

            # Verify migrated LanceDB and Tantivy schemas are now fully v2-compliant.
            migrated_table = lancedb.connect(lancedb_dir).open_table("legacy_rag")
            lance_issues, _ = rag_common.inspect_lance_table_schema(migrated_table)
            self.assertEqual(lance_issues, [])
            self.assertEqual(rag_common.inspect_tantivy_meta_schema(tantivy_dir), [])

            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="legacy_rag",
            )
            client = TestClient(rag_server.app)
            self.assertEqual(client.get("/ready").status_code, 200)
            res = client.post(
                "/search", json={"query": "migrated zenith telemetry", "top_k": 5}
            )
            self.assertEqual(res.status_code, 200)
            top = res.json()["results"][0]
            self.assertEqual(top["doc_id"], "notes/alpha.md")
            self.assertEqual(top["source_class"], "curated")

            # Subsequent incremental edit runs normally without triggering another migration.
            doc_a.write_text(
                "Second edit on migrated schema with solstice signal.", encoding="utf-8"
            )
            followup = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="legacy_rag",
                embed_backend="offline",
                reset=False,
            )
            self.assertFalse(followup["schema_migrated"])
            self.assertFalse(followup["reset"])
            self.assertEqual(followup["files_ingested"], 1)
            self.assertEqual(followup["total_chunks"], 2)


class TestRAG05VendorClassAndRecency(unittest.TestCase):
    """RAG-05: explicit source_class persisted at ingest + real fast-path recency tie-break."""

    def test_vendor_source_class_persisted_for_public_relative_path(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "docs").mkdir(parents=True)
            (data_dir / "guides").mkdir(parents=True)

            vendor_file = data_dir / "docs" / "reference.md"
            curated_file = data_dir / "guides" / "playbook.md"
            shared_text = "Hybrid retrieval index tuning guidance for production workloads."
            vendor_file.write_text(shared_text, encoding="utf-8")
            curated_file.write_text(shared_text, encoding="utf-8")

            v_chunks = ingest_mod.build_chunks_for_file(str(vendor_file), str(data_dir))
            c_chunks = ingest_mod.build_chunks_for_file(str(curated_file), str(data_dir))
            self.assertEqual(v_chunks[0]["source"], "docs/reference.md")
            self.assertEqual(v_chunks[0]["source_class"], "vendor")
            self.assertEqual(c_chunks[0]["source_class"], "curated")
            self.assertTrue(rag_server._is_vendor_doc(v_chunks[0]))
            self.assertFalse(rag_server._is_vendor_doc(c_chunks[0]))

            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="vendor_test",
                embed_backend="offline",
                reset=True,
            )
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="vendor_test",
            )
            client = TestClient(rag_server.app)

            for rerank_flag in (True, False):
                res = client.post(
                    "/search",
                    json={"query": shared_text, "top_k": 2, "rerank": rerank_flag},
                ).json()
                self.assertEqual(res["results"][0]["doc_id"], "guides/playbook.md")
                self.assertEqual(res["results"][1]["doc_id"], "docs/reference.md")

    def test_recency_breaks_real_fast_path_rrf_tie(self) -> None:
        with _ServerStateGuard():
            rag_server.embed_model = rag_common.OfflineHashEmbedder()
            rag_server.reranker_tokenizer = None
            rag_server.reranker_model = rag_common.OfflineLexicalReranker()

            old_doc = {
                "id": "id_old",
                "doc_id": "notes/2020-01-10-policy.md",
                "text": "Policy text.",
                "source": "notes/2020-01-10-policy.md",
                "source_class": "curated",
                "filename": "2020-01-10-policy.md",
                "domain": "notes",
                "url": "",
            }
            new_doc = {
                "id": "id_new",
                "doc_id": "notes/2026-06-01-policy.md",
                "text": "Policy text.",
                "source": "notes/2026-06-01-policy.md",
                "source_class": "curated",
                "filename": "2026-06-01-policy.md",
                "domain": "notes",
                "url": "",
            }
            # Real RRF tie: old_doc is rank-1 in dense only (1/61); new_doc is rank-1 in BM25 only (1/61).
            # Notice 'id_new' even sorts AFTER 'id_old' alphabetically, so ONLY recency puts new_doc first!
            with (
                mock.patch.object(rag_server, "_dense_search", return_value=[dict(old_doc)]),
                mock.patch.object(rag_server, "_bm25_search", return_value=[dict(new_doc)]),
            ):
                fast = rag_server._execute_search(
                    rag_server.SearchRequest(query="policy", rerank=False, top_k=2)
                )
                self.assertEqual([r["id"] for r in fast["results"]], ["id_new", "id_old"])

                reranked = rag_server._execute_search(
                    rag_server.SearchRequest(query="policy", rerank=True, top_k=2)
                )
                self.assertEqual([r["id"] for r in reranked["results"]], ["id_new", "id_old"])
                self.assertGreater(
                    reranked["results"][0]["rerank_score"],
                    reranked["results"][1]["rerank_score"],
                )

            ref = dt.date(2026, 6, 1)
            self.assertGreater(
                rag_server._recency_multiplier(new_doc, today=ref),
                rag_server._recency_multiplier(old_doc, today=ref),
            )


class TestRAG06DeviceAndPortConfiguration(unittest.TestCase):
    """RAG-06: shared DEVICE resolution + PORT/HOST configuration control the server."""

    def test_resolve_device_and_server_cli_args(self) -> None:
        self.assertEqual(rag_common.resolve_device("cpu"), "cpu")
        self.assertEqual(rag_common.resolve_device("mps"), "mps")
        self.assertEqual(rag_common.resolve_device("cuda"), "cuda")
        with self.assertRaises(ValueError):
            rag_common.resolve_device("tpu")

        args = rag_server.parse_args(
            ["--host", "127.0.0.1", "--port", "9444", "--device", "cpu", "--backend", "offline"]
        )
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 9444)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.backend, "offline")

    def test_load_models_uses_resolved_device_not_hardcoded_cuda(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            fake_torch = types.SimpleNamespace(
                float16="float16",
                float32="float32",
                cuda=types.SimpleNamespace(is_available=lambda: False),
                backends=types.SimpleNamespace(
                    mps=types.SimpleNamespace(is_available=lambda: False)
                ),
            )
            fake_st = mock.MagicMock()
            fake_tok = mock.MagicMock()
            fake_tok.pad_token_id = 0
            fake_tok.convert_tokens_to_ids.side_effect = [1, 2]
            fake_model = mock.MagicMock()
            fake_model.to.return_value = fake_model

            fake_auto_tok = mock.MagicMock()
            fake_auto_tok.from_pretrained.return_value = fake_tok
            fake_auto_model = mock.MagicMock()
            fake_auto_model.from_pretrained.return_value = fake_model

            env_patch = {
                "RAG_BACKEND": "hf",
                "DEVICE": "cpu",
                "LANCEDB_PATH": os.path.join(tmpdir, "lancedb"),
                "TANTIVY_PATH": os.path.join(tmpdir, "tantivy"),
            }
            with (
                mock.patch.dict(os.environ, env_patch, clear=False),
                mock.patch.object(rag_server, "torch", fake_torch),
                mock.patch.object(rag_server, "SentenceTransformer", fake_st),
                mock.patch.object(rag_server, "AutoTokenizer", fake_auto_tok),
                mock.patch.object(rag_server, "AutoModelForCausalLM", fake_auto_model),
                mock.patch.object(rag_server, "_open_indexes"),
                mock.patch.object(rag_server, "_start_refresh_timer"),
            ):
                rag_server.load_models()

            self.assertEqual(rag_server.active_device, "cpu")
            self.assertEqual(fake_st.call_args.kwargs["device"], "cpu")
            model_kwargs = fake_auto_model.from_pretrained.call_args.kwargs
            self.assertEqual(model_kwargs["dtype"], "float32")
            self.assertNotIn("device_map", model_kwargs)
            fake_model.to.assert_called_once_with("cpu")


class TestRAG07PrometheusMetricsContract(unittest.TestCase):
    """RAG-07: single-prefix metric namespace and scored-doc rerank accounting."""

    def test_metrics_namespace_and_rerank_scored_docs_counter(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            (data_dir / "notes" / "m1.md").write_text("Prometheus metric contract test one.", encoding="utf-8")
            (data_dir / "notes" / "m2.md").write_text("Prometheus metric contract test two.", encoding="utf-8")

            ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="metrics_test",
                embed_backend="offline",
                reset=True,
            )
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="metrics_test",
            )
            client = TestClient(rag_server.app)

            before = rag_server.RAG_RERANK_SCORED_DOCS.labels(**rag_server.LABEL_RERANK)._value.get()
            resp = client.post(
                "/rerank",
                json={"query": "prometheus", "documents": ["doc a", "doc b", "doc c"], "top_k": 2},
            )
            self.assertEqual(resp.status_code, 200)
            after = rag_server.RAG_RERANK_SCORED_DOCS.labels(**rag_server.LABEL_RERANK)._value.get()
            self.assertEqual(after - before, 3.0)

            scrape = client.get("/metrics").text
            self.assertNotIn("llm_llm_", scrape)
            self.assertNotIn("llm_tokens_output_total", scrape)
            for metric_name in (
                "llm_tokens_input_total",
                "llm_rerank_scored_docs_total",
                "llm_requests_total",
                "llm_tool_calls_total",
            ):
                self.assertIn(metric_name, scrape)


class TestRAG08ExclusiveIngestAdmission(unittest.TestCase):
    """RAG-08: atomic per-corpus exclusive ingest admission across HTTP and CLI writers."""

    def test_concurrent_http_ingest_and_reset_admit_exactly_one_at_popen_pause(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            lancedb_dir = str(Path(tmpdir) / "lancedb")
            rag_server.LANCEDB_PATH = lancedb_dir
            rag_server.TANTIVY_PATH = str(Path(tmpdir) / "tantivy")
            rag_server.INGEST_LOCKFILE = ""
            rag_server.INGEST_LOG = os.path.join(tmpdir, "ingest.log")
            client = TestClient(rag_server.app)

            entered_popen = threading.Event()
            release_popen = threading.Event()
            popen_calls: list[tuple[list[str], dict]] = []
            calls_lock = threading.Lock()

            def controlled_popen(args, **kwargs):
                with calls_lock:
                    popen_calls.append((list(args), dict(kwargs)))
                    pid = 100000 + len(popen_calls)
                entered_popen.set()
                release_popen.wait(timeout=5)
                return types.SimpleNamespace(pid=pid)

            with mock.patch.object(rag_server.subprocess, "Popen", side_effect=controlled_popen):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    first_fut = pool.submit(
                        client.post, "/ingest", json={"incremental": True, "reset": False}
                    )
                    self.assertTrue(
                        entered_popen.wait(timeout=3),
                        "First /ingest request did not enter Popen in time",
                    )

                    # While request 1 is paused inside Popen, a concurrent reset MUST
                    # immediately fail closed with HTTP 409 without touching Popen.
                    second_resp = client.post("/admin/ingest/reset")
                    self.assertEqual(second_resp.status_code, 409)
                    second_body = second_resp.json()
                    self.assertEqual(second_body["status"], "already_running")
                    self.assertEqual(second_body["mode"], "incremental")
                    self.assertEqual(len(popen_calls), 1)

                    # A direct CLI ingest() against the same corpus MUST also fail closed.
                    data_dir = Path(tmpdir) / "data"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    (data_dir / "doc.md").write_text("Blocked CLI doc.", encoding="utf-8")
                    with self.assertRaises(rag_common.IngestLockBusyError):
                        ingest_mod.ingest(
                            data_dir=str(data_dir),
                            lancedb_path=lancedb_dir,
                            tantivy_path=rag_server.TANTIVY_PATH,
                            embed_backend="offline",
                            reset=False,
                        )

                    release_popen.set()
                    first_resp = first_fut.result(timeout=5)

            self.assertEqual(first_resp.status_code, 200)
            first_body = first_resp.json()
            self.assertEqual(first_body["status"], "started")
            self.assertEqual(first_body["mode"], "incremental")
            self.assertEqual(first_body["pid"], 100001)
            self.assertEqual(len(popen_calls), 1)

            # Verify verified handoff fd + token were passed to Popen and lease records the single PID.
            spawn_kwargs = popen_calls[0][1]
            self.assertIn("pass_fds", spawn_kwargs)
            self.assertEqual(len(spawn_kwargs["pass_fds"]), 1)
            spawn_env = spawn_kwargs["env"]
            self.assertEqual(
                spawn_env[rag_common.INGEST_LOCK_ENV_FD],
                str(spawn_kwargs["pass_fds"][0]),
            )
            self.assertTrue(spawn_env[rag_common.INGEST_LOCK_ENV_TOKEN])

            lease_file = rag_common.ingest_lease_path(rag_common.ingest_lock_path(lancedb_dir))
            lease_data = rag_common.read_json_optional(lease_file)
            self.assertIsNotNone(lease_data)
            self.assertEqual(lease_data["pid"], 100001)
            self.assertEqual(lease_data["mode"], "incremental")

    def test_exact_rag08_barrier_counterexample_never_runs_second_popen_and_releases_on_error(
        self,
    ) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            rag_server.LANCEDB_PATH = str(Path(tmpdir) / "lancedb")
            rag_server.TANTIVY_PATH = str(Path(tmpdir) / "tantivy")
            rag_server.INGEST_LOCKFILE = ""
            rag_server.INGEST_LOG = os.path.join(tmpdir, "ingest.log")
            client = TestClient(rag_server.app, raise_server_exceptions=False)

            barrier = threading.Barrier(2, timeout=0.4)
            initiated: list[int] = []
            mutex = threading.Lock()

            def barrier_popen(args, **kwargs):
                with mutex:
                    initiated.append(len(initiated) + 1)
                    pid = 100000 + initiated[-1]
                # Under the old buggy code, BOTH threads reached this barrier and both started.
                # Under the fixed exclusive lock, ONLY ONE thread enters Popen while the second
                # gets HTTP 409 immediately, so this barrier MUST time out with BrokenBarrierError.
                barrier.wait()
                return types.SimpleNamespace(pid=pid)

            with mock.patch.object(rag_server.subprocess, "Popen", side_effect=barrier_popen):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futs = [
                        pool.submit(
                            client.post, "/ingest", json={"incremental": True, "reset": False}
                        ),
                        pool.submit(client.post, "/admin/ingest/reset"),
                    ]
                    responses = [f.result(timeout=5) for f in futs]

            self.assertEqual(
                len(initiated),
                1,
                f"Expected exactly one Popen attempt, saw {initiated}",
            )
            statuses = sorted(r.status_code for r in responses)
            self.assertEqual(statuses, [409, 500])
            conflict = next(r for r in responses if r.status_code == 409)
            self.assertEqual(conflict.json()["status"], "already_running")

            # After the failed Popen, the lock must be cleanly released so a subsequent call starts.
            with mock.patch.object(
                rag_server.subprocess,
                "Popen",
                return_value=types.SimpleNamespace(pid=200001),
            ):
                followup = client.post("/ingest", json={"incremental": True, "reset": False})
            self.assertEqual(followup.status_code, 200)
            self.assertEqual(followup.json()["pid"], 200001)

    def test_cross_process_lock_blocks_http_and_cli_and_cleans_up_on_release(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "notes").mkdir(parents=True)
            (data_dir / "notes" / "sample.md").write_text(
                "Cross-process lock verification passage.", encoding="utf-8"
            )
            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            rag_server.LANCEDB_PATH = lancedb_dir
            rag_server.TANTIVY_PATH = tantivy_dir
            rag_server.INGEST_LOCKFILE = ""
            rag_server.INGEST_LOG = os.path.join(tmpdir, "ingest.log")
            client = TestClient(rag_server.app)

            ready_flag = Path(tmpdir) / "holder.ready"
            stop_flag = Path(tmpdir) / "holder.stop"
            holder_code = (
                "import os, sys, time\n"
                f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
                "import rag_common\n"
                f"lp = rag_common.ingest_lock_path({lancedb_dir!r})\n"
                "with rag_common.acquire_ingest_lock(lp, mode='reset', owner='cli', pid=os.getpid()):\n"
                f"    open({str(ready_flag)!r}, 'w').write(str(os.getpid()))\n"
                f"    while not os.path.exists({str(stop_flag)!r}):\n"
                "        time.sleep(0.02)\n"
            )
            holder_proc = subprocess.Popen(
                [sys.executable, "-c", holder_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5.0
                while not ready_flag.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready_flag.exists(), "Holder subprocess failed to acquire lock")
                holder_pid = int(ready_flag.read_text(encoding="utf-8").strip())

                # HTTP /ingest and /admin/ingest/reset must fail closed with 409 and 0 Popen calls.
                with mock.patch.object(rag_server.subprocess, "Popen") as mock_popen:
                    r1 = client.post("/ingest", json={"incremental": True, "reset": False})
                    r2 = client.post("/admin/ingest/reset")
                mock_popen.assert_not_called()
                self.assertEqual(r1.status_code, 409)
                self.assertEqual(r2.status_code, 409)
                self.assertEqual(r1.json()["status"], "already_running")
                self.assertEqual(r1.json()["pid"], holder_pid)
                self.assertEqual(r2.json()["pid"], holder_pid)

                # Direct Python ingest() must fail closed with IngestLockBusyError.
                with self.assertRaises(rag_common.IngestLockBusyError) as ctx:
                    ingest_mod.ingest(
                        data_dir=str(data_dir),
                        lancedb_path=lancedb_dir,
                        tantivy_path=tantivy_dir,
                        embed_backend="offline",
                        reset=False,
                    )
                self.assertEqual(ctx.exception.holder_pid, holder_pid)

                # Direct CLI main() must exit non-zero (2) without mutating indexes.
                stderr_buf = io.StringIO()
                with mock.patch.object(sys, "stderr", stderr_buf):
                    rc = ingest_mod.main(
                        [
                            "--data-dir",
                            str(data_dir),
                            "--lancedb-path",
                            lancedb_dir,
                            "--tantivy-path",
                            tantivy_dir,
                            "--embed-backend",
                            "offline",
                        ]
                    )
                self.assertEqual(rc, 2)
                self.assertIn("already_running", stderr_buf.getvalue())

                # Spoofed handoff env vars MUST NOT bypass the active cross-process lock.
                with mock.patch.dict(
                    os.environ,
                    {
                        rag_common.INGEST_LOCK_ENV_PATH: rag_common.ingest_lock_path(lancedb_dir),
                        rag_common.INGEST_LOCK_ENV_FD: "0",
                        rag_common.INGEST_LOCK_ENV_TOKEN: "forged-token",
                    },
                    clear=False,
                ):
                    with self.assertRaises(rag_common.IngestLockBusyError):
                        ingest_mod.ingest(
                            data_dir=str(data_dir),
                            lancedb_path=lancedb_dir,
                            tantivy_path=tantivy_dir,
                            embed_backend="offline",
                            reset=False,
                        )

                # A separate corpus has its own per-corpus lock and is NOT blocked.
                other_lance = str(Path(tmpdir) / "other_lancedb")
                other_tantivy = str(Path(tmpdir) / "other_tantivy")
                other_summary = ingest_mod.ingest(
                    data_dir=str(data_dir),
                    lancedb_path=other_lance,
                    tantivy_path=other_tantivy,
                    lance_table_name="other_rag",
                    embed_backend="offline",
                    reset=True,
                )
                self.assertEqual(other_summary["total_chunks"], 1)
            finally:
                stop_flag.write_text("stop", encoding="utf-8")
                holder_proc.communicate(timeout=5)

            # Once the cross-process holder exits, CLI ingest acquires the lock and succeeds cleanly.
            summary = ingest_mod.ingest(
                data_dir=str(data_dir),
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="lock_test",
                embed_backend="offline",
                reset=True,
            )
            self.assertEqual(summary["total_chunks"], 1)
            self.assertIsNone(rag_common.inspect_ingest_lock(rag_common.ingest_lock_path(lancedb_dir)))

    def test_real_http_spawn_handoff_keeps_lock_held_until_child_exits(self) -> None:
        with _ServerStateGuard(), tempfile.TemporaryDirectory() as tmpdir:
            lancedb_dir = str(Path(tmpdir) / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "tantivy")
            worker_script = Path(tmpdir) / "fake_ingest_worker.py"
            adopted_flag = Path(tmpdir) / "worker.adopted"
            release_flag = Path(tmpdir) / "worker.release"

            worker_script.write_text(
                "import os, sys, time\n"
                f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
                "import rag_common\n"
                "lp = rag_common.ingest_lock_path(os.environ['LANCEDB_PATH'])\n"
                "with rag_common.adopt_or_acquire_ingest_lock(lp, mode='incremental', owner='cli') as lease:\n"
                "    assert lease.metadata['owner'] == 'http-worker', lease.metadata\n"
                f"    open({str(adopted_flag)!r}, 'w').write(str(os.getpid()))\n"
                f"    while not os.path.exists({str(release_flag)!r}):\n"
                "        time.sleep(0.02)\n",
                encoding="utf-8",
            )

            rag_server.LANCEDB_PATH = lancedb_dir
            rag_server.TANTIVY_PATH = tantivy_dir
            rag_server.INGEST_LOCKFILE = ""
            rag_server.INGEST_SCRIPT = str(worker_script)
            rag_server.INGEST_PYTHON = sys.executable
            rag_server.INGEST_LOG = os.path.join(tmpdir, "ingest.log")
            client = TestClient(rag_server.app)

            try:
                resp = client.post("/ingest", json={"incremental": True, "reset": False})
                self.assertEqual(resp.status_code, 200)
                child_pid = resp.json()["pid"]

                deadline = time.monotonic() + 5.0
                while not adopted_flag.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(
                    adopted_flag.exists(),
                    f"Child worker did not adopt lock; log={Path(rag_server.INGEST_LOG).read_text(encoding='utf-8', errors='replace')}",
                )
                self.assertEqual(int(adopted_flag.read_text(encoding="utf-8").strip()), child_pid)

                # While the real child holds the handed-off lock, another HTTP request gets 409!
                conflict = client.post("/admin/ingest/reset")
                self.assertEqual(conflict.status_code, 409)
                self.assertEqual(conflict.json()["status"], "already_running")
                self.assertEqual(conflict.json()["pid"], child_pid)
                self.assertEqual(conflict.json()["owner"], "http-worker")

                # And a direct CLI ingest() against the same corpus fails closed!
                data_dir = Path(tmpdir) / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "a.md").write_text("Blocked while child runs.", encoding="utf-8")
                with self.assertRaises(rag_common.IngestLockBusyError) as ctx:
                    ingest_mod.ingest(
                        data_dir=str(data_dir),
                        lancedb_path=lancedb_dir,
                        tantivy_path=tantivy_dir,
                        embed_backend="offline",
                        reset=False,
                    )
                self.assertEqual(ctx.exception.holder_pid, child_pid)
            finally:
                release_flag.write_text("done", encoding="utf-8")

            # Wait for child to exit and release its flock.
            lock_path = rag_common.ingest_lock_path(lancedb_dir)
            deadline = time.monotonic() + 5.0
            while rag_common.inspect_ingest_lock(lock_path) is not None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNone(rag_common.inspect_ingest_lock(lock_path))


class TestOfflineDemoFixture(unittest.TestCase):
    """End-to-end verification of the public synthetic fixture corpus and offline demo."""

    def test_offline_demo_passes_all_checks(self) -> None:
        with _ServerStateGuard():
            result = offline_demo.run_demo()
        self.assertEqual(result["eval"]["passed_queries"], result["eval"]["total_queries"])
        self.assertEqual(result["eval"]["doc_hit_at_1"], result["eval"]["doc_labeled_queries"])
        self.assertEqual(result["eval"]["span_all_matched"], result["eval"]["span_labeled_queries"])
        self.assertEqual(result["eval"]["ood_rejected"], result["eval"]["ood_queries"])
        self.assertEqual(result["parity_checks_passed"], result["eval"]["total_queries"])
        self.assertEqual(result["reconcile"]["remaining_lancedb"], result["reconcile"]["remaining_tantivy"])
        self.assertEqual(result["metrics_contract"], "ok")

    def test_quickstart_dependency_manifests_include_required_packages(self) -> None:
        required = {
            "fastapi",
            "uvicorn",
            "pydantic",
            "lancedb",
            "tantivy",
            "prometheus-client",
            "requests",
            "httpx",
        }
        for filename in ("requirements-offline.txt", "requirements.txt"):
            text = (REPO_ROOT / filename).read_text(encoding="utf-8").lower()
            for pkg in required:
                self.assertIn(pkg, text, f"{filename} is missing required dependency '{pkg}'")
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("requirements-offline.txt", readme_text)
        for pkg in ("requests", "prometheus-client", "httpx"):
            self.assertIn(pkg, readme_text, f"README.md missing dependency '{pkg}'")


if __name__ == "__main__":
    unittest.main()
