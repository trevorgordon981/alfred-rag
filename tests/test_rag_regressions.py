"""Comprehensive regression tests covering RAG-01 through RAG-07 against real
CPU LanceDB and Tantivy indexes.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
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

            # Journal must remain on disk and fail readiness checks.
            self.assertTrue(os.path.isfile(rag_common.journal_file_path(lancedb_dir)))
            rag_server.configure_offline_runtime(
                lancedb_path=lancedb_dir,
                tantivy_path=tantivy_dir,
                lance_table_name="test_rag",
            )
            client = TestClient(rag_server.app)
            unhealthy = client.get("/health")
            self.assertEqual(unhealthy.status_code, 503)
            self.assertIn("journal", unhealthy.json()["reason"].lower())

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

            res = client.post("/search", json={"query": "phoenix keyword", "top_k": 5}).json()
            self.assertEqual([r["doc_id"] for r in res["results"]], ["notes/a.md"])

    def test_generation_mismatch_fails_health_and_recovers(self) -> None:
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
            # Corrupt Tantivy generation marker to simulate store divergence.
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


if __name__ == "__main__":
    unittest.main()
