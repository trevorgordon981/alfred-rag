import importlib.util
import json
import os
import py_compile
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_heavy_dep_stubs() -> None:
    """Stub heavy optional ML/index dependencies only if not installed."""
    for mod_name in ("torch", "sentence_transformers", "transformers"):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                stub = types.ModuleType(mod_name)
                if mod_name == "torch":
                    stub.float16 = "float16"
                    stub.float32 = "float32"
                    stub.no_grad = mock.MagicMock
                    stub.sigmoid = lambda x: x
                    stub.cuda = types.SimpleNamespace(is_available=lambda: False)
                    stub.backends = types.SimpleNamespace(
                        mps=types.SimpleNamespace(is_available=lambda: False)
                    )
                elif mod_name == "sentence_transformers":
                    stub.SentenceTransformer = mock.MagicMock
                elif mod_name == "transformers":
                    stub.AutoTokenizer = mock.MagicMock
                    stub.AutoModelForCausalLM = mock.MagicMock
                sys.modules[mod_name] = stub

    for optional_mod in ("tantivy", "lancedb", "prometheus_client"):
        if optional_mod not in sys.modules:
            try:
                __import__(optional_mod)
            except ImportError:
                stub = types.ModuleType(optional_mod)
                if optional_mod == "tantivy":
                    stub.SchemaBuilder = mock.MagicMock
                    stub.Index = mock.MagicMock
                    stub.Document = mock.MagicMock
                elif optional_mod == "lancedb":
                    stub.connect = mock.MagicMock
                elif optional_mod == "prometheus_client":
                    class _FakeCounter:
                        def __init__(self, *args, **kwargs):
                            pass

                        def labels(self, **kwargs):
                            return self

                        def inc(self, amount=1):
                            pass

                    stub.Counter = _FakeCounter
                    stub.generate_latest = lambda: b""
                    stub.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
                sys.modules[optional_mod] = stub


_install_heavy_dep_stubs()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
import rag_server  # noqa: E402


class _FakeLanceTable:
    def __init__(self, rows: int, fail_on_read: bool = False):
        self._rows = rows
        self._fail = fail_on_read
        self.checkout_calls = 0

    def checkout_latest(self) -> None:
        self.checkout_calls += 1
        if self._fail:
            raise RuntimeError("lancedb unavailable")

    def count_rows(self) -> int:
        return self._rows


class _FakeTantivySearcher:
    def __init__(self, docs: int):
        self.num_docs = docs


class _FakeTantivyIndex:
    def __init__(self, docs: int, fail_on_read: bool = False):
        self._docs = docs
        self._fail = fail_on_read
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1
        if self._fail:
            raise RuntimeError("tantivy unavailable")

    def searcher(self) -> _FakeTantivySearcher:
        return _FakeTantivySearcher(self._docs)


def _load_ingest_module(script_path: str):
    spec = importlib.util.spec_from_file_location("alfred_rag_ingest", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestHealthAndReadiness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_lance = rag_server.lance_table
        self._orig_tantivy = rag_server.tantivy_index
        self._orig_lancedb_path = rag_server.LANCEDB_PATH
        self._orig_tantivy_path = rag_server.TANTIVY_PATH
        self._orig_api_key = rag_server.RAG_API_KEY
        self._orig_admin_key = rag_server.RAG_ADMIN_API_KEY
        rag_server.LANCEDB_PATH = os.path.join(self._tmpdir.name, "lancedb")
        rag_server.TANTIVY_PATH = os.path.join(self._tmpdir.name, "tantivy")
        rag_server.RAG_API_KEY = ""
        rag_server.RAG_ADMIN_API_KEY = ""
        self.client = TestClient(rag_server.app)

    def tearDown(self) -> None:
        rag_server.lance_table = self._orig_lance
        rag_server.tantivy_index = self._orig_tantivy
        rag_server.LANCEDB_PATH = self._orig_lancedb_path
        rag_server.TANTIVY_PATH = self._orig_tantivy_path
        rag_server.RAG_API_KEY = self._orig_api_key
        rag_server.RAG_ADMIN_API_KEY = self._orig_admin_key
        self._tmpdir.cleanup()

    def test_unloaded_indexes_return_503_unhealthy(self) -> None:
        rag_server.lance_table = None
        rag_server.tantivy_index = None

        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 503)
        body = health_resp.json()
        self.assertEqual(body["status"], "unhealthy")
        self.assertFalse(body["ready"])
        self.assertEqual(body["indexes"], {"lancedb": 0, "tantivy": 0})
        self.assertIn("LanceDB", body["reason"])
        self.assertIn("Tantivy", body["reason"])

        ready_resp = self.client.get("/ready")
        self.assertEqual(ready_resp.status_code, 503)
        ready_body = ready_resp.json()
        self.assertEqual(ready_body["status"], "not_ready")
        self.assertFalse(ready_body["ready"])

    def test_empty_indexes_return_503_and_not_ok(self) -> None:
        lance = _FakeLanceTable(rows=0)
        tantivy_idx = _FakeTantivyIndex(docs=0)
        rag_server.lance_table = lance
        rag_server.tantivy_index = tantivy_idx

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertNotEqual(body["status"], "ok")
        self.assertEqual(body["status"], "unhealthy")
        self.assertFalse(body["ready"])
        self.assertEqual(body["indexes"], {"lancedb": 0, "tantivy": 0})
        self.assertTrue(body["reason"])
        self.assertEqual(lance.checkout_calls, 1)
        self.assertEqual(tantivy_idx.reload_calls, 1)

        ready_resp = self.client.get("/ready")
        self.assertEqual(ready_resp.status_code, 503)
        self.assertEqual(ready_resp.json()["status"], "not_ready")

    def test_single_empty_index_returns_503(self) -> None:
        rag_server.lance_table = _FakeLanceTable(rows=15)
        rag_server.tantivy_index = _FakeTantivyIndex(docs=0)
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["status"], "unhealthy")
        self.assertIn("Tantivy", resp.json()["reason"])

        rag_server.lance_table = _FakeLanceTable(rows=0)
        rag_server.tantivy_index = _FakeTantivyIndex(docs=15)
        resp2 = self.client.get("/health")
        self.assertEqual(resp2.status_code, 503)
        self.assertEqual(resp2.json()["status"], "unhealthy")
        self.assertIn("LanceDB", resp2.json()["reason"])

    def test_populated_indexes_return_200_ok(self) -> None:
        rag_server.lance_table = _FakeLanceTable(rows=128)
        rag_server.tantivy_index = _FakeTantivyIndex(docs=128)

        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        body = health_resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["ready"])
        self.assertIsNone(body["reason"])
        self.assertEqual(body["indexes"], {"lancedb": 128, "tantivy": 128})

        ready_resp = self.client.get("/ready")
        self.assertEqual(ready_resp.status_code, 200)
        ready_body = ready_resp.json()
        self.assertEqual(ready_body["status"], "ok")
        self.assertTrue(ready_body["ready"])
        self.assertIsNone(ready_body["reason"])

    def test_index_read_error_returns_503(self) -> None:
        rag_server.lance_table = _FakeLanceTable(rows=10, fail_on_read=True)
        rag_server.tantivy_index = _FakeTantivyIndex(docs=10)
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["status"], "unhealthy")
        self.assertIn("lancedb unavailable", resp.json()["reason"])


class TestIngestBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_script = rag_server.INGEST_SCRIPT
        self._orig_lockfile = rag_server.INGEST_LOCKFILE
        self._orig_log = rag_server.INGEST_LOG
        self._orig_api_key = rag_server.RAG_API_KEY
        self._orig_admin_key = rag_server.RAG_ADMIN_API_KEY
        rag_server.RAG_API_KEY = ""
        rag_server.RAG_ADMIN_API_KEY = ""
        self.client = TestClient(rag_server.app)

    def tearDown(self) -> None:
        rag_server.INGEST_SCRIPT = self._orig_script
        rag_server.INGEST_LOCKFILE = self._orig_lockfile
        rag_server.INGEST_LOG = self._orig_log
        rag_server.RAG_API_KEY = self._orig_api_key
        rag_server.RAG_ADMIN_API_KEY = self._orig_admin_key

    def test_default_ingest_script_exists_in_repo_and_compiles(self) -> None:
        script_path = Path(rag_server._resolve_ingest_script()).resolve()
        self.assertTrue(script_path.is_file(), f"Missing ingest script: {script_path}")
        self.assertTrue(
            str(script_path).startswith(str(REPO_ROOT)),
            f"Ingest script must live inside repo ({REPO_ROOT}), got {script_path}",
        )
        py_compile.compile(str(script_path), doraise=True)

        mod = _load_ingest_module(str(script_path))
        self.assertTrue(callable(mod.main))
        self.assertTrue(callable(mod.ingest))
        self.assertTrue(callable(mod.clean_chunk_text))
        self.assertTrue(callable(mod.build_chunks_for_file))

    def test_ingest_endpoint_fails_explicitly_if_script_missing(self) -> None:
        rag_server.INGEST_SCRIPT = str(REPO_ROOT / "scripts" / "does_not_exist.py")
        with mock.patch.object(rag_server.subprocess, "Popen") as mock_popen:
            resp = self.client.post("/ingest", json={"incremental": True, "reset": False})
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "error")
        self.assertIn("not found", body["error"].lower())
        mock_popen.assert_not_called()

    def test_ingest_endpoint_spawns_existing_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rag_server.INGEST_LOCKFILE = os.path.join(tmpdir, "ingest.lock")
            rag_server.INGEST_LOG = os.path.join(tmpdir, "ingest.log")
            fake_proc = mock.MagicMock(pid=42424)
            with mock.patch.object(rag_server.subprocess, "Popen", return_value=fake_proc) as mock_popen:
                resp = self.client.post("/ingest", json={"incremental": False, "reset": True})

            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["status"], "started")
            self.assertEqual(body["pid"], 42424)
            self.assertEqual(body["mode"], "reset")
            called_args = mock_popen.call_args.args[0]
            self.assertEqual(called_args[1], rag_server._resolve_ingest_script())
            self.assertIn("--reset", called_args)

    def test_ingest_script_dry_run_and_incremental_pipeline(self) -> None:
        script_path = rag_server._resolve_ingest_script()
        ingest_mod = _load_ingest_module(script_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "finance").mkdir(parents=True)
            (data_dir / "notes").mkdir(parents=True)

            md_file = data_dir / "finance" / "options.md"
            md_file.write_text(
                "LOG IN\nImplied volatility reflects expected future movement in options pricing.\n",
                encoding="utf-8",
            )
            jsonl_file = data_dir / "notes" / "entries.jsonl"
            jsonl_file.write_text(
                json.dumps(
                    {
                        "text": "Reciprocal rank fusion merges dense and sparse rankings.",
                        "url": "https://example.com/rrf",
                        "domain": "retrieval",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rc = ingest_mod.main(["--data-dir", str(data_dir), "--dry-run"])
            self.assertEqual(rc, 0)

            chunks = ingest_mod.build_chunks_for_file(str(md_file), str(data_dir))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["domain"], "finance")
            self.assertEqual(chunks[0]["filename"], "options.md")
            self.assertEqual(chunks[0]["doc_id"], "finance/options.md")
            self.assertEqual(chunks[0]["source_class"], "curated")
            self.assertNotIn("LOG IN", chunks[0]["text"])
            for field in (
                "id",
                "doc_id",
                "text",
                "domain",
                "filename",
                "url",
                "source",
                "source_class",
                "format",
                "chunk_index",
            ):
                self.assertIn(field, chunks[0])

            created_rows = []
            added_docs = []

            class _MockDB:
                def __init__(self):
                    self._tables = set()

                def table_names(self):
                    return list(self._tables)

                def create_table(self, name, data, mode="overwrite"):
                    self._tables.add(name)
                    created_rows.extend(data)

            class _MockWriter:
                def delete_documents(self, field, value):
                    pass

                def add_document(self, doc):
                    added_docs.append(doc)

                def commit(self):
                    pass

            class _MockIndex:
                def __init__(self, schema, path):
                    pass

                def writer(self):
                    return _MockWriter()

            mock_db = _MockDB()
            lancedb_dir = str(Path(tmpdir) / "indexes" / "lancedb")
            tantivy_dir = str(Path(tmpdir) / "indexes" / "tantivy")

            with (
                mock.patch.object(
                    ingest_mod,
                    "embed_texts",
                    side_effect=lambda texts, **kw: [[0.1, 0.2]] * len(texts),
                ),
                mock.patch.object(sys.modules["lancedb"], "connect", return_value=mock_db),
                mock.patch.object(sys.modules["tantivy"], "Index", _MockIndex),
            ):
                first = ingest_mod.ingest(
                    data_dir=str(data_dir),
                    lancedb_path=lancedb_dir,
                    tantivy_path=tantivy_dir,
                    reset=True,
                )
                second = ingest_mod.ingest(
                    data_dir=str(data_dir),
                    lancedb_path=lancedb_dir,
                    tantivy_path=tantivy_dir,
                    reset=False,
                )

            self.assertEqual(first["files_ingested"], 2)
            self.assertEqual(first["chunks_written"], 2)
            self.assertEqual(len(created_rows), 2)
            self.assertEqual(len(added_docs), 2)
            self.assertEqual(second["files_scanned"], 2)
            self.assertEqual(second["files_ingested"], 0)
            self.assertEqual(second["chunks_written"], 0)


if __name__ == "__main__":
    unittest.main()
