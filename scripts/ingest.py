#!/usr/bin/env python3
"""
Build or incrementally update the LanceDB (dense) and Tantivy (BM25) indexes
consumed by rag_server.py.

Usage:
    python scripts/ingest.py --data-dir ./data
    python scripts/ingest.py --data-dir ./data --reset
    python scripts/ingest.py --data-dir ./data --embed-url http://127.0.0.1:9000
    python scripts/ingest.py --data-dir ./fixtures/demo_corpus --embed-backend offline --reset
    python scripts/ingest.py --data-dir ./data --dry-run

Each document chunk is written with the schema expected by rag_server.py:
    id, doc_id, text, domain, filename, url, source, source_class, format, chunk_index, vector
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

RAG_ROOT = os.environ.get(
    "RAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)

from rag_common import (  # noqa: E402
    DEFAULT_EMBED_MODEL as COMMON_DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_REVISION as COMMON_DEFAULT_EMBED_REVISION,
    DEFAULT_VENDOR_DIRS,
    INGEST_LOCK_FILENAME,
    MANIFEST_FILENAME,
    IngestLockBusyError,
    OfflineHashEmbedder,
    adopt_or_acquire_ingest_lock,
    build_tantivy_schema,
    classify_source_path,
    clean_chunk_text,
    generation_marker_path,
    ingest_lock_path,
    inspect_lance_table_schema,
    inspect_tantivy_meta_schema,
    journal_file_path,
    lance_table_names,
    normalize_rel_path,
    read_json_optional,
    resolve_device,
    utc_now_iso,
    write_json_atomic,
)

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", os.path.join(RAG_ROOT, "data"))
DEFAULT_LANCEDB_PATH = os.environ.get(
    "LANCEDB_PATH", os.path.join(RAG_ROOT, "indexes", "lancedb")
)
DEFAULT_TANTIVY_PATH = os.environ.get(
    "TANTIVY_PATH", os.path.join(RAG_ROOT, "indexes", "tantivy")
)
DEFAULT_LANCE_TABLE = os.environ.get("LANCE_TABLE", "alfred_rag")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL_PATH", COMMON_DEFAULT_EMBED_MODEL)
DEFAULT_EMBED_REVISION = os.environ.get(
    "EMBED_MODEL_REVISION", COMMON_DEFAULT_EMBED_REVISION
)
DEFAULT_EMBED_URL = os.environ.get("EMBED_URL", "")
DEFAULT_EMBED_BACKEND = os.environ.get(
    "EMBED_BACKEND", os.environ.get("RAG_BACKEND", "hf")
)
DEFAULT_DEVICE = os.environ.get("DEVICE", "")
DEFAULT_API_KEY = os.environ.get("RAG_API_KEY", "")

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".json", ".jsonl"}
_SAFE_DOMAIN_RE = re.compile(r"[^A-Za-z0-9_-]+")
_DELETE_BATCH_SIZE = 500


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/line/word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    cleaned = clean_chunk_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    total_len = len(cleaned)
    while start < total_len:
        end = min(start + chunk_size, total_len)
        if end < total_len:
            # Prefer a paragraph, newline, or space boundary in the back half of the window.
            min_break = start + max(chunk_size // 2, 1)
            cut = cleaned.rfind("\n\n", min_break, end)
            if cut == -1:
                cut = cleaned.rfind("\n", min_break, end)
            if cut == -1:
                cut = cleaned.rfind(" ", min_break, end)
            if cut > start:
                end = cut
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= total_len:
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start
    return chunks


def _sanitize_domain(raw: str) -> str:
    sanitized = _SAFE_DOMAIN_RE.sub("_", (raw or "").strip()).strip("_")
    return sanitized or "general"


def discover_documents(data_dir: str) -> list[str]:
    """Return sorted list of supported document paths under data_dir."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if root.is_file():
        return [str(root)] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(dirpath, fname))
    return files


def load_document_records(
    filepath: str,
    data_dir: str,
    vendor_dirs: frozenset[str] = DEFAULT_VENDOR_DIRS,
) -> list[dict]:
    """Read a supported file into one or more normalized document records.

    Separates canonical corpus-relative `doc_id` (used for stable chunk identity
    and source diversification) from display/upstream `source` and `url`.
    """
    abs_file = os.path.abspath(filepath)
    abs_root = os.path.abspath(data_dir)
    try:
        rel_path = os.path.relpath(abs_file, abs_root)
    except ValueError:
        rel_path = os.path.basename(abs_file)
    norm_rel = normalize_rel_path(rel_path)

    parts = Path(norm_rel).parts
    default_domain = _sanitize_domain(parts[0] if len(parts) > 1 else "general")
    default_source_class = classify_source_path(norm_rel, vendor_dirs=vendor_dirs)
    filename = os.path.basename(abs_file)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")

    if ext in {"md", "txt", "rst"}:
        with open(abs_file, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [
            {
                "doc_id": norm_rel,
                "text": text,
                "domain": default_domain,
                "filename": filename,
                "url": "",
                "source": norm_rel,
                "source_class": default_source_class,
                "format": ext,
            }
        ]

    records: list[dict] = []
    if ext == "jsonl":
        with open(abs_file, encoding="utf-8", errors="replace") as f:
            raw_items = [json.loads(line) for line in f if line.strip()]
    elif ext == "json":
        with open(abs_file, encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if not content:
            return []
        parsed = json.loads(content)
        raw_items = parsed if isinstance(parsed, list) else [parsed]
    else:
        return []

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        # Canonical doc_id is ALWAYS scoped to normalized corpus path + record index,
        # never to a caller-supplied upstream `source` that may collide across files.
        doc_id = f"{norm_rel}#{idx}"
        rec_source = str(item.get("source") or doc_id)
        rec_source_class = classify_source_path(
            norm_rel,
            explicit_class=item.get("source_class"),
            vendor_dirs=vendor_dirs,
        )
        records.append(
            {
                "doc_id": doc_id,
                "text": text,
                "domain": _sanitize_domain(str(item.get("domain") or default_domain)),
                "filename": str(item.get("filename") or filename),
                "url": str(item.get("url") or ""),
                "source": rec_source,
                "source_class": rec_source_class,
                "format": str(item.get("format") or ext),
            }
        )
    return records


def build_chunks_for_file(
    filepath: str,
    data_dir: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    vendor_dirs: frozenset[str] = DEFAULT_VENDOR_DIRS,
) -> list[dict]:
    """Load, clean, and chunk a single file into index-ready chunk dicts."""
    chunks: list[dict] = []
    for rec in load_document_records(filepath, data_dir, vendor_dirs=vendor_dirs):
        pieces = chunk_text(rec["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, piece in enumerate(pieces):
            digest = hashlib.sha256(
                f"{rec['doc_id']}::{chunk_index}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append(
                {
                    "id": digest,
                    "doc_id": rec["doc_id"],
                    "text": piece,
                    "domain": rec["domain"],
                    "filename": rec["filename"],
                    "url": rec["url"],
                    "source": rec["source"],
                    "source_class": rec["source_class"],
                    "format": rec["format"],
                    "chunk_index": chunk_index,
                }
            )
    return chunks


def _validate_embeddings(
    vectors: Any,
    expected_count: int,
    expected_dim: int | None = None,
) -> list[list[float]]:
    """Validate embedding cardinality, finite numeric values, and uniform vector dimension."""
    if not isinstance(vectors, (list, tuple)):
        raise ValueError(
            f"Embedding backend returned invalid payload type {type(vectors).__name__}; expected list of vectors"
        )
    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count} vectors, got {len(vectors)}"
        )
    if expected_count == 0:
        return []

    validated: list[list[float]] = []
    dim: int | None = expected_dim
    for idx, raw_vec in enumerate(vectors):
        if hasattr(raw_vec, "tolist") and not isinstance(raw_vec, (list, tuple)):
            raw_vec = raw_vec.tolist()
        if not isinstance(raw_vec, (list, tuple)) or len(raw_vec) == 0:
            raise ValueError(
                f"Embedding vector at index {idx} must be a non-empty sequence of finite floats"
            )
        vec_len = len(raw_vec)
        if dim is None:
            dim = vec_len
        elif vec_len != dim:
            raise ValueError(
                f"Embedding dimension mismatch at index {idx}: expected {dim}, got {vec_len}"
            )
        clean_vec: list[float] = []
        for val in raw_vec:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"Embedding vector at index {idx} contains non-numeric value {val!r}"
                )
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(
                    f"Embedding vector at index {idx} contains non-finite value {val!r}"
                )
            clean_vec.append(fval)
        validated.append(clean_vec)
    return validated


def _embed_via_http(
    texts: list[str],
    embed_url: str,
    batch_size: int,
    api_key: str = DEFAULT_API_KEY,
) -> list[list[float]]:
    endpoint = embed_url.rstrip("/")
    if not endpoint.endswith("/embed"):
        endpoint = f"{endpoint}/embed"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    vectors: list[list[float]] = []
    batch_dim: int | None = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({"texts": batch}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise ValueError("HTTP /embed response missing 'embeddings' field")
        batch_vecs = _validate_embeddings(
            payload["embeddings"], expected_count=len(batch), expected_dim=batch_dim
        )
        if batch_vecs and batch_dim is None:
            batch_dim = len(batch_vecs[0])
        vectors.extend(batch_vecs)
    return vectors


def embed_texts(
    texts: list[str],
    model_path: str = DEFAULT_EMBED_MODEL,
    revision: str = DEFAULT_EMBED_REVISION,
    device: str = DEFAULT_DEVICE,
    batch_size: int = 32,
    embed_url: str = DEFAULT_EMBED_URL,
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    api_key: str = DEFAULT_API_KEY,
) -> list[list[float]]:
    """Compute dense embeddings via offline hash embedder, HTTP /embed, or SentenceTransformer."""
    if not texts:
        return []
    if embed_url:
        return _embed_via_http(
            texts, embed_url=embed_url, batch_size=batch_size, api_key=api_key
        )
    if (embed_backend or "").strip().lower() == "offline":
        return OfflineHashEmbedder().encode(texts, batch_size=batch_size).tolist()

    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_device(device)
    kwargs: dict = {"device": resolved_device}
    if revision and not os.path.exists(model_path):
        kwargs["revision"] = revision
    model = SentenceTransformer(model_path, **kwargs)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()


def _manifest_path(lancedb_path: str) -> str:
    return os.path.join(lancedb_path, MANIFEST_FILENAME)


def _load_manifest(lancedb_path: str) -> dict:
    """Load v2 manifest, migrating legacy flat v1 manifests transparently."""
    raw = read_json_optional(_manifest_path(lancedb_path))
    if not raw:
        return {"schema_version": 2, "generation": 0, "total_chunks": 0, "files": {}}
    if raw.get("schema_version") == 2 and isinstance(raw.get("files"), dict):
        return {
            "schema_version": 2,
            "generation": int(raw.get("generation", 0)),
            "total_chunks": int(raw.get("total_chunks", 0)),
            "files": dict(raw["files"]),
        }
    # Legacy v1 format: flat dict mapping rel_key -> entry
    files = {k: v for k, v in raw.items() if isinstance(v, dict)}
    total = sum(len(v.get("chunk_ids", [])) for v in files.values())
    return {"schema_version": 2, "generation": 0, "total_chunks": total, "files": files}


def _save_manifest(lancedb_path: str, manifest: dict) -> None:
    write_json_atomic(_manifest_path(lancedb_path), manifest)


def _file_fingerprint(filepath: str) -> dict:
    st = os.stat(filepath)
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "sha256": h.hexdigest()}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _delete_lancedb_ids(table, chunk_ids: list[str]) -> None:
    for start in range(0, len(chunk_ids), _DELETE_BATCH_SIZE):
        batch = chunk_ids[start : start + _DELETE_BATCH_SIZE]
        quoted = ", ".join("'" + cid.replace("'", "''") + "'" for cid in batch)
        table.delete(f"id IN ({quoted})")


def _tantivy_delete_by_id(writer, cid: str) -> None:
    if hasattr(writer, "delete_documents_by_term"):
        writer.delete_documents_by_term("id", cid)
    else:
        writer.delete_documents("id", cid)


def _tantivy_add_chunks(tantivy_mod, writer, chunks: list[dict]) -> None:
    for chunk in chunks:
        writer.add_document(
            tantivy_mod.Document(
                id=chunk["id"],
                doc_id=chunk["doc_id"],
                text=chunk["text"],
                domain=chunk["domain"],
                filename=chunk["filename"],
                url=chunk["url"],
                source=chunk["source"],
                source_class=chunk["source_class"],
            )
        )


def _commit_and_close_writer(writer) -> None:
    writer.commit()
    if hasattr(writer, "wait_merging_threads"):
        writer.wait_merging_threads()


def _cleanup_staging_dirs(tantivy_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(tantivy_path))
    base = os.path.basename(os.path.abspath(tantivy_path))
    if not os.path.isdir(parent):
        return
    for entry in os.listdir(parent):
        if entry.startswith(f"{base}.staging.") or entry.startswith(f"{base}.old."):
            shutil.rmtree(os.path.join(parent, entry), ignore_errors=True)


def _write_generation_markers(
    lancedb_path: str,
    tantivy_path: str,
    generation: int,
    chunk_count: int,
    lance_table_name: str,
) -> None:
    marker = {
        "generation": generation,
        "chunk_count": chunk_count,
        "lance_table": lance_table_name,
        "committed_at": utc_now_iso(),
    }
    write_json_atomic(generation_marker_path(lancedb_path), marker)
    write_json_atomic(generation_marker_path(tantivy_path), marker)


def _remove_journal(lancedb_path: str) -> None:
    jpath = journal_file_path(lancedb_path)
    try:
        os.unlink(jpath)
    except FileNotFoundError:
        pass


def _detect_legacy_schema(
    lancedb_path: str,
    tantivy_path: str,
    lance_table_name: str,
    raw_manifest: dict | None,
) -> tuple[bool, list[str], int | None]:
    """Preflight check for legacy v1 manifest/LanceDB/Tantivy schemas.

    Returns (needs_migration, reasons, existing_vector_dim). Detects missing
    v2 columns/fields such as `doc_id` and `source_class` or legacy non-raw
    Tantivy tokenizers before any incremental mutation runs.
    """
    reasons: list[str] = []
    existing_vector_dim: int | None = None

    if raw_manifest is not None and (
        raw_manifest.get("schema_version") != 2
        or not isinstance(raw_manifest.get("files"), dict)
    ):
        reasons.append("Manifest uses legacy v1 format (missing schema_version=2)")

    reasons.extend(inspect_tantivy_meta_schema(tantivy_path))

    if os.path.isdir(lancedb_path):
        try:
            import lancedb

            db = lancedb.connect(lancedb_path)
            if lance_table_name in lance_table_names(db) and hasattr(db, "open_table"):
                table = db.open_table(lance_table_name)
                lance_reasons, existing_vector_dim = inspect_lance_table_schema(table)
                reasons.extend(lance_reasons)
        except Exception:
            pass

    return bool(reasons), reasons, existing_vector_dim


def _verify_committed_counts(
    db: Any,
    lance_table_name: str,
    tantivy_index_or_path: Any,
    tantivy_mod: Any,
    tantivy_schema: Any,
    expected_count: int,
) -> None:
    """Verify both real LanceDB and Tantivy stores match expected_count before
    writing generation markers and clearing `.ingest_journal.json`.
    """
    tables = lance_table_names(db)
    if hasattr(db, "open_table"):
        if expected_count == 0:
            if lance_table_name in tables:
                tbl = db.open_table(lance_table_name)
                if hasattr(tbl, "count_rows"):
                    actual_lance = int(tbl.count_rows())
                    if actual_lance != 0:
                        raise RuntimeError(
                            f"Post-commit LanceDB count mismatch: expected 0, got {actual_lance}"
                        )
        else:
            if lance_table_name not in tables:
                raise RuntimeError(
                    f"Post-commit LanceDB table '{lance_table_name}' missing (expected {expected_count} rows)"
                )
            tbl = db.open_table(lance_table_name)
            if hasattr(tbl, "count_rows"):
                actual_lance = int(tbl.count_rows())
                if actual_lance != expected_count:
                    raise RuntimeError(
                        f"Post-commit LanceDB count mismatch: expected {expected_count}, got {actual_lance}"
                    )

    t_idx = tantivy_index_or_path
    if isinstance(t_idx, str):
        t_idx = tantivy_mod.Index(tantivy_schema, path=t_idx)
    if t_idx is not None and hasattr(t_idx, "reload") and hasattr(t_idx, "searcher"):
        t_idx.reload()
        actual_tantivy = int(t_idx.searcher().num_docs)
        if actual_tantivy != expected_count:
            raise RuntimeError(
                f"Post-commit Tantivy count mismatch: expected {expected_count}, got {actual_tantivy}"
            )


def _resolve_ingest_lock_path(lancedb_path: str, lock_path: str | None = None) -> str:
    """Resolve the canonical per-corpus lock path shared with rag_server.py."""
    if lock_path:
        return os.path.abspath(lock_path)
    env_override = os.environ.get("INGEST_LOCKFILE", "").strip()
    if env_override:
        return os.path.abspath(env_override)
    return ingest_lock_path(lancedb_path)


def ingest(
    data_dir: str = DEFAULT_DATA_DIR,
    lancedb_path: str = DEFAULT_LANCEDB_PATH,
    tantivy_path: str = DEFAULT_TANTIVY_PATH,
    lance_table_name: str = DEFAULT_LANCE_TABLE,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_revision: str = DEFAULT_EMBED_REVISION,
    embed_url: str = DEFAULT_EMBED_URL,
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    device: str = DEFAULT_DEVICE,
    api_key: str = DEFAULT_API_KEY,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    reset: bool = False,
    dry_run: bool = False,
    lock_path: str | None = None,
) -> dict:
    """Ingest documents from data_dir into LanceDB and Tantivy with dual-store parity.

    Acquires the shared per-corpus exclusive writer lock (or adopts a verified
    lock handoff when launched by rag_server.py), reconciles added, modified,
    emptied, and deleted files across both LanceDB and Tantivy, detects and
    safely migrates legacy v1 schemas via staged rebuild, validates embedding
    cardinality/dimensions before mutating either store, coordinates resets via
    validated staging, writes matching generation markers, and replays
    interrupted updates from `.ingest_journal.json`.
    """
    lease = None
    if not dry_run:
        lease = adopt_or_acquire_ingest_lock(
            _resolve_ingest_lock_path(lancedb_path, lock_path),
            mode="reset" if reset else "incremental",
            owner="cli",
        )
    try:
        return _ingest_locked(
            data_dir=data_dir,
            lancedb_path=lancedb_path,
            tantivy_path=tantivy_path,
            lance_table_name=lance_table_name,
            embed_model=embed_model,
            embed_revision=embed_revision,
            embed_url=embed_url,
            embed_backend=embed_backend,
            device=device,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            reset=reset,
            dry_run=dry_run,
        )
    finally:
        if lease is not None:
            lease.release()


def _ingest_locked(
    *,
    data_dir: str,
    lancedb_path: str,
    tantivy_path: str,
    lance_table_name: str,
    embed_model: str,
    embed_revision: str,
    embed_url: str,
    embed_backend: str,
    device: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    reset: bool,
    dry_run: bool,
) -> dict:
    files = discover_documents(data_dir)
    raw_manifest = read_json_optional(_manifest_path(lancedb_path))
    manifest_state = _load_manifest(lancedb_path)
    prev_generation = int(manifest_state.get("generation", 0))

    journal = read_json_optional(journal_file_path(lancedb_path))
    lance_gen = read_json_optional(generation_marker_path(lancedb_path))
    tantivy_gen = read_json_optional(generation_marker_path(tantivy_path))

    recovered_from_journal = False
    schema_migrated = False
    effective_reset = bool(reset)
    recovery_forced_keys: set[str] = set()
    recovery_stale_ids: list[str] = []

    needs_schema_migration, legacy_reasons, existing_vector_dim = (
        (False, [], None)
        if effective_reset
        else _detect_legacy_schema(lancedb_path, tantivy_path, lance_table_name, raw_manifest)
    )

    if not dry_run:
        if needs_schema_migration and not effective_reset:
            # Stage a clean full rebuild to migrate legacy v1 LanceDB/Tantivy schemas
            # (missing doc_id/source_class or legacy Tantivy tokenizers) safely.
            schema_migrated = True
            effective_reset = True
            existing_vector_dim = None
        if journal is not None:
            recovered_from_journal = True
            if journal.get("mode") == "reset":
                effective_reset = True
                existing_vector_dim = None
            elif not effective_reset:
                recovery_forced_keys.update(journal.get("pending_files", []))
                recovery_stale_ids.extend(journal.get("stale_ids", []))
                recovery_stale_ids.extend(journal.get("new_ids", []))
        elif (
            not effective_reset
            and (lance_gen is not None or tantivy_gen is not None)
            and (
                lance_gen is None
                or tantivy_gen is None
                or lance_gen.get("generation") != tantivy_gen.get("generation")
                or lance_gen.get("chunk_count") != tantivy_gen.get("chunk_count")
            )
        ):
            # Stores diverged without an active journal; force a clean full rebuild.
            recovered_from_journal = True
            effective_reset = True
            existing_vector_dim = None

    manifest_files: dict[str, dict] = (
        {} if effective_reset else dict(manifest_state.get("files", {}))
    )

    abs_data_dir = os.path.abspath(data_dir)
    discovered_map: dict[str, str] = {}
    for filepath in files:
        rel_key = normalize_rel_path(os.path.relpath(os.path.abspath(filepath), abs_data_dir))
        discovered_map[rel_key] = filepath

    deleted_keys: list[str] = (
        []
        if effective_reset
        else sorted(set(manifest_files.keys()) - set(discovered_map.keys()))
    )

    pending_files: list[tuple[str, str, dict]] = []
    for rel_key in sorted(discovered_map.keys()):
        filepath = discovered_map[rel_key]
        fp = _file_fingerprint(filepath)
        prev = manifest_files.get(rel_key)
        unchanged = (
            not effective_reset
            and rel_key not in recovery_forced_keys
            and prev is not None
            and (
                (prev.get("sha256") is not None and prev.get("sha256") == fp["sha256"])
                or (
                    prev.get("sha256") is None
                    and prev.get("mtime_ns") == fp["mtime_ns"]
                    and prev.get("size") == fp["size"]
                )
            )
        )
        if unchanged:
            continue
        pending_files.append((filepath, rel_key, fp))

    all_chunks: list[dict] = []
    per_file_ids: dict[str, list[str]] = {}
    stale_ids: list[str] = list(recovery_stale_ids)

    for rel_key in deleted_keys:
        prev = manifest_files.get(rel_key)
        if prev and prev.get("chunk_ids"):
            stale_ids.extend(prev["chunk_ids"])

    for filepath, rel_key, _ in pending_files:
        prev = manifest_files.get(rel_key)
        if prev and prev.get("chunk_ids"):
            stale_ids.extend(prev["chunk_ids"])
        file_chunks = build_chunks_for_file(
            filepath, data_dir=data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        per_file_ids[rel_key] = [c["id"] for c in file_chunks]
        all_chunks.extend(file_chunks)

    stale_ids = _dedupe_preserve_order(stale_ids)
    new_ids = [c["id"] for c in all_chunks]

    has_mutations = bool(
        effective_reset
        or all_chunks
        or stale_ids
        or deleted_keys
        or pending_files
        or recovered_from_journal
        or schema_migrated
    )

    # Preview total chunks after this run.
    next_files = dict(manifest_files)
    for rel_key in deleted_keys:
        next_files.pop(rel_key, None)
    for _, rel_key, fp in pending_files:
        next_files[rel_key] = {**fp, "chunk_ids": per_file_ids.get(rel_key, [])}
    total_chunks = sum(len(entry.get("chunk_ids", [])) for entry in next_files.values())
    next_generation = prev_generation + (1 if has_mutations and not dry_run else 0)

    summary = {
        "data_dir": abs_data_dir,
        "files_scanned": len(files),
        "files_ingested": len(pending_files),
        "files_deleted": len(deleted_keys),
        "chunks_written": len(all_chunks),
        "chunks_deleted": len(stale_ids),
        "total_chunks": total_chunks,
        "generation": next_generation,
        "reset": effective_reset,
        "dry_run": dry_run,
        "recovered_from_journal": recovered_from_journal,
        "schema_migrated": schema_migrated,
    }
    if legacy_reasons:
        summary["schema_migration_required"] = needs_schema_migration
        summary["schema_migration_reasons"] = legacy_reasons

    if dry_run or not has_mutations:
        return summary

    import lancedb
    import tantivy

    os.makedirs(lancedb_path, exist_ok=True)
    journal_payload: dict[str, Any] = {
        "generation": next_generation,
        "mode": "reset" if effective_reset else "incremental",
        "state": "in_progress",
        "started_at": utc_now_iso(),
        "pending_files": [rel_key for _, rel_key, _ in pending_files],
        "deleted_keys": deleted_keys,
        "stale_ids": stale_ids,
        "new_ids": new_ids,
    }
    if schema_migrated:
        journal_payload["schema_migration"] = True
        journal_payload["schema_migration_reasons"] = legacy_reasons
    write_json_atomic(journal_file_path(lancedb_path), journal_payload)

    # --- Case 1: Empty reset (zero nonempty chunks across the corpus) ---
    if effective_reset and not all_chunks:
        _cleanup_staging_dirs(tantivy_path)
        db = lancedb.connect(lancedb_path)
        if lance_table_name in lance_table_names(db):
            if hasattr(db, "drop_table"):
                db.drop_table(lance_table_name)
            else:
                table = db.open_table(lance_table_name)
                table.delete("true")

        if os.path.exists(tantivy_path):
            shutil.rmtree(tantivy_path)
        os.makedirs(tantivy_path, exist_ok=True)
        schema = build_tantivy_schema(tantivy)
        t_index = tantivy.Index(schema, path=tantivy_path)
        writer = t_index.writer()
        _commit_and_close_writer(writer)
        del writer

        _verify_committed_counts(db, lance_table_name, t_index, tantivy, schema, 0)
        _write_generation_markers(
            lancedb_path, tantivy_path, next_generation, 0, lance_table_name
        )
        _save_manifest(
            lancedb_path,
            {
                "schema_version": 2,
                "generation": next_generation,
                "total_chunks": 0,
                "files": next_files,
            },
        )
        _remove_journal(lancedb_path)
        return summary

    # Embed and strictly validate all new chunks BEFORE touching either store so
    # cardinality or vector-dimension errors preserve the journal and leave both
    # stores completely untouched and consistent.
    lance_rows: list[dict] = []
    if all_chunks:
        vectors = embed_texts(
            [c["text"] for c in all_chunks],
            model_path=embed_model,
            revision=embed_revision,
            device=device,
            batch_size=batch_size,
            embed_url=embed_url,
            embed_backend=embed_backend,
            api_key=api_key,
        )
        validated_vectors = _validate_embeddings(
            vectors,
            expected_count=len(all_chunks),
            expected_dim=None if effective_reset else existing_vector_dim,
        )
        lance_rows = [
            {**chunk, "vector": vec}
            for chunk, vec in zip(all_chunks, validated_vectors, strict=True)
        ]

    db = lancedb.connect(lancedb_path)
    existing_tables = lance_table_names(db)
    schema = build_tantivy_schema(tantivy)

    # --- Case 2: Populated reset via validated Tantivy staging + atomic swap ---
    if effective_reset:
        _cleanup_staging_dirs(tantivy_path)
        staging_dir = f"{os.path.abspath(tantivy_path)}.staging.{next_generation}"
        backup_dir = f"{os.path.abspath(tantivy_path)}.old.{next_generation}"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)

        staging_index = tantivy.Index(schema, path=staging_dir)
        writer = staging_index.writer()
        _tantivy_add_chunks(tantivy, writer, all_chunks)
        _commit_and_close_writer(writer)
        del writer

        if hasattr(staging_index, "reload") and hasattr(staging_index, "searcher"):
            staging_index.reload()
            staged_docs = int(staging_index.searcher().num_docs)
            if staged_docs != len(all_chunks):
                raise RuntimeError(
                    f"Staged Tantivy validation failed: expected {len(all_chunks)} docs, got {staged_docs}"
                )
        del staging_index

        db.create_table(lance_table_name, data=lance_rows, mode="overwrite")

        # Swap validated Tantivy generation into place.
        abs_tantivy = os.path.abspath(tantivy_path)
        os.makedirs(os.path.dirname(abs_tantivy), exist_ok=True)
        had_existing = os.path.exists(abs_tantivy)
        if had_existing:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.replace(abs_tantivy, backup_dir)
        try:
            os.replace(staging_dir, abs_tantivy)
        except Exception:
            if had_existing and os.path.exists(backup_dir) and not os.path.exists(abs_tantivy):
                os.replace(backup_dir, abs_tantivy)
            raise
        finally:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)

        _verify_committed_counts(
            db, lance_table_name, tantivy_path, tantivy, schema, total_chunks
        )
        _write_generation_markers(
            lancedb_path, tantivy_path, next_generation, total_chunks, lance_table_name
        )
        _save_manifest(
            lancedb_path,
            {
                "schema_version": 2,
                "generation": next_generation,
                "total_chunks": total_chunks,
                "files": next_files,
            },
        )
        _remove_journal(lancedb_path)
        return summary

    # --- Case 3: Incremental update (supports add, modify, empty, and delete-only) ---
    if lance_table_name not in existing_tables:
        if lance_rows:
            db.create_table(lance_table_name, data=lance_rows, mode="overwrite")
    else:
        table = db.open_table(lance_table_name)
        if stale_ids:
            _delete_lancedb_ids(table, stale_ids)
        if lance_rows:
            table.add(lance_rows)

    os.makedirs(tantivy_path, exist_ok=True)
    t_index = tantivy.Index(schema, path=tantivy_path)
    writer = t_index.writer()
    try:
        for cid in stale_ids:
            _tantivy_delete_by_id(writer, cid)
        if all_chunks:
            _tantivy_add_chunks(tantivy, writer, all_chunks)
        _commit_and_close_writer(writer)
    except BaseException:
        try:
            if hasattr(writer, "rollback"):
                writer.rollback()
        except Exception:
            pass
        try:
            if hasattr(writer, "wait_merging_threads"):
                writer.wait_merging_threads()
        except Exception:
            pass
        raise
    finally:
        try:
            del writer
        except NameError:
            pass
        import gc

        gc.collect()

    _verify_committed_counts(
        db, lance_table_name, t_index, tantivy, schema, total_chunks
    )
    _write_generation_markers(
        lancedb_path, tantivy_path, next_generation, total_chunks, lance_table_name
    )
    _save_manifest(
        lancedb_path,
        {
            "schema_version": 2,
            "generation": next_generation,
            "total_chunks": total_chunks,
            "files": next_files,
        },
    )
    _remove_journal(lancedb_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or incrementally update LanceDB and Tantivy indexes for rag_server.py."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory of source documents")
    parser.add_argument("--lancedb-path", default=DEFAULT_LANCEDB_PATH, help="LanceDB output directory")
    parser.add_argument("--tantivy-path", default=DEFAULT_TANTIVY_PATH, help="Tantivy output directory")
    parser.add_argument("--lance-table", default=DEFAULT_LANCE_TABLE, help="LanceDB table name")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model path or HF id")
    parser.add_argument("--embed-revision", default=DEFAULT_EMBED_REVISION, help="Embedding model Hub revision")
    parser.add_argument(
        "--embed-url",
        default=DEFAULT_EMBED_URL,
        help="Optional running rag_server base URL to call /embed instead of loading a local model",
    )
    parser.add_argument(
        "--embed-backend",
        default=DEFAULT_EMBED_BACKEND,
        choices=["hf", "offline"],
        help="Embedding backend: 'hf' (SentenceTransformer) or 'offline' (deterministic CPU hash embedder)",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device (auto, cuda, mps, cpu)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Optional API key for --embed-url")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Target chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Rebuild indexes from scratch instead of incremental update",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and chunk documents without writing indexes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = ingest(
            data_dir=args.data_dir,
            lancedb_path=args.lancedb_path,
            tantivy_path=args.tantivy_path,
            lance_table_name=args.lance_table,
            embed_model=args.embed_model,
            embed_revision=args.embed_revision,
            embed_url=args.embed_url,
            embed_backend=args.embed_backend,
            device=args.device,
            api_key=args.api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            reset=args.reset,
            dry_run=args.dry_run,
        )
    except IngestLockBusyError as exc:
        print(
            json.dumps(
                {
                    "status": "already_running",
                    "error": str(exc),
                    "lock_path": exc.lock_path,
                    "holder_pid": exc.holder_pid,
                    "holder_mode": exc.holder_mode,
                    "holder_owner": exc.holder_owner,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
