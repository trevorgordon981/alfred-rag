#!/usr/bin/env python3
"""
Build or incrementally update the LanceDB (dense) and Tantivy (BM25) indexes
consumed by rag_server.py.

Usage:
    python scripts/ingest.py --data-dir ./data
    python scripts/ingest.py --data-dir ./data --reset
    python scripts/ingest.py --data-dir ./data --embed-url http://localhost:9000
    python scripts/ingest.py --data-dir ./data --dry-run

Each document chunk is written with the schema expected by rag_server.py:
    id, text, domain, filename, url, source, format, chunk_index, vector
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

RAG_ROOT = os.environ.get(
    "RAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", os.path.join(RAG_ROOT, "data"))
DEFAULT_LANCEDB_PATH = os.environ.get(
    "LANCEDB_PATH", os.path.join(RAG_ROOT, "indexes", "lancedb")
)
DEFAULT_TANTIVY_PATH = os.environ.get(
    "TANTIVY_PATH", os.path.join(RAG_ROOT, "indexes", "tantivy")
)
DEFAULT_LANCE_TABLE = os.environ.get("LANCE_TABLE", "alfred_rag")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-8B")
DEFAULT_EMBED_REVISION = os.environ.get(
    "EMBED_MODEL_REVISION", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
)
DEFAULT_EMBED_URL = os.environ.get("EMBED_URL", "")
DEFAULT_DEVICE = os.environ.get("DEVICE", "")

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".json", ".jsonl"}
_SAFE_DOMAIN_RE = re.compile(r"[^A-Za-z0-9_-]+")
_BOILERPLATE_PATTERNS = [
    r"(?i)^.*?Rennlist - Porsche Discussion Forums\s*",
    r"(?i)^LOG\s*IN\s*",
    r"(?i)^REGISTER\s*",
    r"(?i)^View First.*?\n",
    r"(?i)^View Public Profile.*?\n",
    r"(?i)^Find More Posts by.*?\n",
    r"(?i)^Share\s*Share Options\s*",
    r"(?i)^Quote:\s*Originally Posted by\s*\S+\s*",
    r"#\s*\d+\s*$",
]


def clean_chunk_text(text: str) -> str:
    """Strip common forum navigation boilerplate and normalize whitespace."""
    cleaned = text
    for pat in _BOILERPLATE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


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


def load_document_records(filepath: str, data_dir: str) -> list[dict]:
    """Read a supported file into one or more normalized document records."""
    abs_file = os.path.abspath(filepath)
    abs_root = os.path.abspath(data_dir)
    try:
        rel_path = os.path.relpath(abs_file, abs_root)
    except ValueError:
        rel_path = os.path.basename(abs_file)

    parts = Path(rel_path).parts
    default_domain = _sanitize_domain(parts[0] if len(parts) > 1 else "general")
    filename = os.path.basename(abs_file)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")

    if ext in {"md", "txt", "rst"}:
        with open(abs_file, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [
            {
                "text": text,
                "domain": default_domain,
                "filename": filename,
                "url": "",
                "source": rel_path.replace(os.sep, "/"),
                "format": ext,
            }
        ]

    records: list[dict] = []
    if ext == "jsonl":
        with open(abs_file, encoding="utf-8", errors="replace") as f:
            raw_items = [json.loads(line) for line in f if line.strip()]
    elif ext == "json":
        with open(abs_file, encoding="utf-8", errors="replace") as f:
            parsed = json.load(f)
        raw_items = parsed if isinstance(parsed, list) else [parsed]
    else:
        return []

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        rec_source = item.get("source") or f"{rel_path.replace(os.sep, '/')}#{idx}"
        records.append(
            {
                "text": text,
                "domain": _sanitize_domain(str(item.get("domain") or default_domain)),
                "filename": str(item.get("filename") or filename),
                "url": str(item.get("url") or ""),
                "source": str(rec_source),
                "format": str(item.get("format") or ext),
            }
        )
    return records


def build_chunks_for_file(
    filepath: str,
    data_dir: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Load, clean, and chunk a single file into index-ready chunk dicts."""
    chunks: list[dict] = []
    for rec in load_document_records(filepath, data_dir):
        pieces = chunk_text(rec["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, piece in enumerate(pieces):
            digest = hashlib.sha256(
                f"{rec['source']}::{chunk_index}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append(
                {
                    "id": digest,
                    "text": piece,
                    "domain": rec["domain"],
                    "filename": rec["filename"],
                    "url": rec["url"],
                    "source": rec["source"],
                    "format": rec["format"],
                    "chunk_index": chunk_index,
                }
            )
    return chunks


def build_tantivy_schema(tantivy_mod):
    """Build the Tantivy schema matching rag_server.py."""
    sb = tantivy_mod.SchemaBuilder()
    sb.add_text_field("id", stored=True)
    sb.add_text_field("text", stored=True, tokenizer_name="en_stem")
    sb.add_text_field("domain", stored=True)
    sb.add_text_field("filename", stored=True)
    sb.add_text_field("url", stored=True)
    sb.add_text_field("source", stored=True)
    return sb.build()


def _embed_via_http(texts: list[str], embed_url: str, batch_size: int) -> list[list[float]]:
    endpoint = embed_url.rstrip("/")
    if not endpoint.endswith("/embed"):
        endpoint = f"{endpoint}/embed"
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({"texts": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        vectors.extend(payload["embeddings"])
    return vectors


def embed_texts(
    texts: list[str],
    model_path: str = DEFAULT_EMBED_MODEL,
    revision: str = DEFAULT_EMBED_REVISION,
    device: str = DEFAULT_DEVICE,
    batch_size: int = 32,
    embed_url: str = DEFAULT_EMBED_URL,
) -> list[list[float]]:
    """Compute dense embeddings via HTTP /embed endpoint or SentenceTransformer."""
    if not texts:
        return []
    if embed_url:
        return _embed_via_http(texts, embed_url=embed_url, batch_size=batch_size)

    import torch
    from sentence_transformers import SentenceTransformer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"device": resolved_device}
    if revision and not os.path.exists(model_path):
        kwargs["revision"] = revision
    model = SentenceTransformer(model_path, **kwargs)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()


def _manifest_path(lancedb_path: str) -> str:
    return os.path.join(lancedb_path, ".ingest_manifest.json")


def _load_manifest(lancedb_path: str) -> dict:
    path = _manifest_path(lancedb_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_manifest(lancedb_path: str, manifest: dict) -> None:
    os.makedirs(lancedb_path, exist_ok=True)
    path = _manifest_path(lancedb_path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _file_fingerprint(filepath: str) -> dict:
    st = os.stat(filepath)
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def ingest(
    data_dir: str = DEFAULT_DATA_DIR,
    lancedb_path: str = DEFAULT_LANCEDB_PATH,
    tantivy_path: str = DEFAULT_TANTIVY_PATH,
    lance_table_name: str = DEFAULT_LANCE_TABLE,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_revision: str = DEFAULT_EMBED_REVISION,
    embed_url: str = DEFAULT_EMBED_URL,
    device: str = DEFAULT_DEVICE,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    reset: bool = False,
    dry_run: bool = False,
) -> dict:
    """Ingest documents from data_dir into LanceDB and Tantivy."""
    files = discover_documents(data_dir)
    manifest = {} if reset else _load_manifest(lancedb_path)

    pending_files: list[tuple[str, str, dict]] = []
    for filepath in files:
        rel_key = os.path.relpath(os.path.abspath(filepath), os.path.abspath(data_dir)).replace(
            os.sep, "/"
        )
        fp = _file_fingerprint(filepath)
        prev = manifest.get(rel_key)
        if (
            not reset
            and prev
            and prev.get("mtime_ns") == fp["mtime_ns"]
            and prev.get("size") == fp["size"]
        ):
            continue
        pending_files.append((filepath, rel_key, fp))

    all_chunks: list[dict] = []
    per_file_ids: dict[str, list[str]] = {}
    stale_ids: list[str] = []
    for filepath, rel_key, _ in pending_files:
        prev = manifest.get(rel_key)
        if prev and prev.get("chunk_ids"):
            stale_ids.extend(prev["chunk_ids"])
        file_chunks = build_chunks_for_file(
            filepath, data_dir=data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        per_file_ids[rel_key] = [c["id"] for c in file_chunks]
        all_chunks.extend(file_chunks)

    summary = {
        "data_dir": os.path.abspath(data_dir),
        "files_scanned": len(files),
        "files_ingested": len(pending_files),
        "chunks_written": len(all_chunks),
        "reset": reset,
        "dry_run": dry_run,
    }
    if dry_run or not all_chunks:
        if reset and not dry_run:
            # Even when reset finds zero chunks, clear old state explicitly.
            if os.path.exists(tantivy_path):
                shutil.rmtree(tantivy_path)
            _save_manifest(lancedb_path, {})
        return summary

    import lancedb
    import tantivy

    vectors = embed_texts(
        [c["text"] for c in all_chunks],
        model_path=embed_model,
        revision=embed_revision,
        device=device,
        batch_size=batch_size,
        embed_url=embed_url,
    )
    lance_rows = [{**chunk, "vector": vec} for chunk, vec in zip(all_chunks, vectors)]

    os.makedirs(lancedb_path, exist_ok=True)
    db = lancedb.connect(lancedb_path)
    existing_tables = set(db.table_names())
    if reset or lance_table_name not in existing_tables:
        db.create_table(lance_table_name, data=lance_rows, mode="overwrite")
    else:
        table = db.open_table(lance_table_name)
        if stale_ids:
            quoted = ", ".join(f"'{cid}'" for cid in stale_ids)
            table.delete(f"id IN ({quoted})")
        table.add(lance_rows)

    if reset and os.path.exists(tantivy_path):
        shutil.rmtree(tantivy_path)
    os.makedirs(tantivy_path, exist_ok=True)
    schema = build_tantivy_schema(tantivy)
    t_index = tantivy.Index(schema, path=tantivy_path)
    writer = t_index.writer()
    for cid in stale_ids:
        writer.delete_documents("id", cid)
    for chunk in all_chunks:
        writer.add_document(
            tantivy.Document(
                id=chunk["id"],
                text=chunk["text"],
                domain=chunk["domain"],
                filename=chunk["filename"],
                url=chunk["url"],
                source=chunk["source"],
            )
        )
    writer.commit()

    for _, rel_key, fp in pending_files:
        manifest[rel_key] = {**fp, "chunk_ids": per_file_ids.get(rel_key, [])}
    _save_manifest(lancedb_path, manifest)
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
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device (cuda, mps, cpu)")
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
    summary = ingest(
        data_dir=args.data_dir,
        lancedb_path=args.lancedb_path,
        tantivy_path=args.tantivy_path,
        lance_table_name=args.lance_table,
        embed_model=args.embed_model,
        embed_revision=args.embed_revision,
        embed_url=args.embed_url,
        device=args.device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reset=args.reset,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
