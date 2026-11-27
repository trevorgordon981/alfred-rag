"""Shared configuration, schema, index-generation, and offline-backend utilities
for alfred-rag (used by both rag_server.py and scripts/ingest.py).
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

# --- Shared filenames for dual-store coordination ---
MANIFEST_FILENAME = ".ingest_manifest.json"
JOURNAL_FILENAME = ".ingest_journal.json"
GENERATION_FILENAME = ".generation.json"
INGEST_LOCK_FILENAME = ".ingest.lock"

# Environment variables for verified parent->child exclusive ingest lock handoff.
INGEST_LOCK_ENV_PATH = "ALFRED_INGEST_LOCK_PATH"
INGEST_LOCK_ENV_FD = "ALFRED_INGEST_LOCK_FD"
INGEST_LOCK_ENV_TOKEN = "ALFRED_INGEST_LOCK_TOKEN"

# --- Shared default model revisions ---
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_EMBED_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_RERANKER_REVISION = "77d193c791ed757ca307ee72715aa132723da912"

# --- Shared text cleaning ---
BOILERPLATE_PATTERNS = [
    r"(?i)^.*?Discussion Forums\s*",
    r"(?i)^LOG\s*IN\s*",
    r"(?i)^REGISTER\s*",
    r"(?i)^View First.*?\n",
    r"(?i)^View Public Profile.*?\n",
    r"(?i)^Find More Posts by.*?\n",
    r"(?i)^Share\s*Share Options\s*",
    r"(?i)^Quote:\s*Originally Posted by\s*\S+\s*",
    r"#\s*\d+\s*$",
]

# --- Source classification ---
VALID_SOURCE_CLASSES = {"curated", "vendor"}
DEFAULT_VENDOR_DIRS = frozenset({"docs", "vendor"})

# --- Loopback hosts allowed without external bind opt-in ---
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

VALID_DEVICES = frozenset({"", "auto", "cuda", "mps", "cpu"})


def clean_chunk_text(text: str) -> str:
    """Strip common forum navigation boilerplate and normalize whitespace."""
    cleaned = text
    for pat in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def normalize_rel_path(path: str) -> str:
    """Normalize a corpus-relative path to POSIX forward slashes without leading ./."""
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm.lstrip("/")


def classify_source_path(
    rel_path: str,
    explicit_class: str | None = None,
    vendor_dirs: frozenset[str] = DEFAULT_VENDOR_DIRS,
) -> str:
    """Determine explicit source_class ('vendor' or 'curated') for a document."""
    if explicit_class:
        normalized = explicit_class.strip().lower()
        if normalized in VALID_SOURCE_CLASSES:
            return normalized
    norm = normalize_rel_path(rel_path)
    parts = [p.lower() for p in Path(norm).parts[:-1]]
    if any(p in vendor_dirs for p in parts):
        return "vendor"
    return "curated"


# --- Canonical v2 schema contracts (shared by ingest and server) ---
REQUIRED_LANCE_COLUMNS = frozenset(
    {
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
        "vector",
    }
)
REQUIRED_TANTIVY_FIELDS = frozenset(
    {"id", "doc_id", "text", "domain", "filename", "url", "source", "source_class"}
)
TANTIVY_RAW_FIELDS = frozenset({"id", "doc_id", "domain", "source_class"})


def build_tantivy_schema(tantivy_mod: Any):
    """Build the canonical Tantivy schema shared by ingest and server."""
    sb = tantivy_mod.SchemaBuilder()
    sb.add_text_field("id", stored=True, tokenizer_name="raw")
    sb.add_text_field("doc_id", stored=True, tokenizer_name="raw")
    sb.add_text_field("text", stored=True, tokenizer_name="en_stem")
    sb.add_text_field("domain", stored=True, tokenizer_name="raw")
    sb.add_text_field("filename", stored=True)
    sb.add_text_field("url", stored=True)
    sb.add_text_field("source", stored=True)
    sb.add_text_field("source_class", stored=True, tokenizer_name="raw")
    return sb.build()


def inspect_tantivy_meta_schema(tantivy_path: str) -> list[str]:
    """Return schema incompatibility reasons for an existing Tantivy index dir.

    Returns an empty list if no Tantivy index exists yet or if its schema matches
    the current v2 contract (all required fields + raw tokenizers on identifier
    and facet fields).
    """
    meta_path = os.path.join(tantivy_path, "meta.json")
    if not os.path.isfile(meta_path):
        return []
    meta = read_json_optional(meta_path)
    if not isinstance(meta, dict) or not isinstance(meta.get("schema"), list):
        return ["Tantivy meta.json schema is unreadable"]

    fields_by_name: dict[str, dict[str, Any]] = {}
    for entry in meta["schema"]:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            fields_by_name[entry["name"]] = entry

    reasons: list[str] = []
    missing = sorted(REQUIRED_TANTIVY_FIELDS - set(fields_by_name.keys()))
    if missing:
        reasons.append(f"Tantivy schema missing fields: {', '.join(missing)}")

    for fname in sorted(TANTIVY_RAW_FIELDS):
        entry = fields_by_name.get(fname)
        if not entry:
            continue
        indexing = (entry.get("options") or {}).get("indexing") or {}
        tok = indexing.get("tokenizer")
        if tok != "raw":
            reasons.append(
                f"Tantivy field '{fname}' uses tokenizer {tok!r} instead of 'raw'"
            )
    return reasons


def inspect_lance_table_schema(table: Any) -> tuple[list[str], int | None]:
    """Return (incompatibility_reasons, vector_dim) for an opened LanceDB table."""
    reasons: list[str] = []
    vector_dim: int | None = None
    schema = getattr(table, "schema", None)
    names = getattr(schema, "names", None)
    if names is None:
        return reasons, vector_dim

    col_set = set(names)
    missing = sorted(REQUIRED_LANCE_COLUMNS - col_set)
    if missing:
        reasons.append(f"LanceDB table schema missing columns: {', '.join(missing)}")

    if "vector" in col_set and hasattr(schema, "field"):
        try:
            vec_field = schema.field("vector")
            list_size = getattr(getattr(vec_field, "type", None), "list_size", None)
            if isinstance(list_size, int) and list_size > 0:
                vector_dim = list_size
        except Exception:
            pass

    return reasons, vector_dim


def lance_table_names(db: Any) -> set[str]:
    """Return existing LanceDB table names across lancedb versions."""
    if hasattr(db, "list_tables"):
        try:
            listing = db.list_tables()
            if hasattr(listing, "tables"):
                return set(listing.tables)
            return set(listing)
        except Exception:
            pass
    if hasattr(db, "table_names"):
        return set(db.table_names())
    return set()


def resolve_device(requested: str | None = None) -> str:
    """Resolve torch device ('cuda', 'mps', or 'cpu') from explicit value or DEVICE env."""
    raw = (requested if requested is not None else os.environ.get("DEVICE", "")).strip().lower()
    if raw not in VALID_DEVICES:
        raise ValueError(
            f"Unsupported DEVICE={raw!r}; expected one of: auto, cuda, mps, cpu"
        )
    if raw and raw != "auto":
        return raw

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_bind_config(
    host: str | None = None,
    port: int | None = None,
    allow_external_bind: bool | None = None,
    api_key: str | None = None,
) -> tuple[str, int]:
    """Resolve and validate server bind host/port with loopback-by-default safety.

    Non-loopback binds (e.g. 0.0.0.0) require BOTH explicit opt-in
    (ALLOW_EXTERNAL_BIND=1 / --allow-external-bind) AND a non-empty RAG_API_KEY.
    """
    resolved_host = (host if host is not None else os.environ.get("HOST", "127.0.0.1")).strip()
    if not resolved_host:
        resolved_host = "127.0.0.1"

    raw_port = port if port is not None else os.environ.get("PORT", "9000")
    try:
        resolved_port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PORT value: {raw_port!r}") from exc
    if not (1 <= resolved_port <= 65535):
        raise ValueError(f"PORT out of valid range (1-65535): {resolved_port}")

    if allow_external_bind is None:
        env_flag = os.environ.get("ALLOW_EXTERNAL_BIND", "").strip().lower()
        allow_external = env_flag in {"1", "true", "yes", "on"}
    else:
        allow_external = bool(allow_external_bind)

    resolved_key = (
        api_key if api_key is not None else os.environ.get("RAG_API_KEY", "")
    ).strip()

    if resolved_host not in LOOPBACK_HOSTS:
        if not allow_external:
            raise ValueError(
                f"Refusing to bind non-loopback host {resolved_host!r} by default. "
                "Set ALLOW_EXTERNAL_BIND=1 (or pass --allow-external-bind) together "
                "with RAG_API_KEY to expose the server on a network interface."
            )
        if not resolved_key:
            raise ValueError(
                f"Refusing to bind non-loopback host {resolved_host!r} without authentication. "
                "Set RAG_API_KEY before exposing the server externally."
            )

    return resolved_host, resolved_port


# --- Generation & journal file helpers ---


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_json_optional(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def generation_marker_path(store_dir: str) -> str:
    return os.path.join(store_dir, GENERATION_FILENAME)


def journal_file_path(lancedb_path: str) -> str:
    return os.path.join(lancedb_path, JOURNAL_FILENAME)


# --- Per-corpus exclusive ingest writer lock (shared by HTTP server and CLI) ---

_INGEST_LOCK_MUTEX = threading.Lock()
_ACTIVE_INGEST_LEASES: dict[str, dict[str, Any]] = {}


def canonical_lock_path(path: str) -> str:
    """Resolve symlinks and relative segments so threads/processes share one key."""
    return os.path.realpath(os.path.abspath(path))


def ingest_lock_path(lancedb_path: str) -> str:
    """Canonical per-corpus lock file path inside the LanceDB directory."""
    return canonical_lock_path(os.path.join(lancedb_path, INGEST_LOCK_FILENAME))


def ingest_lease_path(lock_path: str) -> str:
    """Atomic JSON lease metadata companion path for a canonical lock file."""
    return f"{canonical_lock_path(lock_path)}.lease.json"


class IngestLockBusyError(RuntimeError):
    """Raised when another HTTP or CLI ingest already owns the per-corpus lock."""

    def __init__(
        self,
        lock_path: str,
        holder: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        self.lock_path = canonical_lock_path(lock_path)
        self.holder: dict[str, Any] = dict(holder) if isinstance(holder, dict) else {}
        raw_pid = self.holder.get("pid")
        if raw_pid is None:
            raw_pid = self.holder.get("parent_pid")
        self.holder_pid: int | None = int(raw_pid) if isinstance(raw_pid, int) else None
        self.holder_mode: str | None = (
            str(self.holder["mode"]) if self.holder.get("mode") is not None else None
        )
        self.holder_owner: str | None = (
            str(self.holder["owner"]) if self.holder.get("owner") is not None else None
        )
        self.holder_state: str | None = (
            str(self.holder["state"]) if self.holder.get("state") is not None else None
        )
        if message is None:
            parts: list[str] = []
            if self.holder_pid is not None:
                parts.append(f"pid={self.holder_pid}")
            if self.holder_mode:
                parts.append(f"mode={self.holder_mode}")
            if self.holder_owner:
                parts.append(f"owner={self.holder_owner}")
            if self.holder_state:
                parts.append(f"state={self.holder_state}")
            detail = f" ({', '.join(parts)})" if parts else ""
            message = f"Ingest lock already held for corpus at {self.lock_path}{detail}"
        super().__init__(message)


class IngestLockLease:
    """Exclusive per-corpus writer lease backed by flock + in-process registry + atomic lease file."""

    def __init__(
        self,
        *,
        lock_path: str,
        lease_path: str,
        fd: int,
        token: str,
        metadata: dict[str, Any],
    ) -> None:
        self.lock_path = lock_path
        self.lease_path = lease_path
        self.fd = fd
        self.token = token
        self.metadata = dict(metadata)
        self._closed = False

    def child_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Build child environment carrying verified lock-handoff credentials."""
        env = dict(os.environ if base_env is None else base_env)
        env[INGEST_LOCK_ENV_PATH] = self.lock_path
        env[INGEST_LOCK_ENV_FD] = str(self.fd)
        env[INGEST_LOCK_ENV_TOKEN] = self.token
        return env

    def handoff_to_child(self, child_pid: int) -> None:
        """Hand kernel flock ownership to a spawned child that inherited self.fd.

        Updates the atomic lease file with the child PID if still in 'spawning'
        state, then closes the parent's fd WITHOUT calling LOCK_UN so the child's
        open file description keeps the exclusive flock until the child exits.
        """
        with _INGEST_LOCK_MUTEX:
            if self._closed:
                return
            self._closed = True
            try:
                current = read_json_optional(self.lease_path)
                if (
                    current is not None
                    and isinstance(current.get("token"), str)
                    and hmac.compare_digest(current["token"], self.token)
                    and current.get("state") == "spawning"
                ):
                    updated = dict(current)
                    updated["pid"] = int(child_pid)
                    updated["state"] = "running"
                    write_json_atomic(self.lease_path, updated)
                    self.metadata = updated
            except OSError:
                pass
            try:
                os.close(self.fd)
            except OSError:
                pass
            active = _ACTIVE_INGEST_LEASES.get(self.lock_path)
            if active is not None and active.get("token") == self.token:
                _ACTIVE_INGEST_LEASES.pop(self.lock_path, None)

    def release(self) -> None:
        """Release the exclusive lock completely (on writer completion or spawn failure)."""
        with _INGEST_LOCK_MUTEX:
            if self._closed:
                return
            self._closed = True
            try:
                current = read_json_optional(self.lease_path)
                if (
                    current is not None
                    and isinstance(current.get("token"), str)
                    and hmac.compare_digest(current["token"], self.token)
                ):
                    try:
                        os.unlink(self.lease_path)
                    except FileNotFoundError:
                        pass
            except OSError:
                pass
            try:
                if fcntl is not None:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self.fd)
            except OSError:
                pass
            active = _ACTIVE_INGEST_LEASES.get(self.lock_path)
            if active is not None and active.get("token") == self.token:
                _ACTIVE_INGEST_LEASES.pop(self.lock_path, None)

    def __enter__(self) -> IngestLockLease:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def acquire_ingest_lock(
    lock_path: str,
    *,
    mode: str,
    owner: str,
    state: str = "running",
    pid: int | None = None,
) -> IngestLockLease:
    """Atomically acquire the non-blocking exclusive per-corpus ingest lock.

    Never unlinks the flock anchor file so concurrent openers always synchronize
    on the exact same inode. Raises IngestLockBusyError immediately if any
    thread or process already holds the lock.
    """
    canon = canonical_lock_path(lock_path)
    lpath = ingest_lease_path(canon)
    parent = os.path.dirname(canon)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with _INGEST_LOCK_MUTEX:
        active = _ACTIVE_INGEST_LEASES.get(canon)
        if active is not None:
            raise IngestLockBusyError(canon, active)

        fd = os.open(canon, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                        holder = read_json_optional(lpath)
                        raise IngestLockBusyError(canon, holder) from exc
                    raise

            token = secrets.token_hex(16)
            metadata: dict[str, Any] = {
                "schema_version": 1,
                "lock_path": canon,
                "token": token,
                "pid": pid,
                "parent_pid": os.getpid(),
                "mode": mode,
                "owner": owner,
                "state": state,
                "acquired_at": utc_now_iso(),
            }
            write_json_atomic(lpath, metadata)
            _ACTIVE_INGEST_LEASES[canon] = dict(metadata)
            return IngestLockLease(
                lock_path=canon,
                lease_path=lpath,
                fd=fd,
                token=token,
                metadata=metadata,
            )
        except BaseException:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            raise


def adopt_or_acquire_ingest_lock(
    lock_path: str,
    *,
    mode: str,
    owner: str = "cli",
) -> IngestLockLease:
    """Adopt a verified inherited HTTP-spawned lock fd if present, else acquire a fresh lock."""
    canon = canonical_lock_path(lock_path)
    lpath = ingest_lease_path(canon)

    env_path = os.environ.get(INGEST_LOCK_ENV_PATH, "").strip()
    env_fd_raw = os.environ.get(INGEST_LOCK_ENV_FD, "").strip()
    env_token = os.environ.get(INGEST_LOCK_ENV_TOKEN, "").strip()

    if env_path and env_fd_raw and env_token and canonical_lock_path(env_path) == canon:
        os.environ.pop(INGEST_LOCK_ENV_PATH, None)
        os.environ.pop(INGEST_LOCK_ENV_FD, None)
        os.environ.pop(INGEST_LOCK_ENV_TOKEN, None)
        try:
            inherited_fd = int(env_fd_raw)
            if inherited_fd >= 0:
                st_fd = os.fstat(inherited_fd)
                st_file = os.stat(canon)
                same_inode = (
                    st_fd.st_dev == st_file.st_dev and st_fd.st_ino == st_file.st_ino
                )
                lease_data = read_json_optional(lpath)
                token_ok = (
                    lease_data is not None
                    and isinstance(lease_data.get("token"), str)
                    and hmac.compare_digest(lease_data["token"], env_token)
                )
                if same_inode and token_ok:
                    with _INGEST_LOCK_MUTEX:
                        active = _ACTIVE_INGEST_LEASES.get(canon)
                        if active is not None:
                            raise IngestLockBusyError(canon, active)
                        if fcntl is not None:
                            try:
                                fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except OSError as exc:
                                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                                    raise IngestLockBusyError(canon, lease_data) from exc
                                raise
                        updated: dict[str, Any] = {
                            "schema_version": 1,
                            "lock_path": canon,
                            "token": env_token,
                            "pid": os.getpid(),
                            "parent_pid": lease_data.get("parent_pid"),
                            "mode": mode,
                            "owner": "http-worker",
                            "state": "running",
                            "acquired_at": lease_data.get("acquired_at") or utc_now_iso(),
                        }
                        write_json_atomic(lpath, updated)
                        _ACTIVE_INGEST_LEASES[canon] = dict(updated)
                        return IngestLockLease(
                            lock_path=canon,
                            lease_path=lpath,
                            fd=inherited_fd,
                            token=env_token,
                            metadata=updated,
                        )
        except IngestLockBusyError:
            raise
        except (ValueError, OSError):
            pass

    return acquire_ingest_lock(
        canon,
        mode=mode,
        owner=owner,
        state="running",
        pid=os.getpid(),
    )


def inspect_ingest_lock(lock_path: str) -> dict[str, Any] | None:
    """Return active lease metadata if the per-corpus lock is currently held, else None."""
    canon = canonical_lock_path(lock_path)
    lpath = ingest_lease_path(canon)
    with _INGEST_LOCK_MUTEX:
        active = _ACTIVE_INGEST_LEASES.get(canon)
        if active is not None:
            return dict(active)

        if not os.path.exists(canon) or fcntl is None:
            return None

        try:
            fd = os.open(canon, os.O_RDWR)
        except OSError:
            return None
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                    holder = read_json_optional(lpath) or {}
                    holder.setdefault("lock_path", canon)
                    return holder
                return None
            else:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                return None
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


# --- Deterministic CPU-only offline models (no GPU / no HF weights required) ---

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "vs",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)


def _extract_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stem_token(tok: str) -> str:
    """Tiny suffix normalizer so offline lexical matching handles plurals/gerunds."""
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


class _VectorBatch(list):
    """List wrapper exposing .tolist() for parity with numpy/SentenceTransformer outputs."""

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self]


class OfflineHashEmbedder:
    """Deterministic feature-hashing dense embedder for CPU-only testing and demos.

    Projects lowercased unigrams, stemmed unigrams, bigrams, and 4-char subword
    shingles into an L2-normalized dense vector using BLAKE2b (never Python's
    salted hash()), so results are bit-for-bit reproducible across processes.
    """

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        # Strip instruction prefix if present so query/document spaces align.
        if "\nQuery: " in text:
            text = text.split("\nQuery: ", 1)[1]
        tokens = _extract_tokens(text)
        vec = [0.0] * self.dim
        if not tokens:
            return vec

        features: list[tuple[str, float]] = []
        content_tokens = [t for t in tokens if t not in _STOPWORDS]
        for tok in tokens:
            w = 0.25 if tok in _STOPWORDS else 1.5
            features.append((f"w:{tok}", w))
            stem = _stem_token(tok)
            if stem != tok:
                features.append((f"s:{stem}", w * 0.9))
            if len(tok) >= 4 and tok not in _STOPWORDS:
                for i in range(len(tok) - 3):
                    features.append((f"c:{tok[i:i+4]}", 0.35))
        for i in range(len(content_tokens) - 1):
            features.append((f"b:{content_tokens[i]}_{content_tokens[i+1]}", 1.8))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            inv = 1.0 / norm
            vec = [v * inv for v in vec]
        return vec

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> _VectorBatch:
        del batch_size, show_progress_bar
        return _VectorBatch([self._embed_one(t) for t in texts])


class OfflineLexicalReranker:
    """Deterministic CPU-only reranker scoring (query, document) overlap in (0, 1).

    Rewards content-term coverage, stemmed coverage, and exact bigram matches,
    while assigning low scores (<0.25) when a document shares no meaningful query
    terms (enabling realistic out-of-domain / no-result evaluation).
    """

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        q_tokens = _extract_tokens(query)
        q_content = [t for t in q_tokens if t not in _STOPWORDS and len(t) > 1]
        if not q_content:
            q_content = q_tokens
        q_stems = {_stem_token(t) for t in q_content}
        q_bigrams = {
            (q_content[i], q_content[i + 1]) for i in range(len(q_content) - 1)
        }

        scores: list[float] = []
        for doc in documents:
            cleaned = clean_chunk_text(doc)
            d_tokens = _extract_tokens(cleaned)
            if not d_tokens or not q_content:
                scores.append(0.05)
                continue
            d_set = set(d_tokens)
            d_stems = {_stem_token(t) for t in d_tokens}
            d_content = [t for t in d_tokens if t not in _STOPWORDS and len(t) > 1]
            d_bigrams = {
                (d_content[i], d_content[i + 1]) for i in range(len(d_content) - 1)
            }

            exact_hits = sum(1 for t in set(q_content) if t in d_set)
            stem_hits = sum(1 for s in q_stems if s in d_stems)
            exact_cov = exact_hits / max(len(set(q_content)), 1)
            stem_cov = stem_hits / max(len(q_stems), 1)
            bigram_cov = (
                len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0
            )

            # Linear combination mapped through a calibrated logistic sigmoid.
            # Zero overlap -> logit -2.6 -> ~0.069; strong match -> logit +2.4 -> ~0.92.
            logit = -2.6 + 3.2 * stem_cov + 1.4 * exact_cov + 1.6 * bigram_cov
            prob = 1.0 / (1.0 + math.exp(-logit))
            scores.append(round(prob, 6))
        return scores
