"""Shared configuration, schema, index-generation, and offline-backend utilities
for alfred-rag (used by both rag_server.py and scripts/ingest.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- Shared filenames for dual-store coordination ---
MANIFEST_FILENAME = ".ingest_manifest.json"
JOURNAL_FILENAME = ".ingest_journal.json"
GENERATION_FILENAME = ".generation.json"

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
    tmp = f"{path}.tmp.{os.getpid()}"
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
