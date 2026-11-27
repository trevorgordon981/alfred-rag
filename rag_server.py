"""FastAPI hybrid RAG server: LanceDB dense vector search + Tantivy BM25 +
Reciprocal Rank Fusion + cross-encoder reranking.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from functools import lru_cache
from typing import Any

import lancedb
import tantivy
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field, field_validator

from rag_common import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_REVISION,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    OfflineHashEmbedder,
    OfflineLexicalReranker,
    build_tantivy_schema as _common_build_tantivy_schema,
    clean_chunk_text as _common_clean_chunk_text,
    generation_marker_path,
    journal_file_path,
    lance_table_names,
    normalize_rel_path,
    read_json_optional,
    resolve_bind_config,
    resolve_device,
)

# Optional heavy ML imports (only required when RAG_BACKEND != "offline").
try:
    import torch
except ImportError:  # pragma: no cover - optional for offline CPU mode
    torch = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional for offline CPU mode
    SentenceTransformer = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional for offline CPU mode
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

# Deployment root. Everything below resolves under it, so the server runs from any
# checkout without editing paths. Override any individual path with its own env var.
RAG_ROOT = os.environ.get("RAG_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Immutable Hub revisions. Override alongside a private/local model path when
# deliberately qualifying another revision.
EMBED_MODEL_REVISION = os.environ.get("EMBED_MODEL_REVISION", DEFAULT_EMBED_REVISION)
RERANKER_MODEL_REVISION = os.environ.get("RERANKER_MODEL_REVISION", DEFAULT_RERANKER_REVISION)

# Backend mode: "hf" (Qwen3 HuggingFace models) or "offline" (deterministic CPU demo backend).
RAG_BACKEND = os.environ.get("RAG_BACKEND", "hf").strip().lower()
DEVICE = os.environ.get("DEVICE", "")

# Security & request limits.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "9000"))
ALLOW_EXTERNAL_BIND = os.environ.get("ALLOW_EXTERNAL_BIND", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_API_KEY = os.environ.get("RAG_API_KEY", "").strip()
RAG_ADMIN_API_KEY = os.environ.get("RAG_ADMIN_API_KEY", "").strip()
MAX_REQUEST_BODY_BYTES = int(os.environ.get("MAX_REQUEST_BODY_BYTES", str(2 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "32768"))

# Retrieval policy settings.
QUERY_INSTRUCTION = os.environ.get(
    "QUERY_INSTRUCTION",
    "Instruct: Given a user query, retrieve the most relevant passages from the knowledge base.\nQuery: ",
)
RERANK_CANDIDATE_CAP = int(os.environ.get("RERANK_CANDIDATE_CAP", "10"))

app = FastAPI(title="Alfred RAG Server")


@app.middleware("http")
async def _enforce_body_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request body exceeds limit of {MAX_REQUEST_BODY_BYTES} bytes"
                    },
                )
        except ValueError:
            pass
    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds limit of {MAX_REQUEST_BODY_BYTES} bytes"},
        )
    return await call_next(request)


# --- Authentication dependencies ---


def _extract_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _require_api_auth(request: Request) -> None:
    """Enforce API key authentication when RAG_API_KEY is configured."""
    expected = RAG_API_KEY
    if not expected:
        return
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    if hmac.compare_digest(token, expected):
        return
    if RAG_ADMIN_API_KEY and hmac.compare_digest(token, RAG_ADMIN_API_KEY):
        return
    raise HTTPException(status_code=401, detail="Invalid API key")


def _require_admin_auth(request: Request) -> None:
    """Enforce administrative authorization for destructive operations such as reset."""
    admin_key = RAG_ADMIN_API_KEY
    api_key = RAG_API_KEY
    if not admin_key and not api_key:
        return
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing administrative API key")
    if admin_key:
        if hmac.compare_digest(token, admin_key):
            return
        if api_key and hmac.compare_digest(token, api_key):
            raise HTTPException(status_code=403, detail="Administrative API key required")
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not hmac.compare_digest(token, api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Prometheus metrics (stable single-prefix exporter contract) ---
# This server embeds and reranks; it never generates tokens, so reranker work
# is exposed as scored-document decisions rather than an output-token counter.
RAG_TOKENS_INPUT = Counter(
    "llm_tokens_input_total",
    "Total input tokens processed by embedding and reranking endpoints",
    ["model", "endpoint", "job"],
)
RAG_RERANK_SCORED_DOCS = Counter(
    "llm_rerank_scored_docs_total",
    "Total documents scored by the reranker cross-encoder (one yes/no decision each)",
    ["model", "endpoint", "job"],
)
RAG_REQUESTS = Counter(
    "llm_requests_total",
    "Total inference requests processed",
    ["model", "endpoint", "job"],
)
RAG_TOOL_CALLS = Counter(
    "llm_tool_calls_total",
    "Total search invocations",
    ["model", "endpoint", "job"],
)

LABEL_EMBED = {"model": "Qwen3-Embedding-8B", "endpoint": "/embed", "job": "rag-server"}
LABEL_RERANK = {"model": "Qwen3-Reranker-8B", "endpoint": "/rerank", "job": "rag-server"}

# --- Models and indexes (loaded at startup) ---
embed_model: Any = None
reranker_tokenizer: Any = None
reranker_model: Any = None
lance_table: Any = None
tantivy_index: Any = None
loaded_generation: int | None = None
active_device: str = "cpu"
active_models: list[str] = ["qwen3-embedding-8b", "qwen3-reranker-8b"]

# yes/no token ids for the HF reranker, resolved once at startup.
RERANK_YES_ID: int | None = None
RERANK_NO_ID: int | None = None

# Serialize GPU compute (embed + rerank forward) so concurrent requests on
# FastAPI's threadpool cannot interleave on a single accelerator.
_GPU_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", os.path.join(RAG_ROOT, "indexes", "lancedb"))
TANTIVY_PATH = os.environ.get("TANTIVY_PATH", os.path.join(RAG_ROOT, "indexes", "tantivy"))
LANCE_TABLE = os.environ.get("LANCE_TABLE", "alfred_rag")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, values: list[str]) -> list[str]:
        for text in values:
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"Text entry exceeds max length of {MAX_TEXT_CHARS} chars")
        return values


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    documents: list[str] = Field(..., min_length=1, max_length=64)
    top_k: int = Field(5, ge=1, le=64)

    @field_validator("documents")
    @classmethod
    def _validate_docs(cls, values: list[str]) -> list[str]:
        for doc in values:
            if len(doc) > MAX_TEXT_CHARS:
                raise ValueError(f"Document exceeds max length of {MAX_TEXT_CHARS} chars")
        return values


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(5, ge=1, le=50)
    dense_top_n: int = Field(30, ge=1, le=200)
    bm25_top_n: int = Field(30, ge=1, le=200)
    rrf_k: int = Field(60, ge=1, le=1000)
    domain: str | None = None
    max_per_source: int = Field(2, ge=1, le=20)
    domain_boost: dict[str, float] | None = None
    rerank: bool = True  # set False for fast dense+BM25+RRF path (skips cross-encoder)


class HealthResponse(BaseModel):
    status: str
    ready: bool
    models: list[str]
    indexes: dict
    uptime: float
    reason: str | None = None
    generation: int | None = None


START_TIME = time.time()


def build_tantivy_schema():
    """Rebuild the canonical Tantivy schema matching scripts/ingest.py."""
    return _common_build_tantivy_schema(tantivy)


def rrf_merge_with_scores(
    dense_ids: list[str], bm25_ids: list[str], k: int = 60
) -> tuple[list[str], dict[str, float]]:
    """Reciprocal Rank Fusion with per-leg deduplication.

    Deduplicating each retriever leg ensures repeated IDs inside a single leg
    can never be double-counted.
    """
    scores: dict[str, float] = {}

    seen_dense: set[str] = set()
    rank = 0
    for doc_id in dense_ids:
        if doc_id in seen_dense:
            continue
        seen_dense.add(doc_id)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        rank += 1

    seen_bm25: set[str] = set()
    rank = 0
    for doc_id in bm25_ids:
        if doc_id in seen_bm25:
            continue
        seen_bm25.add(doc_id)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        rank += 1

    merged_ids = sorted(scores.keys(), key=lambda x: (-scores[x], x))
    return merged_ids, scores


def rrf_merge(dense_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. Returns merged list of IDs ordered by RRF score."""
    merged_ids, _ = rrf_merge_with_scores(dense_ids, bm25_ids, k=k)
    return merged_ids


def _open_indexes() -> None:
    """Open or reopen LanceDB and Tantivy handles from LANCEDB_PATH / TANTIVY_PATH."""
    global lance_table, tantivy_index, loaded_generation

    with _INDEX_LOCK:
        db = lancedb.connect(LANCEDB_PATH)
        if LANCE_TABLE in lance_table_names(db):
            lance_table = db.open_table(LANCE_TABLE)
        else:
            lance_table = None

        if os.path.isdir(TANTIVY_PATH):
            schema = build_tantivy_schema()
            tantivy_index = tantivy.Index(schema, path=TANTIVY_PATH)
            tantivy_index.reload()
        else:
            tantivy_index = None

        lance_gen = read_json_optional(generation_marker_path(LANCEDB_PATH))
        tantivy_gen = read_json_optional(generation_marker_path(TANTIVY_PATH))
        if (
            lance_gen
            and tantivy_gen
            and lance_gen.get("generation") == tantivy_gen.get("generation")
        ):
            loaded_generation = int(lance_gen["generation"])
        else:
            loaded_generation = None


def _refresh_indexes():
    """Pull the latest LanceDB fragments and Tantivy segments into live readers.

    Detects generation changes (for example after an atomic reset swap) and
    reopens readers when the generation marker advances.
    """
    global loaded_generation
    try:
        lance_gen = read_json_optional(generation_marker_path(LANCEDB_PATH))
        tantivy_gen = read_json_optional(generation_marker_path(TANTIVY_PATH))
        disk_gen = None
        if (
            lance_gen
            and tantivy_gen
            and lance_gen.get("generation") == tantivy_gen.get("generation")
        ):
            disk_gen = int(lance_gen["generation"])

        if disk_gen is not None and disk_gen != loaded_generation:
            _open_indexes()
            return

        with _INDEX_LOCK:
            if lance_table is not None:
                lance_table.checkout_latest()
            if tantivy_index is not None:
                tantivy_index.reload()
    except Exception as e:  # never let a refresh blip kill the timer thread
        print(f"[refresh] index refresh failed: {e}")


def _start_refresh_timer(interval: float = 30.0):
    def _loop():
        while True:
            time.sleep(interval)
            _refresh_indexes()

    threading.Thread(target=_loop, daemon=True, name="index-refresh").start()


def configure_offline_runtime(
    lancedb_path: str | None = None,
    tantivy_path: str | None = None,
    lance_table_name: str | None = None,
) -> None:
    """Configure deterministic CPU-only models and open local indexes (used by offline mode/demo)."""
    global embed_model, reranker_tokenizer, reranker_model, active_device, active_models
    global LANCEDB_PATH, TANTIVY_PATH, LANCE_TABLE, LABEL_EMBED, LABEL_RERANK

    if lancedb_path is not None:
        LANCEDB_PATH = lancedb_path
    if tantivy_path is not None:
        TANTIVY_PATH = tantivy_path
    if lance_table_name is not None:
        LANCE_TABLE = lance_table_name

    embed_model = OfflineHashEmbedder()
    reranker_tokenizer = None
    reranker_model = OfflineLexicalReranker()
    active_device = "cpu"
    active_models = ["offline-hash-embedder-256d", "offline-lexical-reranker"]
    LABEL_EMBED = {
        "model": "offline-hash-embedder-256d",
        "endpoint": "/embed",
        "job": "rag-server",
    }
    LABEL_RERANK = {
        "model": "offline-lexical-reranker",
        "endpoint": "/rerank",
        "job": "rag-server",
    }
    _cached_query_embedding.cache_clear()
    _open_indexes()


@app.on_event("startup")
def load_models():
    global embed_model, reranker_tokenizer, reranker_model
    global RERANK_YES_ID, RERANK_NO_ID, active_device, active_models

    backend = os.environ.get("RAG_BACKEND", RAG_BACKEND).strip().lower()
    if backend == "offline":
        print("Starting in deterministic CPU offline mode (RAG_BACKEND=offline)...")
        configure_offline_runtime()
        _start_refresh_timer()
        return

    if torch is None or SentenceTransformer is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "PyTorch, sentence-transformers, and transformers are required for RAG_BACKEND=hf. "
            "Install requirements.txt or set RAG_BACKEND=offline for the CPU demo backend."
        )

    resolved_device = resolve_device(os.environ.get("DEVICE", DEVICE))
    active_device = resolved_device
    active_models = ["qwen3-embedding-8b", "qwen3-reranker-8b"]
    embed_path = os.environ.get("EMBED_MODEL_PATH", DEFAULT_EMBED_MODEL)
    reranker_path = os.environ.get("RERANKER_MODEL_PATH", DEFAULT_RERANKER_MODEL)

    print(f"Loading embedding model ({embed_path}) on device={resolved_device}...")
    embed_kwargs: dict[str, Any] = {"device": resolved_device}
    if EMBED_MODEL_REVISION and not os.path.exists(embed_path):
        embed_kwargs["revision"] = EMBED_MODEL_REVISION
    embed_model = SentenceTransformer(embed_path, **embed_kwargs)
    print("Embedding model loaded.")

    print(f"Loading reranker model ({reranker_path}) on device={resolved_device}...")
    tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if RERANKER_MODEL_REVISION and not os.path.exists(reranker_path):
        tok_kwargs["revision"] = RERANKER_MODEL_REVISION
        model_kwargs["revision"] = RERANKER_MODEL_REVISION

    model_dtype = torch.float16 if resolved_device in {"cuda", "mps"} else torch.float32
    model_kwargs["dtype"] = model_dtype
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "cuda"

    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path, **tok_kwargs)
    reranker_model = AutoModelForCausalLM.from_pretrained(reranker_path, **model_kwargs)
    if resolved_device != "cuda":
        reranker_model = reranker_model.to(resolved_device)
    reranker_model.eval()

    if reranker_tokenizer.pad_token_id is None:
        reranker_tokenizer.pad_token = reranker_tokenizer.eos_token
    reranker_tokenizer.padding_side = "left"
    RERANK_YES_ID = reranker_tokenizer.convert_tokens_to_ids("yes")
    RERANK_NO_ID = reranker_tokenizer.convert_tokens_to_ids("no")
    print("Reranker model loaded.")

    _open_indexes()
    _start_refresh_timer()


def _collect_index_health() -> tuple[int, int, list[str], int | None]:
    """Return (lance_count, tantivy_count, issues, generation) for the loaded indexes."""
    issues: list[str] = []
    generation: int | None = None

    # Check on-disk journal and generation alignment if present.
    journal = read_json_optional(journal_file_path(LANCEDB_PATH))
    if journal is not None:
        issues.append(
            f"Interrupted ingest journal detected (generation {journal.get('generation')}); "
            "run ingest to reconcile"
        )

    lance_gen = read_json_optional(generation_marker_path(LANCEDB_PATH))
    tantivy_gen = read_json_optional(generation_marker_path(TANTIVY_PATH))
    if lance_gen is not None or tantivy_gen is not None:
        if lance_gen is None or tantivy_gen is None:
            issues.append("Index generation marker missing from one store (LanceDB vs Tantivy)")
        elif lance_gen.get("generation") != tantivy_gen.get("generation"):
            issues.append(
                f"Index generation mismatch: LanceDB={lance_gen.get('generation')} "
                f"vs Tantivy={tantivy_gen.get('generation')}"
            )
        else:
            generation = int(lance_gen["generation"])

    if lance_table is not None:
        try:
            lance_table.checkout_latest()
            lance_count = int(lance_table.count_rows())
        except Exception as e:
            lance_count = 0
            issues.append(f"LanceDB index error: {e}")
    else:
        lance_count = 0

    if tantivy_index is not None:
        try:
            tantivy_index.reload()
            tantivy_count = int(tantivy_index.searcher().num_docs)
        except Exception as e:
            tantivy_count = 0
            issues.append(f"Tantivy index error: {e}")
    else:
        tantivy_count = 0

    if lance_count <= 0 and not any("LanceDB" in issue for issue in issues):
        issues.append("LanceDB index is empty (0 rows)")
    if tantivy_count <= 0 and not any("Tantivy" in issue for issue in issues):
        issues.append("Tantivy index is empty (0 docs)")

    return lance_count, tantivy_count, issues, generation


def _build_health_response(unhealthy_status: str) -> JSONResponse:
    lance_count, tantivy_count, issues, generation = _collect_index_health()
    is_ready = not issues
    payload = HealthResponse(
        status="ok" if is_ready else unhealthy_status,
        ready=is_ready,
        models=list(active_models),
        indexes={"lancedb": lance_count, "tantivy": tantivy_count},
        uptime=time.time() - START_TIME,
        reason=None if is_ready else "; ".join(issues),
        generation=generation,
    )
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content=payload.model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> JSONResponse:
    return _build_health_response(unhealthy_status="unhealthy")


@app.get("/ready", response_model=HealthResponse)
def ready() -> JSONResponse:
    return _build_health_response(unhealthy_status="not_ready")


@app.get("/metrics", dependencies=[Depends(_require_api_auth)])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/embed", dependencies=[Depends(_require_api_auth)])
def embed(req: EmbedRequest):
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")
    start = time.time()
    with _GPU_LOCK:
        embeddings = embed_model.encode(req.texts, batch_size=32).tolist()
    elapsed = time.time() - start
    token_count = sum(max(1, int(len(t.split()) / 0.75)) for t in req.texts)
    RAG_TOKENS_INPUT.labels(**LABEL_EMBED).inc(token_count)
    RAG_REQUESTS.labels(**LABEL_EMBED).inc()
    return {
        "embeddings": embeddings,
        "dim": len(embeddings[0]),
        "count": len(embeddings),
        "elapsed": round(elapsed, 3),
    }


@app.post("/rerank", dependencies=[Depends(_require_api_auth)])
def rerank(req: RerankRequest):
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model is not loaded")
    start = time.time()
    scores = _rerank_docs(req.query, req.documents)
    ranked = sorted(
        zip(req.documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )[: req.top_k]
    elapsed = time.time() - start
    RAG_REQUESTS.labels(**LABEL_RERANK).inc()
    return {
        "results": [{"document": doc, "score": round(s, 4)} for doc, s in ranked],
        "elapsed": round(elapsed, 3),
    }


@lru_cache(maxsize=4096)
def _clean_chunk_text(text: str) -> str:
    """Memoized boilerplate cleaner shared with scripts/ingest.py."""
    return _common_clean_chunk_text(text)


RERANK_SYSTEM_PROMPT = (
    "Judge whether the document answers the user's query. "
    "A relevant document directly addresses the specific question asked, "
    "not just the general topic. Respond with 'yes' or 'no'."
)
RERANK_BATCH_SIZE = 16


def _rerank_docs(query: str, documents: list[str]) -> list[float]:
    """Score documents against a query using either the offline or HF cross-encoder."""
    if not documents:
        return []

    # Offline / pluggable callable reranker path (CPU demo & tests).
    if reranker_tokenizer is None and hasattr(reranker_model, "score"):
        approx_tokens = sum(
            max(1, int((len(query.split()) + len(doc.split())) / 0.75))
            for doc in documents
        )
        RAG_TOKENS_INPUT.labels(**LABEL_RERANK).inc(approx_tokens)
        RAG_RERANK_SCORED_DOCS.labels(**LABEL_RERANK).inc(len(documents))
        return [float(s) for s in reranker_model.score(query, documents)]

    yes_id = RERANK_YES_ID
    no_id = RERANK_NO_ID

    prompts = []
    for doc in documents:
        cleaned = _clean_chunk_text(doc)
        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"query": query, "document": cleaned},
                    ensure_ascii=False,
                ),
            },
        ]
        prompts.append(
            reranker_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    scores: list[float] = []
    with _GPU_LOCK:
        for start in range(0, len(prompts), RERANK_BATCH_SIZE):
            batch = prompts[start : start + RERANK_BATCH_SIZE]
            inputs = reranker_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True,
            ).to(reranker_model.device)

            RAG_TOKENS_INPUT.labels(**LABEL_RERANK).inc(int(inputs["input_ids"].numel()))
            RAG_RERANK_SCORED_DOCS.labels(**LABEL_RERANK).inc(len(batch))

            with torch.no_grad():
                logits = reranker_model(**inputs, logits_to_keep=1).logits[:, -1, :]
                diff = logits[:, yes_id] - logits[:, no_id]
                scores.extend(torch.sigmoid(diff).tolist())
    return scores


_DOMAIN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TANTIVY_META_RE = re.compile(r"""['":()\[\]{}+\-^?*\\\/!~]""")

# Domains excluded from default retrieval (still searchable via explicit domain="<name>").
OPT_IN_DOMAINS: set[str] = set()

# Deprioritization factor applied to bulk vendor reference docs (source_class == "vendor").
DEPRIORITIZE_FACTOR = 0.6


def _is_vendor_doc(chunk: dict) -> bool:
    """True when a chunk belongs to the bulk 'vendor' source class.

    Prefers the explicit persisted `source_class` column written by scripts/ingest.py.
    Falls back to normalized path inspection only for legacy indexes missing `source_class`.
    """
    explicit = (chunk.get("source_class") or "").strip().lower()
    if explicit:
        return explicit == "vendor"
    norm = normalize_rel_path(str(chunk.get("doc_id") or chunk.get("source") or ""))
    parts = [p.lower() for p in norm.split("/")[:-1]]
    return any(p in {"docs", "vendor"} for p in parts)


def _diversification_key(chunk: dict) -> str:
    """Canonical per-document identity used for max_per_source diversification.

    Uses corpus-relative `doc_id` rather than bare filename so distinct files
    with the same basename (e.g. operations/notes.md vs engineering/notes.md)
    are never collapsed into a single source bucket.
    """
    return (
        chunk.get("doc_id")
        or chunk.get("source")
        or chunk.get("filename")
        or chunk.get("id")
        or ""
    )


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _recency_multiplier(chunk: dict, today: _dt.date | None = None) -> float:
    """Small decay multiplier from a YYYY-MM-DD date embedded in doc_id/filename/source.

    Newer => multiplier closer to 1.0; ~10y old bottoms out at 0.7.
    No parseable date => 1.0 (no penalty).
    """
    name = chunk.get("doc_id") or chunk.get("filename") or chunk.get("source") or ""
    m = _DATE_RE.search(name)
    if not m:
        return 1.0
    try:
        d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return 1.0
    ref_day = today or _dt.date.today()
    age_days = max(0, (ref_day - d).days)
    return 1.0 - min(age_days, 3650) / 3650 * 0.3


def _dense_search(query_vec: list[float], top_n: int, domain: str | None = None) -> list[dict]:
    """LanceDB vector search returning normalized candidate dicts."""
    if lance_table is None:
        return []
    q = lance_table.search(query_vec)
    if hasattr(q, "refine_factor"):
        q = q.refine_factor(20)
    if domain:
        if not _DOMAIN_RE.match(domain):
            return []
        q = q.where(f"domain = '{domain}'", prefilter=True)
    elif OPT_IN_DOMAINS:
        excl = ", ".join(f"'{d}'" for d in sorted(OPT_IN_DOMAINS))
        q = q.where(f"domain NOT IN ({excl})", prefilter=True)
    results = q.limit(top_n).to_list()
    out = []
    for r in results:
        out.append(
            {
                "id": r["id"],
                "doc_id": r.get("doc_id", "") or r.get("source", ""),
                "text": r["text"],
                "source": r.get("source", ""),
                "source_class": r.get("source_class", ""),
                "filename": r.get("filename", ""),
                "domain": r.get("domain", ""),
                "url": r.get("url", ""),
                "format": r.get("format", ""),
                "chunk_index": r.get("chunk_index", 0),
                "dense_score": float(r.get("_distance", 0.0)),
            }
        )
    return out


def _safe_doc_first(doc: Any, field: str) -> str:
    try:
        val = doc.get_first(field)
        return val or ""
    except Exception:
        return ""


def _bm25_search(query: str, top_n: int, domain: str | None = None) -> list[dict]:
    """Tantivy BM25 search returning normalized candidate dicts."""
    if tantivy_index is None:
        return []
    searcher = tantivy_index.searcher()
    safe_query = _TANTIVY_META_RE.sub(" ", query).strip()
    if not safe_query:
        return []
    fetch_n = top_n
    if domain:
        if not _DOMAIN_RE.match(domain):
            return []
        parsed_query = tantivy_index.parse_query(
            f"({safe_query}) AND domain:{domain}", ["text"]
        )
    else:
        parsed_query = tantivy_index.parse_query(safe_query, ["text"])
        if OPT_IN_DOMAINS:
            fetch_n = top_n * 3
    results = searcher.search(parsed_query, limit=fetch_n).hits
    out = []
    for score, doc_addr in results:
        doc = searcher.doc(doc_addr)
        doc_id_chunk = _safe_doc_first(doc, "id")
        doc_domain = _safe_doc_first(doc, "domain")
        if not domain and doc_domain in OPT_IN_DOMAINS:
            continue
        source_val = _safe_doc_first(doc, "source")
        out.append(
            {
                "id": doc_id_chunk,
                "doc_id": _safe_doc_first(doc, "doc_id") or source_val,
                "text": _safe_doc_first(doc, "text"),
                "source": source_val,
                "source_class": _safe_doc_first(doc, "source_class"),
                "filename": _safe_doc_first(doc, "filename"),
                "domain": doc_domain,
                "url": _safe_doc_first(doc, "url"),
                "bm25_score": float(score),
            }
        )
        if len(out) >= top_n:
            break
    return out


@lru_cache(maxsize=1024)
def _cached_query_embedding(model_key: int, key: str) -> tuple[float, ...]:
    """Memoize query embeddings keyed by active embedder identity + exact query text."""
    del model_key
    with _GPU_LOCK:
        return tuple(embed_model.encode([key], batch_size=1).tolist()[0])


def _apply_diversification(
    candidates: list[dict], max_per_source: int, top_k: int
) -> list[dict]:
    seen_sources: dict[str, int] = {}
    selected: list[dict] = []
    for c in candidates:
        src = _diversification_key(c)
        seen_sources[src] = seen_sources.get(src, 0) + 1
        if seen_sources[src] <= max_per_source:
            selected.append(c)
        if len(selected) >= top_k:
            break
    return selected


def _execute_search(req: SearchRequest, include_stages: bool = False) -> dict[str, Any]:
    """Single canonical hybrid retrieval pipeline used by both /search and /search/debug.

    Guarantees that /search and /search/debug execute identical query embedding,
    dense/BM25 retrieval, deduplicated RRF fusion, candidate capping, reranking,
    domain boosting, recency/vendor weighting, and doc_id source diversification.
    """
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")
    if req.rerank and reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model is not loaded")

    start = time.time()
    timings: dict[str, float] = {}
    stages: dict[str, list[dict]] = {}
    RAG_TOOL_CALLS.labels(**LABEL_EMBED).inc()

    # 1. Embed instructed query
    t0 = time.time()
    instructed_query = QUERY_INSTRUCTION + req.query
    query_vec = list(_cached_query_embedding(id(embed_model), instructed_query))
    timings["embed"] = round(time.time() - t0, 3)
    query_tokens = max(1, int(len(req.query.split()) / 0.75))
    RAG_TOKENS_INPUT.labels(**LABEL_EMBED).inc(query_tokens)
    RAG_REQUESTS.labels(**LABEL_EMBED).inc()

    # 2. Dense search
    t0 = time.time()
    dense_results = _dense_search(query_vec, req.dense_top_n, req.domain)
    timings["dense"] = round(time.time() - t0, 3)
    if include_stages:
        stages["dense"] = [{**r, "rank": i + 1} for i, r in enumerate(dense_results)]

    # 3. BM25 search
    t0 = time.time()
    bm25_results = _bm25_search(req.query, req.bm25_top_n, req.domain)
    timings["bm25"] = round(time.time() - t0, 3)
    if include_stages:
        stages["bm25"] = [{**r, "rank": i + 1} for i, r in enumerate(bm25_results)]

    # 4. Deduplicated RRF fusion
    t0 = time.time()
    dense_ids = [r["id"] for r in dense_results]
    bm25_ids = [r["id"] for r in bm25_results]
    merged_ids, rrf_scores = rrf_merge_with_scores(dense_ids, bm25_ids, k=req.rrf_k)

    # Build lookup by chunk id, preserving first (highest-ranked) payload per leg.
    all_results: dict[str, dict] = {}
    for r in dense_results:
        if r["id"] not in all_results:
            all_results[r["id"]] = dict(r)
    for r in bm25_results:
        if r["id"] not in all_results:
            all_results[r["id"]] = dict(r)
        elif "bm25_score" not in all_results[r["id"]]:
            all_results[r["id"]]["bm25_score"] = r.get("bm25_score", 0.0)
            if not all_results[r["id"]].get("doc_id") and r.get("doc_id"):
                all_results[r["id"]]["doc_id"] = r["doc_id"]
            if not all_results[r["id"]].get("source_class") and r.get("source_class"):
                all_results[r["id"]]["source_class"] = r["source_class"]

    for doc_id, item in all_results.items():
        item["rrf_score"] = round(rrf_scores.get(doc_id, 0.0), 6)

    if include_stages:
        stages["rrf_merged"] = [
            {**all_results[doc_id], "rank": i + 1}
            for i, doc_id in enumerate(merged_ids)
            if doc_id in all_results
        ]

    candidate_cap = max(RERANK_CANDIDATE_CAP, req.top_k)
    rerank_candidates: list[dict] = []
    for doc_id in merged_ids:
        if doc_id in all_results:
            rerank_candidates.append(dict(all_results[doc_id]))
        if len(rerank_candidates) >= candidate_cap:
            break
    timings["rrf"] = round(time.time() - t0, 3)

    # 5a. Fast path (rerank=False): RRF + vendor tiering + meaningful recency tie-break + doc_id cap
    if not req.rerank:
        def _fast_key(ic: tuple[int, dict]) -> tuple[int, float, float, int]:
            idx, c = ic
            tier = 1 if _is_vendor_doc(c) else 0
            eff_rrf = float(c.get("rrf_score", 0.0))
            if req.domain_boost:
                eff_rrf *= float(req.domain_boost.get(c.get("domain", ""), 1.0))
            recency = _recency_multiplier(c)
            # Round RRF score to 6 decimals so real RRF ties (e.g. rank-1 dense vs rank-1 BM25)
            # are genuinely broken by recency before falling back to ordinal idx.
            return (tier, -round(eff_rrf, 6), -recency, idx)

        ordered = [c for _, c in sorted(enumerate(rerank_candidates), key=_fast_key)]
        for c in ordered:
            c.setdefault("rerank_score", 0.0)
        final = _apply_diversification(ordered, req.max_per_source, req.top_k)
        timings["rerank"] = 0.0
        timings["total"] = round(time.time() - start, 3)

        counts = {
            "dense": len(dense_results),
            "bm25": len(bm25_results),
            "rrf_merged": len(merged_ids),
            "reranked": 0,
            "returned": len(final),
        }
        payload: dict[str, Any] = {
            "query": req.query,
            "results": final,
            "counts": counts,
            "timings": timings,
            "mode": "fast",
        }
        if include_stages:
            stages["reranked"] = []
            stages["final"] = [{**r, "rank": i + 1} for i, r in enumerate(final)]
            payload["domain"] = req.domain
            payload["stages"] = stages
        return payload

    # 5b. Full cross-encoder rerank path
    t0 = time.time()
    candidate_texts = [c["text"] for c in rerank_candidates]
    rerank_scores = _rerank_docs(req.query, candidate_texts)
    for i, score in enumerate(rerank_scores):
        rerank_candidates[i]["rerank_score"] = round(score, 4)

    if req.domain_boost:
        for c in rerank_candidates:
            boost = float(req.domain_boost.get(c.get("domain", ""), 1.0))
            c["rerank_score"] = round(c["rerank_score"] * boost, 4)

    for c in rerank_candidates:
        mult = _recency_multiplier(c)
        if _is_vendor_doc(c):
            mult *= DEPRIORITIZE_FACTOR
        c["rerank_score"] = round(c["rerank_score"] * mult, 4)

    rerank_candidates.sort(
        key=lambda x: (x["rerank_score"], x.get("rrf_score", 0.0)), reverse=True
    )
    final = _apply_diversification(rerank_candidates, req.max_per_source, req.top_k)

    timings["rerank"] = round(time.time() - t0, 3)
    timings["total"] = round(time.time() - start, 3)

    counts = {
        "dense": len(dense_results),
        "bm25": len(bm25_results),
        "rrf_merged": len(merged_ids),
        "reranked": len(rerank_candidates),
        "returned": len(final),
    }
    payload = {
        "query": req.query,
        "results": final,
        "counts": counts,
        "timings": timings,
    }
    if include_stages:
        stages["reranked"] = [
            {**r, "rank": i + 1} for i, r in enumerate(rerank_candidates)
        ]
        stages["final"] = [{**r, "rank": i + 1} for i, r in enumerate(final)]
        payload["domain"] = req.domain
        payload["stages"] = stages
    return payload


@app.post("/search", dependencies=[Depends(_require_api_auth)])
def search(req: SearchRequest):
    """Full hybrid search: dense + BM25 -> RRF merge -> rerank."""
    return _execute_search(req, include_stages=False)


@app.post("/search/debug", dependencies=[Depends(_require_api_auth)])
def search_debug(req: SearchRequest):
    """Full hybrid search returning the exact /search ranking plus stage diagnostics."""
    return _execute_search(req, include_stages=True)


# ---------------------------------------------------------------------------
# /ingest and /admin/ingest/reset — background ingestion triggers
# ---------------------------------------------------------------------------

try:
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
except (ValueError, OSError):
    pass

INGEST_SCRIPT = os.environ.get("INGEST_SCRIPT", os.path.join(RAG_ROOT, "scripts", "ingest.py"))
INGEST_PYTHON = os.environ.get("INGEST_PYTHON", sys.executable)
INGEST_LOCKFILE = os.environ.get("INGEST_LOCKFILE", "/tmp/alfred-rag-ingest.lock")
INGEST_LOG = os.environ.get("INGEST_LOG", os.path.join(RAG_ROOT, "ingest.log"))


class IngestRequest(BaseModel):
    incremental: bool = True
    reset: bool = False


def _resolve_ingest_script() -> str:
    """Resolve INGEST_SCRIPT relative to RAG_ROOT when not an absolute path."""
    if os.path.isabs(INGEST_SCRIPT):
        return INGEST_SCRIPT
    return os.path.abspath(os.path.join(RAG_ROOT, INGEST_SCRIPT))


def _ingest_already_running() -> int | None:
    """Return the pid of a running ingest.py, or None (reaping zombies on Linux)."""
    if not os.path.exists(INGEST_LOCKFILE):
        return None
    try:
        with open(INGEST_LOCKFILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        if os.path.isdir("/proc"):
            try:
                with open(f"/proc/{pid}/status") as sf:
                    for line in sf:
                        if line.startswith("State:"):
                            if line.split()[1].startswith("Z"):
                                raise ProcessLookupError(f"pid {pid} is zombie")
                            break
            except FileNotFoundError:
                raise ProcessLookupError(f"pid {pid} no /proc entry")
        return pid
    except (ValueError, ProcessLookupError, PermissionError, FileNotFoundError):
        try:
            os.unlink(INGEST_LOCKFILE)
        except FileNotFoundError:
            pass
        return None


def _spawn_ingest(*, reset: bool) -> JSONResponse | dict[str, Any]:
    script_path = _resolve_ingest_script()
    if not os.path.isfile(script_path):
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": f"Ingest builder script not found: {script_path}",
            },
        )

    existing = _ingest_already_running()
    if existing is not None:
        return {
            "status": "already_running",
            "pid": existing,
            "log": INGEST_LOG,
        }

    args = [INGEST_PYTHON, script_path]
    if reset:
        args.append("--reset")

    with open(INGEST_LOG, "ab") as log_f:
        proc = subprocess.Popen(
            args,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=RAG_ROOT,
            start_new_session=True,
        )
    try:
        with open(INGEST_LOCKFILE, "w") as f:
            f.write(str(proc.pid))
    except OSError:
        pass

    return {
        "status": "started",
        "pid": proc.pid,
        "mode": "reset" if reset else "incremental",
        "script": script_path,
        "log": INGEST_LOG,
    }


@app.post("/ingest", dependencies=[Depends(_require_api_auth)])
def ingest(req: IngestRequest, request: Request):
    """Trigger incremental (default) or full-reset ingestion in the background.

    When reset=True, administrative authorization (_require_admin_auth) is enforced.
    """
    if req.reset:
        _require_admin_auth(request)
    return _spawn_ingest(reset=req.reset)


@app.post("/admin/ingest/reset", dependencies=[Depends(_require_admin_auth)])
def admin_ingest_reset():
    """Dedicated administrative endpoint to trigger a full index reset."""
    return _spawn_ingest(reset=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Alfred RAG FastAPI server.")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", HOST),
        help="Bind host (default: 127.0.0.1; non-loopback requires --allow-external-bind + RAG_API_KEY)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", str(PORT))),
        help="Bind port (default: 9000 or PORT env var)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", DEVICE),
        help="Torch device: auto, cuda, mps, or cpu",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("RAG_BACKEND", RAG_BACKEND),
        choices=["hf", "offline"],
        help="Model backend: 'hf' (Qwen3) or 'offline' (deterministic CPU demo backend)",
    )
    parser.add_argument(
        "--allow-external-bind",
        action="store_true",
        default=ALLOW_EXTERNAL_BIND,
        help="Explicitly allow binding a non-loopback host (requires RAG_API_KEY)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global DEVICE, RAG_BACKEND
    args = parse_args(argv)
    DEVICE = args.device
    RAG_BACKEND = args.backend
    os.environ["DEVICE"] = args.device
    os.environ["RAG_BACKEND"] = args.backend
    if args.backend == "hf":
        resolve_device(args.device)  # fail fast on invalid DEVICE values
    bind_host, bind_port = resolve_bind_config(
        host=args.host,
        port=args.port,
        allow_external_bind=args.allow_external_bind,
        api_key=RAG_API_KEY,
    )
    import uvicorn

    uvicorn.run(app, host=bind_host, port=bind_port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
