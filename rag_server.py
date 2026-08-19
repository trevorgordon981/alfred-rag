import torch
import json
import re
import time
import threading
from functools import lru_cache
import tantivy
import lancedb
import numpy as np
from fastapi import FastAPI, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Deployment root. Everything below resolves under it, so the server runs from any
# checkout without editing paths. Override any individual path with its own env var.
RAG_ROOT = os.environ.get("RAG_ROOT", os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Alfred RAG Server")

# --- Prometheus metrics ---
RAG_TOKENS_INPUT = Counter(
    "llm_llm_tokens_input_total",
    "Total input tokens processed",
    ["model", "endpoint", "job"]
)
RAG_TOKENS_OUTPUT = Counter(
    "llm_llm_tokens_output_total",
    "Total output tokens processed",
    ["model", "endpoint", "job"]
)
RAG_REQUESTS = Counter(
    "llm_llm_requests_total",
    "Total requests processed",
    ["model", "endpoint", "job"]
)
RAG_TOOL_CALLS = Counter(
    "llm_llm_tool_calls_total",
    "Total tool calls (search operations)",
    ["model", "endpoint", "job"]
)

LABEL_EMBED = {"model": "Qwen3-Embedding-8B", "endpoint": "/embed", "job": "rag-server"}
LABEL_RERANK = {"model": "Qwen3-Reranker-8B", "endpoint": "/rerank", "job": "rag-server"}

# --- Models and indexes (loaded at startup) ---
embed_model = None
reranker_tokenizer = None
reranker_model = None
lance_table = None
tantivy_index = None

# yes/no token ids for the reranker, resolved once at startup (L2).
RERANK_YES_ID = None
RERANK_NO_ID = None

# Single GPU: serialize every GPU op (embed + rerank forward) so concurrent
# requests on FastAPI's threadpool can't interleave on the device (H1) or
# thrash it. A plain module-level lock; held only around the compute section.
_GPU_LOCK = threading.Lock()

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", os.path.join(RAG_ROOT, "indexes", "lancedb"))
TANTIVY_PATH = os.environ.get("TANTIVY_PATH", os.path.join(RAG_ROOT, "indexes", "tantivy"))
LANCE_TABLE = "alfred_rag"

class EmbedRequest(BaseModel):
    texts: list[str]

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 5

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    dense_top_n: int = 30
    bm25_top_n: int = 30
    rrf_k: int = 60
    domain: str | None = None
    max_per_source: int = 2
    domain_boost: dict[str, float] | None = None
    rerank: bool = True  # set False for a fast dense+BM25+RRF path (skips the cross-encoder)

class HealthResponse(BaseModel):
    status: str
    models: list[str]
    indexes: dict
    uptime: float

START_TIME = time.time()


def build_tantivy_schema():
    """Rebuild the schema matching the existing index."""
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("id", stored=True)
    sb.add_text_field("text", stored=True, tokenizer_name="en_stem")
    sb.add_text_field("domain", stored=True)
    sb.add_text_field("filename", stored=True)
    sb.add_text_field("url", stored=True)
    sb.add_text_field("source", stored=True)
    return sb.build()


def rrf_merge(dense_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. Returns merged list of IDs ordered by RRF score."""
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def _refresh_indexes():
    """Pull the latest LanceDB fragments + Tantivy segments into the live
    readers. Runs on a timer (M4) instead of once per query — data added by a
    background /ingest becomes visible within one interval, and the hot query
    path (_dense_search/_bm25_search) no longer pays a checkout_latest() /
    reload() on every request."""
    try:
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


@app.on_event("startup")
def load_models():
    global embed_model, reranker_tokenizer, reranker_model, lance_table, tantivy_index
    global RERANK_YES_ID, RERANK_NO_ID

    print("Loading Qwen3-Embedding-8B...")
    embed_model = SentenceTransformer(
        os.environ.get("EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-8B"),
        device="cuda"
    )
    print("Embedding model loaded.")

    print("Loading Qwen3-Reranker-8B...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("RERANKER_MODEL_PATH", "Qwen/Qwen3-Reranker-8B"),
        trust_remote_code=True
    )
    reranker_model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("RERANKER_MODEL_PATH", "Qwen/Qwen3-Reranker-8B"),
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    reranker_model.eval()
    # Batched reranking left-pads; ensure a pad token exists.
    if reranker_tokenizer.pad_token_id is None:
        reranker_tokenizer.pad_token = reranker_tokenizer.eos_token
    # Left-pad ONCE at startup so the last position is the real final token for
    # every row. Never mutated per-request (H1): a concurrent call flipping this
    # back to right-padding mid-batch would make logits[:, -1, :] read a PAD.
    reranker_tokenizer.padding_side = "left"
    # Resolve yes/no token ids once (L2) instead of per rerank call.
    RERANK_YES_ID = reranker_tokenizer.convert_tokens_to_ids("yes")
    RERANK_NO_ID = reranker_tokenizer.convert_tokens_to_ids("no")
    print("Reranker model loaded.")

    print("Connecting LanceDB...")
    db = lancedb.connect(LANCEDB_PATH)
    lance_table = db.open_table(LANCE_TABLE)
    print(f"LanceDB: {lance_table.count_rows()} rows in '{LANCE_TABLE}'")

    print("Opening Tantivy index...")
    schema = build_tantivy_schema()
    tantivy_index = tantivy.Index(schema, path=TANTIVY_PATH)
    tantivy_index.reload()
    searcher = tantivy_index.searcher()
    print(f"Tantivy: {searcher.num_docs} docs")

    # M4: refresh index readers on a timer, not per query.
    _start_refresh_timer()


@app.get("/health")
def health() -> HealthResponse:
    if lance_table:
        lance_table.checkout_latest()
        lance_count = lance_table.count_rows()
    else:
        lance_count = 0
    tantivy_count = 0
    if tantivy_index:
        tantivy_index.reload()
        tantivy_count = tantivy_index.searcher().num_docs
    return HealthResponse(
        status="ok",
        models=["qwen3-embedding-8b", "qwen3-reranker-8b"],
        indexes={"lancedb": lance_count, "tantivy": tantivy_count},
        uptime=time.time() - START_TIME
    )


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/embed")
def embed(req: EmbedRequest):
    start = time.time()
    with _GPU_LOCK:
        embeddings = embed_model.encode(req.texts, batch_size=32).tolist()
    elapsed = time.time() - start
    # Count tokens: approximate via whitespace split / 0.75 (same as ingest.py)
    token_count = sum(int(len(t.split()) / 0.75) for t in req.texts)
    RAG_TOKENS_INPUT.labels(**LABEL_EMBED).inc(token_count)
    RAG_REQUESTS.labels(**LABEL_EMBED).inc()
    return {
        "embeddings": embeddings,
        "dim": len(embeddings[0]),
        "count": len(embeddings),
        "elapsed": round(elapsed, 3)
    }


@app.post("/rerank")
def rerank(req: RerankRequest):
    start = time.time()
    scores = _rerank_docs(req.query, req.documents)
    ranked = sorted(
        zip(req.documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:req.top_k]
    elapsed = time.time() - start
    RAG_REQUESTS.labels(**LABEL_RERANK).inc()
    return {
        "results": [{"document": doc, "score": round(s, 4)} for doc, s in ranked],
        "elapsed": round(elapsed, 3)
    }


@lru_cache(maxsize=4096)
def _clean_chunk_text(text: str) -> str:
    """Strip forum boilerplate and navigation from chunk text.

    M3: `import re` moved to module top (was re-imported per call), and results
    are memoized by exact text so repeated chunks across queries don't re-run
    the 9 regex subs. NOTE: ingest.py already applies the same cleaning before
    storing (scripts/ingest.py clean_chunk_text), so for freshly-ingested rows
    this is largely redundant — the right long-term home for this is ingest,
    leaving the query path to do none of it.
    """
    patterns = [
        r'(?i)^.*?Rennlist - Porsche Discussion Forums\s*',
        r'(?i)^LOG\s*IN\s*',
        r'(?i)^REGISTER\s*',
        r'(?i)^View First.*?\n',
        r'(?i)^View Public Profile.*?\n',
        r'(?i)^Find More Posts by.*?\n',
        r'(?i)^Share\s*Share Options\s*',
        r'(?i)^Quote:\s*Originally Posted by\s*\S+\s*',
        r'#\s*\d+\s*$',
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


RERANK_SYSTEM_PROMPT = (
    "Judge whether the document answers the user's query. "
    "A relevant document directly addresses the specific question asked, "
    "not just the general topic. Respond with 'yes' or 'no'."
)
RERANK_BATCH_SIZE = 16  # all typical candidate sets (<=15) fit in one batch


def _rerank_docs(query: str, documents: list[str]) -> list[float]:
    """Score documents against a query using the reranker.

    All candidates are scored in left-padded batched forward passes rather than
    one sequential pass per document. Left padding keeps the final real token at
    position -1 for every row, so logits[:, -1, :] reads the yes/no prediction
    for each document. Score parity with the old per-doc path is exact (same
    inputs, same logits) — only the batching changes.
    """
    if not documents:
        return []

    yes_id = RERANK_YES_ID
    no_id = RERANK_NO_ID

    # Build per-doc prompts.
    prompts = []
    for doc in documents:
        cleaned = _clean_chunk_text(doc)
        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"query": query, "document": cleaned},
                ensure_ascii=False
            )}
        ]
        prompts.append(reranker_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # padding_side is fixed to "left" once at startup (load_models), so the last
    # position is the real final token for every row — no per-call mutation (H1).
    # The GPU forward is serialized under _GPU_LOCK (H1/single-GPU) so concurrent
    # requests can't interleave on the device.
    scores: list[float] = []
    with _GPU_LOCK:
        for start in range(0, len(prompts), RERANK_BATCH_SIZE):
            batch = prompts[start:start + RERANK_BATCH_SIZE]
            inputs = reranker_tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=4096, padding=True
            ).to(reranker_model.device)

            RAG_TOKENS_INPUT.labels(**LABEL_RERANK).inc(int(inputs["input_ids"].numel()))
            RAG_TOKENS_OUTPUT.labels(**LABEL_RERANK).inc(len(batch))  # one yes/no per doc

            with torch.no_grad():
                # H2: logits_to_keep=1 computes logits for only the last position
                # (the yes/no slot), not the full batch×seq×vocab tensor (~2-3GB
                # fp16/call). Output is [batch, 1, vocab]; [:, -1, :] takes that
                # single kept position, so the yes_id/no_id indexing is unchanged.
                logits = reranker_model(**inputs, logits_to_keep=1).logits[:, -1, :]
                diff = logits[:, yes_id] - logits[:, no_id]
                scores.extend(torch.sigmoid(diff).tolist())
    return scores

_DOMAIN_RE = __import__("re").compile(r"^[A-Za-z0-9_-]+$")

# Domains excluded from default retrieval (still searchable via explicit
# domain="<name>"). Empty = everything participates in default search.
# News was here while it was a stale 7.6k-clipping pile; now that it's a fresh,
# relevant, rolling window, it's back in default retrieval (the reranker only
# floats a news chunk when it's actually on-topic). Re-add a domain here to
# gate it to opt-in again.
OPT_IN_DOMAINS = set()

# Bulk auto-scraped vendor reference docs. They stay FULLY in default retrieval
# (Trevor wants everything searchable, no opt-in), but in the rerank-free fast
# path (per-turn auto-prefetch) they sort AFTER everything else, so a personal/
# curated hit fills the injected context first and these only appear when nothing
# better matched. The full reranked tool path is unaffected. Trevor's own content
# is <5% of chunks, so without this the prefetch block skews to vendor docs.
# Path marker for bulk auto-scraped vendor docs (live under data/docs/<vendor>/).
# Robust to the leaf-subdir domain labels that defeated the old name list.
VENDOR_PATH_MARKER = "/data/docs/"
# Multiplier applied to a deprioritized chunk's fast-path rank weight (kept in
# retrieval, just sorted after personal/curated hits). Not dropped entirely.
DEPRIORITIZE_FACTOR = 0.6


def _is_vendor_doc(chunk: dict) -> bool:
    """True if a chunk is a bulk auto-scraped vendor reference doc.

    Matched by source PATH (contains data/docs/), not the leaf-subdir domain
    label. Trevor's own curated content lives elsewhere under data/<domain>/.
    """
    return VENDOR_PATH_MARKER in (chunk.get("source") or "")

# --- Recency weighting (P2) -------------------------------------------------
import datetime as _dt

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _recency_multiplier(chunk: dict) -> float:
    """Small decay multiplier from a date embedded in the source filename.

    Extracts the first YYYY-MM-DD found in the filename/source. Newer => higher
    multiplier (closer to 1.0); a doc ~10y old bottoms out at ~0.7. No parseable
    date => 1.0 (no penalty). Bounded so it only breaks near-ties, never
    overrides a clear relevance win.
    """
    name = (chunk.get("filename") or chunk.get("source") or "")
    m = _DATE_RE.search(name)
    if not m:
        return 1.0
    try:
        d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return 1.0
    age_days = (_dt.date.today() - d).days
    if age_days < 0:
        age_days = 0
    return 1.0 - min(age_days, 3650) / 3650 * 0.3



def _dense_search(query_vec: list[float], top_n: int, domain: str | None = None) -> list[dict]:
    """LanceDB vector search. Returns list of {id, text, score, ...metadata}."""
    # M4: table freshness is handled by the background refresh timer
    # (_refresh_indexes), not a checkout_latest() on every query.
    # C1 recall fix (2026-07-20): the IVF_PQ index is lossy — nprobes=20 alone
    # returns only ~0.47 recall@10 vs an exhaustive cosine scan (verified on the
    # live 38.9k-row table). refine_factor=20 re-ranks the top k*20 PQ candidates
    # by their exact (non-quantized) vectors, lifting recall@10 to ~0.91 and
    # recall@30 to ~0.94 for ~+3ms/query, while keeping the ~15x latency win over
    # the brute-force scan (14ms vs 210ms). Metric stays cosine (from the index).
    q = lance_table.search(query_vec).refine_factor(20)
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
        out.append({
            "id": r["id"],
            "text": r["text"],
            "source": r.get("source", ""),
            "filename": r.get("filename", ""),
            "domain": r.get("domain", ""),
            "url": r.get("url", ""),
            "format": r.get("format", ""),
            "chunk_index": r.get("chunk_index", 0),
            "dense_score": float(r.get("_distance", 0.0)),
        })
    return out


def _bm25_search(query: str, top_n: int, domain: str | None = None) -> list[dict]:
    """Tantivy BM25 search. Returns list of {id, text, score, ...metadata}."""
    # M4: reader freshness is handled by the background refresh timer
    # (_refresh_indexes), not a reload() on every query.
    searcher = tantivy_index.searcher()
    safe_query = query.replace("'", " ").replace('"', ' ').replace(':', ' ').replace('(', ' ').replace(')', ' ').strip()
    if not safe_query:
        return []
    fetch_n = top_n
    if domain:
        if not _DOMAIN_RE.match(domain):
            return []
        parsed_query = tantivy_index.parse_query(f"({safe_query}) AND domain:{domain}", ["text"])
    else:
        parsed_query = tantivy_index.parse_query(safe_query, ["text"])
        # Over-fetch so post-filtering opt-in domains still yields ~top_n hits.
        if OPT_IN_DOMAINS:
            fetch_n = top_n * 3
    results = searcher.search(parsed_query, limit=fetch_n).hits
    out = []
    for score, doc_addr in results:
        doc = searcher.doc(doc_addr)
        doc_id = doc.get_first("id")
        doc_domain = doc.get_first("domain")
        if not domain and doc_domain in OPT_IN_DOMAINS:
            continue
        out.append({
            "id": doc_id,
            "text": doc.get_first("text"),
            "source": doc.get_first("source") or "",
            "filename": doc.get_first("filename") or "",
            "domain": doc_domain or "",
            "url": doc.get_first("url") or "",
            "bm25_score": float(score),
        })
        if len(out) >= top_n:
            break
    return out


@lru_cache(maxsize=1024)
def _cached_query_embedding(key: str) -> tuple:
    """L1: memoize query embeddings keyed by the exact instruction+query string,
    so a repeated query (common with per-turn auto-prefetch) skips the GPU embed
    entirely. Returns a tuple (hashable) that callers list()."""
    with _GPU_LOCK:
        return tuple(embed_model.encode([key], batch_size=1)[0].tolist())


@app.post("/search")
def search(req: SearchRequest):
    """Full hybrid search: dense + BM25 -> RRF merge -> rerank."""
    start = time.time()
    timings = {}
    RAG_TOOL_CALLS.labels(**LABEL_EMBED).inc()

    # 1. Embed query
    t0 = time.time()
    instruction = "Instruct: Given a question from Trevor, retrieve the most relevant passages from his personal knowledge base (Porsche 971 Panamera, finance & options, trading journal, Stoic/Buddhist philosophy, medical, daily journal, homelab).\nQuery: "
    query_vec = list(_cached_query_embedding(instruction + req.query))
    timings["embed"] = round(time.time() - t0, 3)
    query_tokens = int(len(req.query.split()) / 0.75)
    RAG_TOKENS_INPUT.labels(**LABEL_EMBED).inc(query_tokens)
    RAG_REQUESTS.labels(**LABEL_EMBED).inc()

    # 2. Dense search
    t0 = time.time()
    dense_results = _dense_search(query_vec, req.dense_top_n, req.domain)
    timings["dense"] = round(time.time() - t0, 3)

    # 3. BM25 search
    t0 = time.time()
    bm25_results = _bm25_search(req.query, req.bm25_top_n, req.domain)
    timings["bm25"] = round(time.time() - t0, 3)

    # 4. RRF merge
    t0 = time.time()
    dense_ids = [r["id"] for r in dense_results]
    bm25_ids = [r["id"] for r in bm25_results]
    merged_ids = rrf_merge(dense_ids, bm25_ids, k=req.rrf_k)

    # Build lookup of all result metadata by id
    all_results = {}
    for r in dense_results:
        all_results[r["id"]] = r
    for r in bm25_results:
        if r["id"] not in all_results:
            all_results[r["id"]] = r
        else:
            all_results[r["id"]]["bm25_score"] = r.get("bm25_score", 0.0)

    # Take top candidates for reranking (union may be larger than either top_n)
    rerank_candidates = []
    for doc_id in merged_ids:
        if doc_id in all_results:
            rerank_candidates.append(all_results[doc_id])
        if len(rerank_candidates) >= 10:  # 15->10: ~33% fewer cross-encoder rows, eval-neutral
            break
    timings["rrf"] = round(time.time() - t0, 3)

    # Fast path: skip the cross-encoder and return RRF-ranked (dense+BM25) results.
    # Used by per-turn auto-prefetch, where sub-second latency under load matters
    # more than the last bit of rerank precision. The explicit alfred_memory_search
    # tool keeps the full reranked path (rerank=True, the default).
    if not req.rerank:
        # Stable-sort so bulk vendor-doc domains sink below personal/curated hits
        # (preserves RRF order within each tier). Everything still participates;
        # this only rebalances which chunks fill the limited prefetch slots.
        # P3: sink bulk vendor docs (matched by source PATH, not domain label)
        # below personal/curated hits; preserve RRF order within each tier.
        # P2: apply a small recency tie-breaker among same-tier chunks.
        def _fast_key(ic):
            idx, c = ic
            tier = 1 if _is_vendor_doc(c) else 0
            # Lower recency_rank => newer => sorts earlier. Scaled tiny so it
            # only breaks ties, never reorders across the RRF order materially.
            recency_rank = _recency_multiplier(c)
            return (tier, idx, -recency_rank)

        ordered = sorted(enumerate(rerank_candidates), key=_fast_key)
        seen_sources = {}
        fast = []
        for _, c in ordered:
            c.setdefault("rerank_score", 0.0)  # keep response shape stable
            src = c.get("filename", "")
            seen_sources[src] = seen_sources.get(src, 0) + 1
            if seen_sources[src] <= req.max_per_source:
                fast.append(c)
            if len(fast) >= req.top_k:
                break
        timings["rerank"] = 0.0
        timings["total"] = round(time.time() - start, 3)
        return {
            "query": req.query,
            "results": fast,
            "counts": {
                "dense": len(dense_results), "bm25": len(bm25_results),
                "rrf_merged": len(merged_ids), "reranked": 0, "returned": len(fast),
            },
            "timings": timings,
            "mode": "fast",
        }

    # 5. Rerank
    t0 = time.time()
    candidate_texts = [c["text"] for c in rerank_candidates]
    rerank_scores = _rerank_docs(req.query, candidate_texts)
    for i, score in enumerate(rerank_scores):
        rerank_candidates[i]["rerank_score"] = round(score, 4)

    # Apply domain boost if provided
    if req.domain_boost:
        for c in rerank_candidates:
            boost = req.domain_boost.get(c.get("domain", ""), 1.0)
            c["rerank_score"] = round(c["rerank_score"] * boost, 4)

    # P2: recency weighting — newer docs get a small edge on otherwise-equal
    # relevance. P3: bulk vendor docs get a deprioritization multiplier so
    # Trevor's own content surfaces first when scores are comparable.
    for c in rerank_candidates:
        mult = _recency_multiplier(c)
        if _is_vendor_doc(c):
            mult *= DEPRIORITIZE_FACTOR
        c["rerank_score"] = round(c["rerank_score"] * mult, 4)

    # Sort by rerank score
    rerank_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Deduplicate: max N chunks per source file
    seen_sources = {}
    deduped = []
    for c in rerank_candidates:
        src = c.get("filename", "")
        seen_sources[src] = seen_sources.get(src, 0) + 1
        if seen_sources[src] <= req.max_per_source:
            deduped.append(c)
    ranked = deduped[:req.top_k]

    timings["rerank"] = round(time.time() - t0, 3)
    timings["total"] = round(time.time() - start, 3)

    return {
        "query": req.query,
        "results": ranked,
        "counts": {
            "dense": len(dense_results),
            "bm25": len(bm25_results),
            "rrf_merged": len(merged_ids),
            "reranked": len(rerank_candidates),
            "returned": len(ranked),
        },
        "timings": timings,
    }




# ---------------------------------------------------------------------------
# ADD THIS TO rag_server.py — paste above the `if __name__` block
# ---------------------------------------------------------------------------

@app.post("/search/debug")
def search_debug(req: SearchRequest):
    """Full hybrid search with intermediate results at every stage."""
    start = time.time()
    timings = {}
    stages = {}

    # 1. Embed query
    t0 = time.time()
    with _GPU_LOCK:
        query_vec = embed_model.encode([req.query], batch_size=1)[0].tolist()
    timings["embed"] = round(time.time() - t0, 3)

    # 2. Dense search
    t0 = time.time()
    dense_results = _dense_search(query_vec, req.dense_top_n, req.domain)
    timings["dense"] = round(time.time() - t0, 3)
    stages["dense"] = [
        {**r, "rank": i + 1} for i, r in enumerate(dense_results)
    ]

    # 3. BM25 search
    t0 = time.time()
    bm25_results = _bm25_search(req.query, req.bm25_top_n, req.domain)
    timings["bm25"] = round(time.time() - t0, 3)
    stages["bm25"] = [
        {**r, "rank": i + 1} for i, r in enumerate(bm25_results)
    ]

    # 4. RRF merge
    t0 = time.time()
    dense_ids = [r["id"] for r in dense_results]
    bm25_ids = [r["id"] for r in bm25_results]
    merged_ids = rrf_merge(dense_ids, bm25_ids, k=req.rrf_k)

    all_results = {}
    for r in dense_results:
        all_results[r["id"]] = r
    for r in bm25_results:
        if r["id"] not in all_results:
            all_results[r["id"]] = r
        else:
            all_results[r["id"]]["bm25_score"] = r.get("bm25_score", 0.0)

    rerank_candidates = []
    for doc_id in merged_ids:
        if doc_id in all_results:
            rerank_candidates.append(all_results[doc_id])
        if len(rerank_candidates) >= 15:
            break
    timings["rrf"] = round(time.time() - t0, 3)

    # Compute RRF scores for logging
    rrf_scores = {}
    for i, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (req.rrf_k + i + 1)
    for i, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (req.rrf_k + i + 1)

    stages["rrf_merged"] = [
        {**all_results[doc_id], "rrf_score": round(rrf_scores.get(doc_id, 0.0), 6), "rank": i + 1}
        for i, doc_id in enumerate(merged_ids)
        if doc_id in all_results
    ]

    # 5. Rerank
    t0 = time.time()
    candidate_texts = [c["text"] for c in rerank_candidates]
    rerank_scores = _rerank_docs(req.query, candidate_texts)
    for i, score in enumerate(rerank_scores):
        rerank_candidates[i]["rerank_score"] = round(score, 4)

    reranked_all = sorted(rerank_candidates, key=lambda x: x["rerank_score"], reverse=True)
    timings["rerank"] = round(time.time() - t0, 3)

    stages["reranked"] = [
        {**r, "rank": i + 1} for i, r in enumerate(reranked_all)
    ]

    final = reranked_all[:req.top_k]
    stages["final"] = [
        {**r, "rank": i + 1} for i, r in enumerate(final)
    ]

    timings["total"] = round(time.time() - start, 3)

    return {
        "query": req.query,
        "domain": req.domain,
        "stages": stages,
        "timings": timings,
        "counts": {
            "dense": len(dense_results),
            "bm25": len(bm25_results),
            "rrf_merged": len(merged_ids),
            "reranked": len(rerank_candidates),
            "returned": len(final),
        },
    }

# ---------------------------------------------------------------------------
# /ingest — fire-and-forget incremental ingestion trigger
# ---------------------------------------------------------------------------

import subprocess
import os
import sys
import signal

# Auto-reap zombie subprocesses from the /ingest endpoint's fire-and-forget
# Popen calls. Without this, completed ingest.py processes linger as
# defunct and keep the lockfile pid check returning True.
try:
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
except (ValueError, OSError):
    pass

INGEST_SCRIPT = os.environ.get("INGEST_SCRIPT", os.path.join(RAG_ROOT, "scripts", "ingest.py"))
INGEST_PYTHON = os.environ.get("INGEST_PYTHON", sys.executable)
INGEST_LOCKFILE = "/tmp/alfred-rag-ingest.lock"
INGEST_LOG = os.environ.get("INGEST_LOG", os.path.join(RAG_ROOT, "ingest.log"))


class IngestRequest(BaseModel):
    incremental: bool = True
    reset: bool = False


def _ingest_already_running() -> int | None:
    """Return the pid of a running ingest.py, or None.

    Detects and cleans up zombies: os.kill(pid, 0) returns success for
    defunct processes, so we also check /proc/<pid>/status State field.
    """
    if not os.path.exists(INGEST_LOCKFILE):
        return None
    try:
        with open(INGEST_LOCKFILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        # Zombie check: /proc/<pid>/status State field starts with 'Z' for
        # defunct processes, which os.kill still accepts as valid.
        try:
            with open(f"/proc/{pid}/status") as sf:
                for line in sf:
                    if line.startswith("State:"):
                        state_char = line.split()[1]
                        if state_char.startswith("Z"):
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


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Trigger incremental (default) or full-reset ingestion in the background."""
    existing = _ingest_already_running()
    if existing is not None:
        return {
            "status": "already_running",
            "pid": existing,
            "log": INGEST_LOG,
        }

    args = [INGEST_PYTHON, INGEST_SCRIPT]
    if req.reset:
        args.append("--reset")

    log_f = open(INGEST_LOG, "ab")
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
        "mode": "reset" if req.reset else "incremental",
        "log": INGEST_LOG,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
