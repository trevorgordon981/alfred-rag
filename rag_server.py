"""
Hybrid RAG server: dense embeddings (LanceDB) + BM25 (Tantivy) with RRF fusion
and a Qwen3-Reranker-8B cross-encoder. FastAPI + Prometheus metrics.

All paths/models are env-configurable (see README for defaults):
  EMBED_MODEL_PATH    (default: Qwen/Qwen3-Embedding-8B)
  RERANKER_MODEL_PATH (default: Qwen/Qwen3-Reranker-8B)
  LANCEDB_PATH        (default: ./indexes/lancedb)
  TANTIVY_PATH        (default: ./indexes/tantivy)
  LANCE_TABLE         (default: corpus)
  DEVICE              (default: cuda if available else cpu)
  PORT                (default: 9000)
"""

import json
import os
import time

import torch
import tantivy
import lancedb
from fastapi import FastAPI, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Hybrid RAG Server")

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

LABEL_EMBED = {"model": "embed", "endpoint": "/embed", "job": "rag-server"}
LABEL_RERANK = {"model": "rerank", "endpoint": "/rerank", "job": "rag-server"}

# --- Models and indexes (loaded at startup) ---
embed_model = None
reranker_tokenizer = None
reranker_model = None
lance_table = None
tantivy_index = None

EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-8B")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "Qwen/Qwen3-Reranker-8B")
LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "./indexes/lancedb")
TANTIVY_PATH = os.environ.get("TANTIVY_PATH", "./indexes/tantivy")
LANCE_TABLE = os.environ.get("LANCE_TABLE", "corpus")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

class EmbedRequest(BaseModel):
    texts: list[str]

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 5

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    dense_top_n: int = 20
    bm25_top_n: int = 20
    rrf_k: int = 60
    domain: str | None = None
    max_per_source: int = 2
    domain_boost: dict[str, float] | None = None

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


@app.on_event("startup")
def load_models():
    global embed_model, reranker_tokenizer, reranker_model, lance_table, tantivy_index

    print(f"Loading embedding model from {EMBED_MODEL_PATH}...")
    embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
    print("Embedding model loaded.")

    print(f"Loading reranker from {RERANKER_MODEL_PATH}...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_PATH, trust_remote_code=True
    )
    reranker_model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    reranker_model.eval()
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


@app.get("/health")
def health() -> HealthResponse:
    lance_count = lance_table.count_rows() if lance_table else 0
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


def _clean_chunk_text(text: str) -> str:
    """Strip common forum/site boilerplate from chunk text before reranking.

    Extend CHUNK_CLEAN_PATTERNS for your own corpus. The examples here cover
    generic forum navigation text. Running the reranker on boilerplate wastes
    tokens and can confuse the cross-encoder when chunks differ only in the
    boilerplate prefix.
    """
    import re
    patterns = [
        r'(?i)^log\s*in\s*$',
        r'(?i)^register\s*$',
        r'(?i)^view public profile.*?\n',
        r'(?i)^find more posts by.*?\n',
        r'(?i)^share\s*share options\s*$',
        r'(?i)^quote:\s*originally posted by\s*\S+\s*',
        r'^#\s*\d+\s*$',  # post counters
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def _rerank_docs(query: str, documents: list[str]) -> list[float]:
    """Score a list of documents against a query using the reranker."""
    SYSTEM_PROMPT = (
        "Judge whether the document answers the user's query. "
        "A relevant document directly addresses the specific question asked, "
        "not just the general topic. Respond with 'yes' or 'no'."
    )
    scores = []
    for doc in documents:
        cleaned = _clean_chunk_text(doc)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"query": query, "document": cleaned},
                ensure_ascii=False
            )}
        ]
        input_text = reranker_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = reranker_tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(reranker_model.device)

        input_tokens = inputs["input_ids"].shape[1]
        RAG_TOKENS_INPUT.labels(**LABEL_RERANK).inc(input_tokens)
        RAG_TOKENS_OUTPUT.labels(**LABEL_RERANK).inc(1)  # single yes/no logit

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits[0, -1, :]
            yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
            no_id = reranker_tokenizer.convert_tokens_to_ids("no")
            yes_score = logits[yes_id].item()
            no_score = logits[no_id].item()
            score = torch.sigmoid(torch.tensor(yes_score - no_score)).item()
            scores.append(score)
    return scores

def _dense_search(query_vec: list[float], top_n: int, domain: str | None = None) -> list[dict]:
    """LanceDB vector search. Returns list of {id, text, score, ...metadata}."""
    results = lance_table.search(query_vec).limit(top_n).to_list()
    if domain:
        results = [r for r in results if r.get("domain") == domain]
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
    tantivy_index.reload()
    searcher = tantivy_index.searcher()
    safe_query = query.replace("'", " ").replace('"', ' ').replace(':', ' ').replace('(', ' ').replace(')', ' ')
    parsed_query = tantivy_index.parse_query(safe_query, ["text"])
    results = searcher.search(parsed_query, limit=top_n).hits
    out = []
    for score, doc_addr in results:
        doc = searcher.doc(doc_addr)
        doc_id = doc.get_first("id")
        doc_domain = doc.get_first("domain")
        if domain and doc_domain != domain:
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
    return out


@app.post("/search")
def search(req: SearchRequest):
    """Full hybrid search: dense + BM25 -> RRF merge -> rerank."""
    start = time.time()
    timings = {}
    RAG_TOOL_CALLS.labels(**LABEL_EMBED).inc()

    # 1. Embed query
    t0 = time.time()
    query_vec = embed_model.encode([req.query], batch_size=1)[0].tolist()
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
        if len(rerank_candidates) >= 15:
            break
    timings["rrf"] = round(time.time() - t0, 3)

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
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
