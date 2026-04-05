# alfred-rag

A hybrid RAG stack: dense embeddings (LanceDB) plus BM25 (Tantivy) with Reciprocal Rank Fusion, then cross-encoder reranking with a Qwen3-Reranker-8B. Plus a sentence-transformers fine-tuning recipe for adapting Qwen3-Embedding-8B to your own corpus.

Built originally for a personal knowledge-base RAG running on a DGX Spark, serving a multi-domain corpus of technical documentation, personal notes, and reference material.

## Components

| File | Purpose |
|---|---|
| `rag_server.py` | FastAPI server: `/embed`, `/rerank`, `/search` (hybrid + RRF + rerank), `/health`, `/metrics` |
| `rag_eval.py` | Eval harness: runs a list of queries, scores keyword hits, writes per-query JSON + summary CSV |
| `finetune_embedding.py` | Fine-tune Qwen3-Embedding-8B (or any sentence-transformers base) on query/positive pairs with MultipleNegativesRankingLoss |
| `eval_queries.example.json` | Schema example for eval queries |

## Architecture

```
query
  |
  |----> embed (dense) ------+
  |                           |
  +----> BM25 (tantivy) ------+
                              |
                              v
                       RRF fusion (k=60)
                              |
                              v
                        top N candidates
                              |
                              v
                 Qwen3-Reranker-8B cross-encoder
                              |
                              v
                         final top_k
```

- **Dense**: LanceDB + Qwen3-Embedding-8B (4096-d), optionally fine-tuned on your corpus
- **Sparse**: Tantivy BM25 with English stemming
- **Fusion**: RRF with configurable `k` (default 60)
- **Reranking**: Qwen3-Reranker-8B scoring yes/no logit on (query, doc) pairs
- **Optional**: per-domain filtering, max-per-source diversification, domain boosts

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Index your corpus

You'll need to build `indexes/lancedb/` (vector) and `indexes/tantivy/` (BM25) from your documents. This repo doesn't ship an ingest script because corpus-prep is inherently bespoke. At minimum each document chunk should have: `id`, `text`, `domain`, `filename`, `url`, `source`, `format`, `chunk_index`, `vector`.

### 3. Serve

```bash
# With env-configured paths
export EMBED_MODEL_PATH=Qwen/Qwen3-Embedding-8B
export RERANKER_MODEL_PATH=Qwen/Qwen3-Reranker-8B
export LANCEDB_PATH=./indexes/lancedb
export TANTIVY_PATH=./indexes/tantivy
export LANCE_TABLE=corpus
export PORT=9000
export DEVICE=cuda  # or "mps" / "cpu"

python rag_server.py
# or: uvicorn rag_server:app --host 0.0.0.0 --port 9000
```

### 4. Query

```bash
curl -s http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does reciprocal rank fusion work?", "top_k": 5}' | jq .
```

### 5. Run eval

Copy `eval_queries.example.json` to `eval_queries.json` and author your own queries + expected keywords:

```bash
python rag_eval.py --server http://localhost:9000 --queries eval_queries.json
```

Writes per-query JSON to `eval_results/` plus a summary CSV.

## Fine-tuning

If the base embedding model misses domain-specific vocabulary, fine-tune it on query/positive pairs:

```bash
# training_data/embedding_pairs.jsonl format:
# {"query": "...", "positive": "..."}
# {"query": "...", "positive": "..."}

python finetune_embedding.py \
  --base-model Qwen/Qwen3-Embedding-8B \
  --data-path ./training_data/embedding_pairs.jsonl \
  --output-dir ./models/qwen3-embedding-8b-ft \
  --epochs 3 --batch-size 4 --lr 3e-5
```

Uses `MultipleNegativesRankingLoss` (in-batch negatives) so you only need positive pairs. Evaluator is `InformationRetrievalEvaluator` over a held-out split. A 3600-pair corpus takes roughly 9 hours on a DGX Spark (H100-class) at batch size 4 for 3 epochs.

## Configuration

All paths and models are env-configurable. Server defaults:

| Variable | Default |
|---|---|
| `EMBED_MODEL_PATH` | `Qwen/Qwen3-Embedding-8B` |
| `RERANKER_MODEL_PATH` | `Qwen/Qwen3-Reranker-8B` |
| `LANCEDB_PATH` | `./indexes/lancedb` |
| `TANTIVY_PATH` | `./indexes/tantivy` |
| `LANCE_TABLE` | `corpus` |
| `DEVICE` | `cuda` if available else `cpu` |
| `PORT` | `9000` |

## Observability

The server exposes Prometheus metrics at `/metrics`:

- `llm_llm_tokens_input_total{endpoint, model, job}` — embedded + reranked input tokens
- `llm_llm_tokens_output_total{endpoint, model, job}` — reranker yes/no logit count
- `llm_llm_requests_total{endpoint, model, job}` — request count per endpoint
- `llm_llm_tool_calls_total` — search invocations

Pair with Grafana for a basic retrieval dashboard.

## Related projects

Part of a self-hosted LLM operations toolkit:

- [blockops-proxy](https://github.com/trevorgordon981/blockops-proxy) — tool-call-translating proxy that fronts this RAG server for OpenAI-compatible clients
- [llm-otel-proxy](https://github.com/trevorgordon981/llm-otel-proxy) — OTel metrics proxy that tracks tokens/cost/latency on this server's traffic
- [alfred-infra](https://github.com/trevorgordon981/alfred-infra) — monitoring + backup infrastructure for multi-machine local-LLM clusters
- [context-bench](https://github.com/trevorgordon981/context-bench) — benchmark the embedding/reranker throughput across context sizes

## License

MIT
