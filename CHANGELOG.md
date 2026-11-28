# Changelog

All notable changes to `alfred-rag` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-27

### Added

- **Dual-store hybrid retrieval & reranking (`rag_server.py`, `rag_common.py`)**: combines dense vector search in LanceDB with BM25 full-text search in Tantivy, deduplicated Reciprocal Rank Fusion (RRF), optional cross-encoder reranking (`Qwen3-Reranker-8B`), domain boosts, recency tie-breaking, canonical corpus-relative `doc_id` diversification, and explicit `source_class` (`curated` vs `vendor`) persistence across a unified `_execute_search` pipeline shared by `/search` and `/search/debug`.
- **Generation and chunk-count markers**: tracks synchronized `.generation.json` commit markers (generation, chunk count, and configured `LANCE_TABLE`) across both LanceDB and Tantivy, with atomic dual-store reader reopen that retries boundedly on mid-open commits so readers never pair mismatched store generations.
- **Write-ahead ingest journal & safe migrations (`scripts/ingest.py`)**: records in-flight mutations in `.ingest_journal.json`, validates embedding batch cardinality and vector dimensions before mutating either store, rolls back and releases the Tantivy writer lock on interrupted ingest, reconciles added/edited/emptied/deleted files across both stores, and auto-migrates legacy v1 schemas via staged rebuilds.
- **Per-corpus exclusive ingest locking**: enforces atomic cross-process exclusion (`INGEST_LOCKFILE`) across HTTP (`/ingest`, `/admin/ingest/reset`) and CLI (`scripts/ingest.py`) writers, holding the lock cleanly across spawn handoff until child exit.
- **Fail-closed readiness & search guardrails (HTTP 503)**: `/health`, `/ready`, `/search`, and `/search/debug` return HTTP 503 whenever indexes are empty or unloaded, a `.ingest_journal.json` recovery is pending, store generations/chunk counts disagree, the configured `LANCE_TABLE` mismatches committed markers, or an active writer blocks a consistent reader snapshot.
- **Security defaults & observability**: loopback-only bind by default (`127.0.0.1`) requiring both `ALLOW_EXTERNAL_BIND=1` and `RAG_API_KEY` for external binds, dedicated `RAG_ADMIN_API_KEY` for destructive resets, request body/parameter bounds, and single-prefix Prometheus counters on `/metrics`.
- **Offline CPU demo, evaluation harness & CI (`scripts/offline_demo.py`, `rag_eval.py`, `tests/`)**: deterministic `RAG_BACKEND=offline` stack with a synthetic public fixture corpus (`fixtures/demo_corpus/`), 10 labeled evaluation queries (`fixtures/demo_eval_queries.json`), `requirements-offline.txt`, GitHub Actions CI, and a 38-test unit and real-index regression suite.
