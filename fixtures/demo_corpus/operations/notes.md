# Operations Security and Readiness Runbook

Production deployments bind `127.0.0.1` by default so corpus reads and ingestion endpoints are never exposed unintentionally on external network interfaces.

Before exposing the HTTP server beyond loopback (`HOST=0.0.0.0`), operators must set both `ALLOW_EXTERNAL_BIND=1` and `RAG_API_KEY`. Full index resets are separated onto the administrative endpoint `POST /admin/ingest/reset` and protected by `RAG_ADMIN_API_KEY`.

Readiness probes (`GET /ready` and `GET /health`) return HTTP 503 whenever either store is empty, an interrupted ingest journal is present, or LanceDB and Tantivy report mismatched generation numbers.
