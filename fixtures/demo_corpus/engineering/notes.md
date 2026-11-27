# Engineering Index Reconciliation Notes

Incremental ingestion must reconcile three mutations across both LanceDB and Tantivy: added files, edited or emptied files, and deleted files removed from disk.

Whenever a source file is deleted or cleared to empty text, its previously indexed chunk IDs are purged from both LanceDB and Tantivy even when zero new chunks are written. An empty `--reset` drops the LanceDB table and clears Tantivy so both stores agree at zero chunks.

Generation markers (`.generation.json`) and a write-ahead `.ingest_journal.json` ensure interrupted updates are detected by readiness checks and cleanly replayed on the next ingest run.
