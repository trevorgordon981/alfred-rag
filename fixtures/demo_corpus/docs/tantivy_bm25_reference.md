# Tantivy BM25 Full-Text Index Reference

Tantivy provides inverted-index BM25 scoring with configurable tokenizers such as `en_stem` for English morphological stemming and `raw` for exact identifier matching.

Operational notes:
- Use `tokenizer_name="raw"` on identifier fields such as `id`, `doc_id`, and `domain` so term deletions match exact values rather than tokenized fragments.
- Deleting documents requires committing the `IndexWriter`, waiting for merge threads or dropping the writer to release the lockfile, and calling `index.reload()` on readers so deleted segments disappear from search results.

This file is a synthetic vendor reference document stored under `docs/` so it carries `source_class=vendor`.
