# 2026 Chunking and Source Diversification Guidelines

Paragraph-aware chunking splits documents at double-newline boundaries first, falling back to single newlines and word spaces, with a 1200-character target window and 200-character overlap.

Source diversification limits results per document (`max_per_source`) using the canonical corpus-relative `doc_id` rather than bare filenames. Keying on `doc_id` prevents two distinct files that share a basename (such as `operations/notes.md` and `engineering/notes.md`) from collapsing into a single bucket.

Updated policy for 2026: prefer paragraph-aware chunking with canonical `doc_id` diversification on all new corpora.
