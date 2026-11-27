# Hybrid Retrieval and Reciprocal Rank Fusion Notes

Hybrid search combines dense vector retrieval from LanceDB with sparse BM25 retrieval from Tantivy so semantic paraphrases and exact lexical identifiers reinforce each other.

Reciprocal Rank Fusion (RRF) merges ranked lists using the formula `score(d) = sum(1 / (k + rank_i))` with a smoothing constant `k=60`. Deduplicating each retriever leg before summation prevents any repeated chunk ID from being double-counted.

After RRF fusion, the top 10 candidates pass to a cross-encoder reranker that scores each `(query, document)` pair with a calibrated relevance probability. Both `/search` and `/search/debug` must execute this exact same pipeline so offline evaluations match production traffic.
