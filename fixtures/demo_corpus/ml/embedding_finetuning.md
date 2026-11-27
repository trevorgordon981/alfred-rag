# Embedding Fine-Tuning Strategy

Fine-tuning dense embedding models with `MultipleNegativesRankingLoss` uses in-batch negatives so training requires only `(query, positive)` pairs without manual hard-negative mining.

Hold out a tail evaluation split (`--eval-split 0.1`) scored with `InformationRetrievalEvaluator` to measure domain adaptation gains before swapping the served checkpoint. Always pin HuggingFace model revisions via `EMBED_MODEL_REVISION` and `RERANKER_MODEL_REVISION` for reproducible builds.
