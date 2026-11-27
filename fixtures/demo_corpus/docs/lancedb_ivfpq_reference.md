# LanceDB IVF_PQ Vector Index Reference

LanceDB supports IVF_PQ approximate nearest-neighbor indexes for cosine similarity search across high-dimensional dense embeddings.

Key tuning parameters:
- `nprobes`: number of inverted file partitions scanned per query. Low `nprobes` reduces latency but misses boundary neighbors.
- `refine_factor`: re-ranks the top `k * refine_factor` quantized PQ candidates using their exact uncompressed vectors. Setting `refine_factor=20` lifts recall@10 from roughly 0.47 to over 0.91 with only a few milliseconds of overhead compared to an exhaustive scan.

This file is a synthetic vendor reference document stored under `docs/` so it carries `source_class=vendor`.
