"""
RAG Retrieval Eval Harness — 20 queries across all domains
"""

import json
import os
import csv
import requests
from datetime import datetime

RAG_SERVER = "http://localhost:9000"
EVAL_OUTPUT_DIR = "eval_results"
DENSE_TOP_N = 20
BM25_TOP_N = 20
RRF_K = 60
TOP_K = 5

EVAL_QUERIES = [
    # --- AUTOMOTIVE (2) ---
    {
        "query": "What is the recommended oil type and capacity for a 2019 Panamera GTS",
        "domain": None,
        "expected_keywords": ["oil", "synthetic", "quarts", "10,000"],
        "notes": "Should hit training data or manual with LL-04-FE spec",
    },
    {
        "query": "What are the common reliability issues on the 971 Panamera",
        "domain": None,
        "expected_keywords": ["air suspension", "fuel pump", "water pump"],
        "notes": "Should hit automotive training data reliability section",
    },
    # --- BUDDHIST (2) ---
    {
        "query": "What are the twelve links of dependent origination",
        "domain": None,
        "expected_keywords": ["ignorance", "formations", "consciousness", "contact"],
        "notes": "Should hit 05-dependent-origination.md",
    },
    {
        "query": "What are the four noble truths and how do they relate to suffering",
        "domain": None,
        "expected_keywords": ["dukkha", "noble", "truth", "cessation"],
        "notes": "Should hit 03-four-noble-truths.md",
    },
    # --- FINANCE (3) ---
    {
        "query": "How does implied volatility differ from historical volatility",
        "domain": None,
        "expected_keywords": ["implied", "historical", "realized", "premium"],
        "notes": "Should hit options/volatility training docs",
    },
    {
        "query": "What is the Black-Scholes model and what are its key inputs",
        "domain": None,
        "expected_keywords": ["black-scholes", "strike", "volatility", "expiration"],
        "notes": "Should hit 01-black-scholes-model.md",
    },
    {
        "query": "How do you calculate the Sharpe ratio and what does it measure",
        "domain": None,
        "expected_keywords": ["sharpe", "risk", "return", "standard deviation"],
        "notes": "Should hit 01-sharpe-and-sortino-ratios.md",
    },
    # --- MANUAL (2) ---
    {
        "query": "What is the fuel tank capacity of the Panamera",
        "domain": None,
        "expected_keywords": ["fuel", "tank", "gallons", "litres"],
        "notes": "Should hit owners manual filling capacities section",
    },
    {
        "query": "What are the tire pressure specifications for the Panamera",
        "domain": None,
        "expected_keywords": ["tire", "pressure", "psi"],
        "notes": "Should hit owners manual tire pressure section",
    },
    # --- RENNLIST (2) ---
    {
        "query": "How do you change the PDK transmission fluid on a 971 Panamera",
        "domain": None,
        "expected_keywords": ["PDK", "fluid", "fill", "drain"],
        "notes": "Should hit PDK fluid change thread",
    },
    {
        "query": "What suspension differences exist between the GTS and Turbo Panamera",
        "domain": None,
        "expected_keywords": ["suspension", "air", "sport", "comfort"],
        "notes": "Should hit comfort thread or air suspension thread",
    },
    # --- STOIC (2) ---
    {
        "query": "What does Marcus Aurelius say about controlling your reactions",
        "domain": None,
        "expected_keywords": ["control", "judgment", "mind", "power"],
        "notes": "Should hit meditations or core teachings",
    },
    {
        "query": "What does Seneca teach about the shortness of life and wasting time",
        "domain": None,
        "expected_keywords": ["time", "life", "waste", "busy"],
        "notes": "Should hit letters-wisdom.md",
    },
    # --- STRATEGY (2) ---
    {
        "query": "What are Sun Tzu's core principles about deception in warfare",
        "domain": None,
        "expected_keywords": ["deception", "appear", "weak", "strong"],
        "notes": "Should hit art-of-war or deception doc",
    },
    {
        "query": "How does game theory apply to competitive business strategy",
        "domain": None,
        "expected_keywords": ["game theory", "nash", "equilibrium", "prisoner"],
        "notes": "Should hit game-theory-competitive-dynamics.md",
    },
    # --- NEWS (1) ---
    {
        "query": "What happened with VIX and implied volatility in 2024 and 2025",
        "domain": None,
        "expected_keywords": ["VIX", "implied", "realized", "0DTE"],
        "notes": "Should hit news corpus volatility analysis",
    },
    # --- PERSONAL (1) ---
    {
        "query": "How does Trevor make decisions when he lacks full conviction",
        "domain": None,
        "expected_keywords": ["conviction", "wait", "opportunity"],
        "notes": "Should hit decision-examples doc",
    },
    # --- TRADING (1) ---
    {
        "query": "What positions has Trevor traded in his Fidelity account",
        "domain": None,
        "expected_keywords": ["fidelity", "trade", "position"],
        "notes": "Should hit trading export docs",
    },
    # --- TSB (1) ---
    {
        "query": "What technical service bulletins exist for the Panamera engine",
        "domain": None,
        "expected_keywords": ["engine", "TSB", "bulletin"],
        "notes": "Should hit TSB index docs",
    },
    # --- CROSS-DOMAIN (1) ---
    {
        "query": "What is the best approach to maintaining a Panamera long term",
        "domain": None,
        "expected_keywords": ["maintenance", "service", "interval", "oil"],
        "notes": "Should prefer automotive/manual/TSB over forum chatter",
    },
]


def run_debug_search(query, domain=None):
    payload = {"query": query, "dense_top_n": DENSE_TOP_N, "bm25_top_n": BM25_TOP_N, "rrf_k": RRF_K, "top_k": TOP_K}
    if domain:
        payload["domain"] = domain
    resp = requests.post(f"{RAG_SERVER}/search/debug", json=payload)
    resp.raise_for_status()
    return resp.json()


def compute_diagnostics(sr, eq):
    diag = {}
    stages = sr["stages"]
    expected_kw = [kw.lower() for kw in eq.get("expected_keywords", [])]
    query_domain = eq.get("domain")

    for stage_name, chunks in stages.items():
        keyword_hits = []
        for chunk in chunks:
            text_lower = chunk.get("text", "").lower()
            matched = [kw for kw in expected_kw if kw in text_lower]
            if matched:
                keyword_hits.append({"id": chunk.get("id", ""), "rank": chunk.get("rank", 0), "domain": chunk.get("domain", ""), "matched_keywords": matched, "rerank_score": chunk.get("rerank_score"), "rrf_score": chunk.get("rrf_score"), "bm25_score": chunk.get("bm25_score")})
        diag[stage_name] = {"total_chunks": len(chunks), "keyword_hits": keyword_hits, "keyword_hit_count": len(keyword_hits)}

    final = stages.get("final", [])
    if query_domain and final:
        wrong = [c for c in final if c.get("domain") != query_domain]
        diag["cross_domain"] = {"wrong_count": len(wrong), "total": len(final), "details": [{"id": c["id"], "domain": c.get("domain"), "rank": c["rank"], "rerank_score": c.get("rerank_score")} for c in wrong]}

    reranked = stages.get("reranked", [])
    if reranked:
        scores = [c.get("rerank_score", 0) for c in reranked]
        diag["reranker_scores"] = {"max": max(scores), "min": min(scores), "spread": round(max(scores) - min(scores), 4), "top5": scores[:5], "bottom5": scores[-5:], "top1_vs_top5_gap": round(scores[0] - scores[4], 4) if len(scores) >= 5 else None}

    dense_hits = set(h["id"] for h in diag.get("dense", {}).get("keyword_hits", []))
    bm25_hits = set(h["id"] for h in diag.get("bm25", {}).get("keyword_hits", []))
    rrf_hits = set(h["id"] for h in diag.get("rrf_merged", {}).get("keyword_hits", []))
    final_hits = set(h["id"] for h in diag.get("final", {}).get("keyword_hits", []))
    retrieval_hits = dense_hits | bm25_hits
    diag["dropout_analysis"] = {"in_dense": len(dense_hits), "in_bm25": len(bm25_hits), "in_rrf": len(rrf_hits), "in_final": len(final_hits), "dropped_by_reranker": list(retrieval_hits - final_hits), "never_retrieved": not bool(retrieval_hits) and bool(expected_kw)}
    return diag


def truncate(text, max_len=120):
    return text[:max_len] + "..." if len(text) > max_len else text


def print_summary(eq, sr, diag):
    print(f"\n{'='*80}")
    print(f"QUERY: {eq['query']}")
    if eq.get("notes"): print(f"NOTES: {eq['notes']}")
    print(f"TIMINGS: {sr['timings']}")

    final = sr["stages"].get("final", [])
    print(f"\n--- FINAL ({len(final)} chunks) ---")
    for chunk in final:
        domain_tag = f"[{chunk.get('domain', '')}]"
        score_parts = []
        if chunk.get("rerank_score") is not None: score_parts.append(f"rerank={chunk['rerank_score']:.4f}")
        print(f"  #{chunk.get('rank', '?'):2}  {'  '.join(score_parts):20s}  {domain_tag:14s}  {truncate(chunk.get('text', ''))}")

    dropout = diag.get("dropout_analysis", {})
    rs = diag.get("reranker_scores", {})
    print(f"\n  KW HITS: dense={dropout.get('in_dense',0)} bm25={dropout.get('in_bm25',0)} final={dropout.get('in_final',0)}")
    if dropout.get("dropped_by_reranker"): print(f"  DROPPED: {len(dropout['dropped_by_reranker'])} chunks")
    if rs: print(f"  RERANKER: spread={rs.get('spread','')} gap_1v5={rs.get('top1_vs_top5_gap','')}")


def save_result(eq, sr, diag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c == " " else "_" for c in eq["query"])[:60].strip().replace(" ", "_")
    path = os.path.join(output_dir, f"{safe}.json")
    data = {"query": eq["query"], "domain": eq.get("domain"), "expected_keywords": eq.get("expected_keywords", []), "notes": eq.get("notes", ""), "timestamp": datetime.now().isoformat(), "timings": sr["timings"], "counts": sr["counts"], "diagnostics": diag, "stages": {}}
    for sn, chunks in sr["stages"].items():
        data["stages"][sn] = [{**{k: v for k, v in c.items() if k != "text"}, "text_preview": truncate(c.get("text", ""), 200), "text_full": c.get("text", "")} for c in chunks]
    with open(path, "w") as f: json.dump(data, f, indent=2)
    return path


def write_summary_csv(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_summary.csv")
    rows = []
    for item in results:
        eq, sr, diag = item["eval_query"], item["server_response"], item["diagnostics"]
        dropout = diag.get("dropout_analysis", {})
        rs = diag.get("reranker_scores", {})
        cd = diag.get("cross_domain", {})
        t = sr.get("timings", {})
        # Get top result domain
        final = sr["stages"].get("final", [])
        top_domain = final[0].get("domain", "") if final else ""
        top_score = final[0].get("rerank_score", 0) if final else 0
        rows.append({"query": eq["query"], "top_domain": top_domain, "top_score": round(top_score, 4), "rerank_ms": t.get("rerank", 0), "total_ms": t.get("total", 0), "kw_in_final": dropout.get("in_final", 0), "dropped": len(dropout.get("dropped_by_reranker", [])), "spread": rs.get("spread", ""), "gap_1v5": rs.get("top1_vs_top5_gap", ""), "notes": eq.get("notes", "")})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    try: requests.get(f"{RAG_SERVER}/docs", timeout=5)
    except: print(f"Cannot reach {RAG_SERVER}"); return

    print(f"Running {len(EVAL_QUERIES)} eval queries\n")
    results = []
    for i, eq in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{len(EVAL_QUERIES)}] {eq['query']}")
        try:
            sr = run_debug_search(eq["query"], eq.get("domain"))
            diag = compute_diagnostics(sr, eq)
            results.append({"eval_query": eq, "server_response": sr, "diagnostics": diag})
            print_summary(eq, sr, diag)
            save_result(eq, sr, diag, EVAL_OUTPUT_DIR)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if results:
        csv_path = write_summary_csv(results, EVAL_OUTPUT_DIR)
        print(f"\n{'='*80}")
        print(f"Summary: {csv_path}")
        print(f"Ran {len(results)}/{len(EVAL_QUERIES)} queries")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
