"""
Fine-tune Qwen3-Embedding-8B (or any sentence-transformers model) on
query/positive pairs using MultipleNegativesRankingLoss.

Training data format (JSONL):
  {"query": "...", "positive": "..."}
  {"query": "...", "positive": "..."}
  ...

Usage:
  python finetune_embedding.py \\
    --base-model Qwen/Qwen3-Embedding-8B \\
    --data-path ./training_data/embedding_pairs.jsonl \\
    --output-dir ./models/qwen3-embedding-8b-finetuned \\
    --epochs 3 --batch-size 4 --lr 3e-5

All knobs can also be set via env vars: BASE_MODEL, DATA_PATH, OUTPUT_DIR, EPOCHS,
BATCH_SIZE, LR, WARMUP_RATIO, EVAL_SPLIT, SEED.
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-Embedding-8B"),
                   help="HuggingFace model id or local path")
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", "./training_data/embedding_pairs.jsonl"),
                   help="JSONL file with {\"query\": ..., \"positive\": ...} per line")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./models/finetuned"),
                   help="Where to save the fine-tuned model")
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "3")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "4")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "3e-5")))
    p.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("WARMUP_RATIO", "0.1")))
    p.add_argument("--eval-split", type=float, default=float(os.environ.get("EVAL_SPLIT", "0.1")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def load_data(path: str) -> list[tuple[str, str]]:
    """Load query/positive pairs from JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            query = d.get("query", "")
            positive = d.get("positive", "")
            if query and positive:
                pairs.append((query, positive))
    return pairs


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sep = "=" * 60
    print(sep)
    print("  Embedding Fine-Tuning")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base model: {args.base_model}")
    print(f"  Output:     {args.output_dir}")
    print(sep)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    pairs = load_data(args.data_path)
    print(f"  Total pairs: {len(pairs)}")
    if not pairs:
        raise SystemExit("No training pairs loaded. Check --data-path.")

    # Split train/eval
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.eval_split))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    print(f"  Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # Create InputExamples for MNR loss (query, positive)
    train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]

    # Build IR evaluator from eval split
    eval_queries = {str(i): q for i, (q, _) in enumerate(eval_pairs)}
    eval_corpus = {str(i): p for i, (_, p) in enumerate(eval_pairs)}
    eval_relevant = {str(i): {str(i)} for i in range(len(eval_pairs))}

    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant,
        name="eval",
        show_progress_bar=True,
    )

    # Load model
    print("\nLoading base model...")
    model = SentenceTransformer(args.base_model)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    eval_steps = max(len(train_dataloader) // 2, 1)
    print("\nTraining config:")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Warmup steps:   {warmup_steps}")
    print(f"  Steps per epoch: {len(train_dataloader)}")
    print(f"  Total steps:    {len(train_dataloader) * args.epochs}")
    print("\nStarting training...\n")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
        evaluation_steps=eval_steps,
        save_best_model=True,
        checkpoint_save_steps=eval_steps,
        checkpoint_path=f"{args.output_dir}/checkpoints",
    )

    print(sep)
    print("  Training complete.")
    print(f"  Model saved to: {args.output_dir}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)


if __name__ == "__main__":
    main()
