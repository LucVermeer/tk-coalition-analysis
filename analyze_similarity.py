#!/usr/bin/env python3
"""
Aggregate ManifestoBERTa predictions per party and compute similarity matrices.

Outputs:
  - topic vectors per party saved as Parquet/CSV
  - cosine similarity matrix between parties
  - per-pair topic contributions (top overlaps/divergences)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_PREDICTION_DIR = Path("verkiezingsprogrammas_predictions")
DEFAULT_OUTPUT_DIR = Path("analysis")
DEFAULT_TOP_K = 5


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise manifesto topic vectors and compute similarities."
    )
    parser.add_argument(
        "-p",
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Directory containing JSON predictions (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store aggregated outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding for JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of topics to report for overlap/divergence per pair (default: %(default)s).",
    )
    return parser.parse_args()


def collect_prediction_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {root}")
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_probabilities(path: Path, encoding: str) -> pd.DataFrame:
    with path.open(encoding=encoding) as handle:
        data = json.load(handle)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    probs = df["probabilities"].apply(_prob_list_to_dict)
    prob_df = pd.DataFrame(probs.tolist())
    # convert percentages back to 0-1 range
    prob_df = prob_df.astype(float) / 100.0
    return prob_df


def _prob_list_to_dict(items: List[List]) -> Dict[str, float]:
    # Each probability entry is expected as [label, value]
    return {label: value for label, value in items}


def aggregate_vectors_safe(prediction_paths: List[Path], base_dir: Path, encoding: str) -> pd.DataFrame:
    records = []
    for path in prediction_paths:
        prob_df = load_probabilities(path, encoding)
        if prob_df.empty:
            continue
        vector = prob_df.mean(axis=0)
        relative_name = path.relative_to(base_dir).with_suffix("").as_posix()
        vector.name = relative_name
        records.append(vector)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).fillna(0.0)
    df.index.name = "program"
    df.sort_index(inplace=True)
    return df


def compute_cosine_similarity(vectors: pd.DataFrame) -> pd.DataFrame:
    matrix = vectors.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    similarity = normalized @ normalized.T
    sim_df = pd.DataFrame(similarity, index=vectors.index, columns=vectors.index)
    return sim_df


def pairwise_contributions(
    vectors: pd.DataFrame,
    similarity: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, object]]:
    matrix = vectors.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    topics = list(vectors.columns)
    entries: List[Dict[str, object]] = []
    for i in range(len(vectors.index)):
        for j in range(i + 1, len(vectors.index)):
            a_name = vectors.index[i]
            b_name = vectors.index[j]
            contrib_values = normalized[i] * normalized[j]
            top_shared = sorted(
                zip(topics, contrib_values),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            diff = matrix[i] - matrix[j]
            top_a = sorted(
                zip(topics, diff),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            top_b = sorted(
                zip(topics, diff),
                key=lambda item: item[1],
            )[:top_k]
            entries.append(
                {
                    "party_a": a_name,
                    "party_b": b_name,
                    "similarity": float(similarity.loc[a_name, b_name]),
                    "shared_topics": [
                        {"topic": topic, "contribution": round(float(value), 4)}
                        for topic, value in top_shared
                    ],
                    "topics_more_in_a": [
                        {"topic": topic, "delta": round(float(value), 4)}
                        for topic, value in top_a
                    ],
                    "topics_more_in_b": [
                        {"topic": topic, "delta": round(float(value), 4)}
                        for topic, value in top_b
                    ],
                }
            )
    return entries


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_paths = collect_prediction_files(args.predictions_dir)
    vectors = aggregate_vectors_safe(prediction_paths, args.predictions_dir, args.encoding)
    if vectors.empty:
        print("No prediction vectors aggregated; exiting.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = args.output_dir / "party_topic_vectors.parquet"
    vectors.to_parquet(vectors_path, compression="snappy")
    vectors.to_csv(args.output_dir / "party_topic_vectors.csv")
    print(f"Saved topic vectors -> {vectors_path}")

    similarity = compute_cosine_similarity(vectors)
    similarity_path = args.output_dir / "party_similarity_matrix.csv"
    similarity.to_csv(similarity_path)
    print(f"Saved similarity matrix -> {similarity_path}")

    contributions = pairwise_contributions(vectors, similarity, args.top_k)
    contributions_path = args.output_dir / "pairwise_contributions.json"
    with contributions_path.open("w", encoding="utf-8") as handle:
        json.dump(contributions, handle, ensure_ascii=False, indent=2)
    print(f"Saved pairwise contributions -> {contributions_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
