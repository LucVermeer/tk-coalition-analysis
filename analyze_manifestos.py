#!/usr/bin/env python3
"""
Analyze Dutch election manifestos using ManifestoBERTa with contextual windows.

This script loads preprocessed text files (one per manifesto) and classifies
individual sentences or segments using the Manifesto Project's RoBERTa model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

DEFAULT_INPUT_DIR = Path("verkiezingsprogrammas_text")
DEFAULT_OUTPUT_DIR = Path("verkiezingsprogrammas_predictions")
DEFAULT_MODEL = "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1"
DEFAULT_TOKENIZER = "xlm-roberta-large"
DEFAULT_MAX_LENGTH = 200
DEFAULT_CONTEXT_SENTENCES = 3


@dataclass
class SegmentPrediction:
    party: str
    manifest_path: Path
    sentence_index: int
    sentence: str
    context: str
    probabilities: List[Tuple[str, float]]
    predicted_label: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify manifesto sentences with ManifestoBERTa."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with cleaned text files (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store JSON predictions (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="ManifestoBERTa model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length passed to the model (default: %(default)s).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_SENTENCES,
        help="Number of previous sentences to include as context (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of sentences per file (0 means no limit).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for input text files (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: %(default)s).",
    )
    return parser.parse_args()


def detect_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(choice)


def iter_text_files(root: Path) -> Iterator[Path]:
    return (path for path in sorted(root.rglob("*.txt")) if path.is_file())


def read_sentences(file_path: Path, encoding: str, limit: int) -> List[str]:
    text = file_path.read_text(encoding=encoding)
    sentences = split_into_sentences(text)
    if limit > 0:
        return sentences[:limit]
    return sentences


def split_into_sentences(text: str) -> List[str]:
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []
    sentences: List[str] = []
    buffer: List[str] = []
    for char in normalized:
        buffer.append(char)
        if char == ".":
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []
    if buffer:
        trailing = "".join(buffer).strip()
        if trailing:
            sentences.append(trailing)
    return sentences


def build_context_with_limit(
    tokenizer,
    sentences: List[str],
    index: int,
    context_window: int,
    max_length: int,
) -> str:
    start = max(0, index - context_window)
    selected = sentences[start:index] + [sentences[index]]
    while selected:
        context_text = " ".join(selected)
        tokenized = tokenizer(
            sentences[index],
            context_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        token_count = len(tokenized["input_ids"])
        if token_count <= max_length or len(selected) == 1:
            return context_text
        selected.pop(0)
    return sentences[index]


def run_inference(
    tokenizer,
    model,
    batches: List[Tuple[str, str]],
    device: torch.device,
    max_length: int,
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    inputs = tokenizer(
        [sentence for sentence, context in batches],
        [context for sentence, context in batches],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).cpu().tolist()

    label_map = model.config.id2label
    predictions = []
    for probs in probabilities:
        label_probs = [(label_map[i], float(prob)) for i, prob in enumerate(probs)]
        label_probs.sort(key=lambda item: item[1], reverse=True)
        predicted_label = label_probs[0][0]
        predictions.append((predicted_label, label_probs))
    return predictions


def analyze_manifesto(
    file_path: Path,
    tokenizer,
    model,
    args: argparse.Namespace,
    device: torch.device,
) -> List[SegmentPrediction]:
    sentences = read_sentences(file_path, args.encoding, args.limit)
    if not sentences:
        return []

    party = file_path.parent.name
    results: List[SegmentPrediction] = []
    batch: List[Tuple[int, str, str]] = []

    for index, sentence in enumerate(tqdm(sentences, desc=f"Sentences in {file_path.name}", leave=False)):
        context = build_context_with_limit(
            tokenizer=tokenizer,
            sentences=sentences,
            index=index,
            context_window=args.context_window,
            max_length=args.max_length,
        )
        batch.append((index, sentence, context))

        if len(batch) == args.batch_size or index == len(sentences) - 1:
            payload = [(sentence, context) for _, sentence, context in batch]
            predictions = run_inference(tokenizer, model, payload, device, args.max_length)

            for (sentence_idx, sentence_text, context_text), (predicted, label_probs) in zip(batch, predictions):
                probs_percentage = [(label, round(prob * 100, 2)) for label, prob in label_probs]
                results.append(
                    SegmentPrediction(
                        party=party,
                        manifest_path=file_path,
                        sentence_index=sentence_idx,
                        sentence=sentence_text,
                        context=context_text,
                        probabilities=probs_percentage,
                        predicted_label=predicted,
                    )
                )
            batch.clear()
    return results


def write_predictions(predictions: List[SegmentPrediction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "party": prediction.party,
            "manifest_path": str(prediction.manifest_path),
            "sentence_index": prediction.sentence_index,
            "sentence": prediction.sentence,
            "context": prediction.context,
            "predicted_label": prediction.predicted_label,
            "probabilities": prediction.probabilities,
        }
        for prediction in predictions
    ]
    output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    device = detect_device(args.device)

    if args.output_dir.exists():
        existing_outputs = {path.relative_to(args.output_dir).with_suffix("") for path in args.output_dir.rglob("*.json")}
    else:
        existing_outputs = set()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)
    model.eval()

    file_paths = list(iter_text_files(args.input_dir))
    for file_path in tqdm(file_paths, desc="Programs"):
        relative_stem = file_path.relative_to(args.input_dir).with_suffix("")
        if relative_stem in existing_outputs:
            print(f"Skipping already processed file {file_path}")
            continue

        predictions = analyze_manifesto(file_path, tokenizer, model, args, device)
        if not predictions:
            continue
        relative = file_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative.with_suffix(".json")
        write_predictions(predictions, output_path)
        print(f"Wrote predictions -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
