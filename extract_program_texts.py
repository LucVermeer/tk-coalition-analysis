#!/usr/bin/env python3
"""
Extract text from election programs using a hybrid PDF text + OCR pipeline.

Primary source: https://www.verkiezingsprogrammasdownloaden.nl/
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import fitz  # type: ignore[import-untyped]
import pytesseract
from PIL import Image
from pytesseract import TesseractError
from tqdm import tqdm

DEFAULT_INPUT_DIR = Path("verkiezingsprogrammas")
DEFAULT_OUTPUT_DIR = Path("verkiezingsprogrammas_text")
MIN_TEXT_CHARS_PER_PAGE = 80


@dataclass
class ExtractionResult:
    path: Path
    text: str
    used_ocr_pages: int
    total_pages: int


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from Tweede Kamer 2025 verkiezingsprogramma's."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing PDF files (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store extracted text (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI for OCR fallback (default: %(default)s).",
    )
    parser.add_argument(
        "--lang",
        default="nld+eng",
        help="Tesseract language(s) to use, e.g. 'nld' or 'nld+eng' (default: %(default)s).",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Run OCR on every page, even when PDF text is present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate text files even if they already exist.",
    )
    parser.add_argument(
        "--tesseract-config",
        default="--psm 6",
        help="Extra config string forwarded to Tesseract (default: %(default)s).",
    )
    return parser.parse_args(argv)


def iter_pdf_files(root: Path) -> Sequence[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def compute_output_path(pdf_path: Path, output_root: Path, input_root: Path) -> Path:
    try:
        relative = pdf_path.relative_to(input_root)
    except ValueError:
        relative = Path(pdf_path.name)
    return output_root / Path(relative).with_suffix(".txt")


def should_use_pdf_text(text: str, page_index: int) -> bool:
    if not text:
        return False
    normalized = re.sub(r"\s+", "", text)
    if not normalized:
        return False
    threshold = MIN_TEXT_CHARS_PER_PAGE
    return len(normalized) >= threshold


def page_to_image(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def ocr_image(image: Image.Image, lang: str, config: str) -> str:
    try:
        return pytesseract.image_to_string(image, lang=lang, config=config)
    except TesseractError as error:
        message = str(error)
        if "Error opening data file" in message and lang != "eng":
            return pytesseract.image_to_string(image, lang="eng", config=config)
        raise RuntimeError(f"Tesseract OCR failed: {error}") from error


def extract_pdf(pdf_path: Path, dpi: int, lang: str, config: str, force_ocr: bool) -> ExtractionResult:
    doc = fitz.open(pdf_path)
    texts: List[str] = []
    ocr_pages = 0

    for index, page in enumerate(doc, start=1):
        text = page.get_text("text")
        use_pdf = should_use_pdf_text(text, index) and not force_ocr
        if use_pdf:
            texts.append(text)
            continue

        image = page_to_image(page, dpi=dpi)
        ocr_text = ocr_image(image, lang=lang, config=config)
        texts.append(ocr_text)
        ocr_pages += 1

    merged = "\n".join(texts)
    return ExtractionResult(
        path=pdf_path,
        text=merged,
        used_ocr_pages=ocr_pages,
        total_pages=len(doc),
    )


def write_text(result: ExtractionResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.text, encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    pdf_files = iter_pdf_files(args.input_dir)
    if not pdf_files:
        print(f"No PDF files found under {args.input_dir}", file=sys.stderr)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed: List[str] = []
    ocr_page_total = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        output_path = compute_output_path(
            pdf_path=pdf_path,
            output_root=args.output_dir,
            input_root=args.input_dir,
        )
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            result = extract_pdf(
                pdf_path=pdf_path,
                dpi=args.dpi,
                lang=args.lang,
                config=args.tesseract_config,
                force_ocr=args.force_ocr,
            )
            write_text(result=result, output_path=output_path)
            processed += 1
            ocr_page_total += result.used_ocr_pages
        except Exception as error:
            failed.append(f"{pdf_path}: {error}")

    print(
        f"Done. Generated: {processed}, skipped: {skipped}, failures: {len(failed)}, "
        f"OCR pages: {ocr_page_total}."
    )
    if failed:
        print("Failures:", file=sys.stderr)
        for entry in failed:
            print(f"- {entry}", file=sys.stderr)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
