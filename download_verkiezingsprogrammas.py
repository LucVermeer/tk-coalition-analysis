#!/usr/bin/env python3
"""
Download all available verkiezingsprogramma's for the Tweede Kamerverkiezingen 2025.

Data source: https://www.verkiezingsprogrammasdownloaden.nl/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://www.verkiezingsprogrammasdownloaden.nl"
API_URL = f"{BASE_URL}/api/parties"
USER_AGENT = "tk-coalitie-bouwer/1.0 (+https://www.verkiezingsprogrammasdownloaden.nl/)"


def fetch_parties() -> List[dict]:
    request = Request(API_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response:
            if response.status != 200:
                raise RuntimeError(f"API request failed with status {response.status}")
            payload = response.read()
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Failed to fetch party list: {error}") from error

    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("Failed to decode API response as JSON") from error


def normalize_party_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower())
    slug = slug.strip("-")
    return slug or "onbekend"


def build_program_url(program_path: str) -> str:
    normalized = program_path.lstrip("/")
    encoded = quote(normalized, safe="/:%+-._()")
    return f"{BASE_URL}/{encoded}"


def download_file(url: str, destination: Path, force: bool) -> bool:
    if destination.exists() and not force:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response, destination.open("wb") as handle:
            handle.write(response.read())
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Failed to download {url}: {error}") from error

    return True


def download_programs(output_dir: Path, force: bool) -> None:
    parties = fetch_parties()
    available = [
        party
        for party in parties
        if party.get("hasProgram") and party.get("programPath")
    ]

    if not available:
        print("No programs available to download.", file=sys.stderr)
        return

    downloaded = 0
    skipped = 0
    failures: List[str] = []

    for party in available:
        party_name = party.get("name", "Onbekend")
        slug = normalize_party_name(party_name)
        program_path = party["programPath"]
        program_url = build_program_url(program_path)
        filename = Path(program_path).name
        destination = output_dir / slug / filename

        try:
            changed = download_file(program_url, destination, force=force)
        except RuntimeError as error:
            failures.append(f"{party_name}: {error}")
            continue

        if changed:
            downloaded += 1
            print(f"Downloaded {party_name} -> {destination}")
        else:
            skipped += 1
            print(f"Skipped (already exists) {party_name} -> {destination}")

    print(
        f"Done. Downloaded: {downloaded}, skipped: {skipped}, failed: {len(failures)}."
    )
    if failures:
        print("Failures:", file=sys.stderr)
        for message in failures:
            print(f"- {message}", file=sys.stderr)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all available verkiezingsprogramma's for the "
            "Tweede Kamerverkiezingen 2025."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("verkiezingsprogrammas"),
        help="Directory to store downloaded files (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        download_programs(args.output_dir, args.force)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
