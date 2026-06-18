#!/usr/bin/env python3
"""Convert labeled TweetClaw exports to the notebook's hate-speech CSV format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

TEXT_FIELDS = (
    "tweet",
    "Tweet",
    "text",
    "full_text",
    "content",
    "tweet_text",
    "message",
)

LABEL_FIELDS = (
    "label",
    "hate_speech",
    "hate_speech_label",
    "category",
    "target",
    "classification",
)

ID_FIELDS = ("id", "tweet_id", "Tweet ID", "tweetId", "status_id")

LIST_FIELDS = ("tweets", "data", "results", "items", "records")

NEGATIVE_LABELS = {
    "1",
    "true",
    "hate",
    "hateful",
    "hate speech",
    "hate_speech",
    "racist",
    "sexist",
    "racist/sexist",
    "offensive",
    "toxic",
}

NON_HATE_LABELS = {
    "0",
    "false",
    "not hate",
    "not_hate",
    "non hate",
    "non_hate",
    "normal",
    "clean",
    "safe",
    "none",
    "neutral_hate",
}


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """Return dict records from common TweetClaw JSON export shapes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for field in LIST_FIELDS:
        value = payload.get(field)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def read_json_records(path: Path) -> list[dict[str, Any]]:
    return extract_records(json.loads(path.read_text(encoding="utf-8")))


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        records.append(item)
    return records


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json_records(path)
    if suffix in {".jsonl", ".ndjson"}:
        return read_jsonl_records(path)
    if suffix == ".csv":
        return read_csv_records(path)
    raise ValueError("Input must be .json, .jsonl, .ndjson, or .csv")


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return ""


def normalize_label(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in NON_HATE_LABELS:
        return "0"
    if lowered in NEGATIVE_LABELS:
        return "1"
    raise ValueError(
        "Unsupported label "
        f"{value!r}; use 0/1 or explicit hate-speech labels, not generic sentiment"
    )


def convert_record(record: dict[str, Any], index: int) -> dict[str, str]:
    tweet = first_text(record, TEXT_FIELDS)
    label = first_text(record, LABEL_FIELDS)
    if not tweet:
        raise ValueError(f"Record {index} has no tweet text")
    if not label:
        raise ValueError(f"Record {index} has no binary hate-speech label")
    tweet_id = first_text(record, ID_FIELDS) or str(index)
    return {"id": tweet_id, "label": normalize_label(label), "tweet": tweet}


def write_rows(input_path: Path, output_path: Path) -> int:
    records = read_records(input_path)
    rows = [convert_record(record, index) for index, record in enumerate(records, 1)]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "label", "tweet"))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert labeled TweetClaw JSON, JSONL, NDJSON, or CSV exports to "
            "the id,label,tweet CSV shape used by the notebook."
        )
    )
    parser.add_argument("input", type=Path, help="TweetClaw labeled export path")
    parser.add_argument(
        "output",
        type=Path,
        help='Output CSV path, for example "Twitter Sentiments.csv"',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = write_rows(args.input, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
