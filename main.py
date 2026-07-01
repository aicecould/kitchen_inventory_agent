"""Command-line entry point for the backend-only prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kitchen Inventory Agent")
    parser.add_argument("--text", required=True, help="User request")
    parser.add_argument("--image", type=Path, help="Optional ingredient image")
    parser.add_argument("--user-id", default="prototype-user")
    parser.add_argument("--language", default="zh", help="Baidu target language code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_bytes = args.image.read_bytes() if args.image else None
    pipeline = build_pipeline()
    result = pipeline.process_request(
        user_id=args.user_id,
        text=args.text,
        image_bytes=image_bytes,
        target_language=args.language,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
