from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export train/val/test TF names from a saved tf_split.json."
    )
    parser.add_argument("tf_split", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Which split to export. Default: test.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.tf_split.open() as handle:
        split = json.load(handle)
    key = f"{args.split}_tf_names"
    names = split.get(key)
    if not isinstance(names, list) or not names:
        raise ValueError(f"{args.tf_split} does not contain a non-empty {key} list")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for name in sorted(str(name) for name in names):
            handle.write(f"{name}\n")
    print(f"Wrote {len(names)} {args.split} TF names to: {args.output}")


if __name__ == "__main__":
    main()
