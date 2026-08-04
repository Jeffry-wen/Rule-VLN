#!/usr/bin/env python3
"""Upload the Rule-VLN Hugging Face dataset card and local rule images."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path


IGNORE_PATTERNS = [
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "__pycache__/**",
    "*.tmp",
]


def summarize_folder(folder: Path) -> tuple[int, int, Counter[str]]:
    count = 0
    total_bytes = 0
    extensions: Counter[str] = Counter()
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        count += 1
        total_bytes += path.stat().st_size
        extensions[path.suffix.lower() or "<none>"] += 1
    return count, total_bytes, extensions


def require_rules_dataset(folder: Path) -> None:
    missing = []
    for relative in ["label.json", "img"]:
        if not (folder / relative).exists():
            missing.append(relative)
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"{folder} is missing required entries: {joined}")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload HF_DATASET_CARD.md as README.md and upload the local "
            "City_Navigation_Rules_Dataset folder to a Hugging Face dataset repo."
        )
    )
    parser.add_argument("--repo-id", default="jeffry77/Rule-VLN")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument(
        "--rules-dir",
        type=Path,
        default=Path(r"D:\City_Navigation_Rules_Dataset"),
        help="Local City_Navigation_Rules_Dataset directory.",
    )
    parser.add_argument(
        "--rules-path-in-repo",
        default="City_Navigation_Rules_Dataset",
        help="Remote folder path inside the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=Path("HF_DATASET_CARD.md"),
        help="Dataset card to upload as README.md.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN or cached login.",
    )
    parser.add_argument(
        "--skip-card",
        action="store_true",
        help="Do not upload the dataset card.",
    )
    parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Do not upload the semantic rule image dataset.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without contacting Hugging Face.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check Hugging Face authentication and repo access, then exit.",
    )
    args = parser.parse_args()

    rules_dir = args.rules_dir.expanduser().resolve()
    card = args.card.expanduser().resolve()
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print(
            "HF_TOKEN is not set. The script will try a cached Hugging Face login."
        )

    if not args.dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: huggingface_hub. Install it with "
                "`pip install huggingface_hub` or run inside the Rule-VLN environment."
            ) from exc

        api = HfApi(token=token)
        try:
            whoami = api.whoami()
            username = whoami.get("name") or whoami.get("fullname") or "<unknown>"
            print(f"Authenticated on Hugging Face as: {username}")
        except Exception as exc:
            raise SystemExit(
                "Hugging Face authentication failed before upload.\n"
                f"Original error: {type(exc).__name__}: {exc}\n"
                "Set HF_TOKEN to a token with write permission, pass --token, "
                "or run `hf auth login` in the same Python environment. "
                "If you pasted a token in chat, revoke it after use."
            ) from exc
    else:
        api = None

    if not args.skip_rules:
        require_rules_dataset(rules_dir)
        count, total_bytes, extensions = summarize_folder(rules_dir)
        print(f"Rules dataset: {rules_dir}")
        print(f"  files: {count}")
        print(f"  size:  {format_bytes(total_bytes)}")
        print("  extensions:")
        for extension, ext_count in extensions.most_common(10):
            print(f"    {extension}: {ext_count}")

    if not args.skip_card:
        if not card.exists():
            raise FileNotFoundError(f"Dataset card not found: {card}")
        print(f"Dataset card: {card} -> README.md")

    if args.dry_run:
        print("Dry run complete. No upload was performed.")
        return

    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    print(f"Dataset repo is accessible: {args.repo_id}")

    if args.preflight_only:
        print("Preflight complete. No upload was performed.")
        return

    if not args.skip_card:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            commit_message="Update Rule-VLN dataset card",
        )
        print("Uploaded dataset card.")

    if not args.skip_rules:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(rules_dir),
            path_in_repo=args.rules_path_in_repo,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message="Add semantic rule image dataset",
        )
        print(f"Uploaded rules dataset to {args.rules_path_in_repo}/.")


if __name__ == "__main__":
    main()
