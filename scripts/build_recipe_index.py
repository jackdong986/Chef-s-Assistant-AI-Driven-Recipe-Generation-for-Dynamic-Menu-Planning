"""Build or refresh the complete local SQLite recipe index."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recipe_store import RecipeStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            os.getenv(
                "RECIPE_DATASET_PATH",
                str(Path.home() / "Downloads" / "RAW_recipes_with_amount.csv"),
            )
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            os.getenv(
                "RECIPE_INDEX_PATH",
                str(PROJECT_ROOT / ".cache" / "recipes.sqlite3"),
            )
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild a current index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    last_report = 0

    def report(count: int) -> None:
        nonlocal last_report
        if count - last_report >= 25_000:
            print(f"Indexed {count:,} CSV records...")
            last_report = count

    store = RecipeStore(args.dataset, args.index)
    stats = store.ensure_index(force=args.force, progress=report)
    print(f"Recipe index ready: {stats.row_count:,} recipes")
    print(f"Index path: {stats.index_path}")


if __name__ == "__main__":
    main()
