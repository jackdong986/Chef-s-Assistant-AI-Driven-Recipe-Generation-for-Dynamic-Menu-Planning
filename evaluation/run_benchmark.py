"""Run fixed recipe-generation cases and record constraint accuracy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gradio_app import generate_recipe_record  # noqa: E402
from recipe_quality import quality_score, validate_recipe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("benchmark_cases.json"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only a named case; repeat this flag for more than one case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "evaluation" / "latest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_id:
        requested_ids = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested_ids]
    if args.limit > 0:
        cases = cases[: args.limit]
    results = []
    for position, case in enumerate(cases, 1):
        print(f"[{position}/{len(cases)}] {case['id']}")
        started = time.perf_counter()
        recipe, _, audited = generate_recipe_record(
            case["request"], case["servings"], case["dietary"]
        )
        issues = validate_recipe(
            recipe,
            cuisines=case.get("cuisines", []),
            dietary_notes=case["dietary"],
            servings=case["servings"],
        )
        results.append(
            {
                "id": case["id"],
                "quality_score": quality_score(issues),
                "passed": not any(issue.severity == "error" for issue in issues),
                "audited": audited,
                "latency_seconds": round(time.perf_counter() - started, 2),
                "issues": [asdict(issue) for issue in issues],
                "recipe": recipe,
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "pass_rate": round(
            sum(result["passed"] for result in results) / max(1, len(results)), 4
        ),
        "average_quality_score": round(
            sum(result["quality_score"] for result in results) / max(1, len(results)),
            2,
        ),
        "average_latency_seconds": round(
            sum(result["latency_seconds"] for result in results) / max(1, len(results)),
            2,
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"Detailed report: {args.output}")


if __name__ == "__main__":
    main()
