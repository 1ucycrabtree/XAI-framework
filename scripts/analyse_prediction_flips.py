from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERTURBATION_NAME_MAP = {
    "localgaussiannoise": "Noise",
    "directionaldrift": "Drift",
    "topkfeatures": "K",
    "noise": "Noise",
    "drift": "Drift",
    "k": "K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find instance IDs with prediction flips per sample group from result JSONs."  # noqa: E501
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory to search recursively for *_result.json",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="RelativeInputStability",
        choices=["RelativeInputStability", "RankBiasedOverlap", "SignConsistencyRate"],
        help="Metric family to read preserved IDs from (<metric>_with_ids).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: <results-dir>/prediction_flip_analysis)",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def discover_result_jsons(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_result.json"))


def parse_sample_group(run_dir_name: str) -> str:
    parts = run_dir_name.split("_")
    if len(parts) > 2:
        token = parts[2].upper()
        if token in {"TP", "FP", "TN", "FN"}:
            return token
    return "UNKNOWN"


def parse_perturbation(run_dir_name: str, file_name: str) -> str:
    parts = run_dir_name.split("_")
    candidate = parts[3] if len(parts) > 3 else None
    if not candidate:
        stem_parts = Path(file_name).stem.split("_")
        candidate = stem_parts[1] if len(stem_parts) > 1 else "unknown"
    token = str(candidate).lower().replace("_", "")
    return PERTURBATION_NAME_MAP.get(token, str(candidate))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def analyse_file(result_file: Path, metric_name: str) -> tuple[list[dict], dict]:
    with result_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    run_dir = result_file.parent.name
    sample_group = parse_sample_group(run_dir)
    perturbation = parse_perturbation(run_dir, result_file.name)
    metric_key = f"{metric_name}_with_ids"

    preserved_items = metrics.get(metric_key, [])
    if not isinstance(preserved_items, list):
        preserved_items = []

    preserved_counts: Counter = Counter()
    for item in preserved_items:
        if not isinstance(item, dict):
            continue
        instance_id = item.get("instance_id")
        if instance_id is None:
            continue
        preserved_counts[str(instance_id)] += 1

    # Best source for total perturbations per instance (all rows): perturbation logs when present.
    log_items = metrics.get("PerturbationLog_with_ids", [])
    total_counts: Counter = Counter()
    if isinstance(log_items, list) and log_items:
        for item in log_items:
            if not isinstance(item, dict):
                continue
            instance_id = item.get("instance_id")
            if instance_id is None:
                continue
            total_counts[str(instance_id)] += 1

    expected_per_instance = None
    if total_counts:
        # Use modal total count per instance from logs
        expected_per_instance = Counter(total_counts.values()).most_common(1)[0][0]
    elif preserved_counts:
        # Fallback: infer from preserved rows only (cannot detect fully-flipped missing IDs)
        expected_per_instance = max(preserved_counts.values())
        for iid in preserved_counts:
            total_counts[iid] = expected_per_instance
    else:
        expected_per_instance = 0

    rows_all: list[dict] = []
    n_changed_instances = 0
    for iid, total in total_counts.items():
        preserved = preserved_counts.get(iid, 0)
        changed = max(int(total) - int(preserved), 0)
        if changed > 0:
            n_changed_instances += 1
        rows_all.append(
            {
                "sample_group": sample_group,
                "perturbation": perturbation,
                "instance_id": iid,
                "total_perturbation_rows": int(total),
                "preserved_rows": int(preserved),
                "changed_rows": int(changed),
                "changed_rate": float(changed) / float(total) if total else 0.0,
            }
        )

    summary = {
        "sample_group": sample_group,
        "perturbation": perturbation,
        "metric_used": metric_name,
        "n_instances_with_total_counts": int(len(total_counts)),
        "n_instances_with_any_change": int(n_changed_instances),
        "expected_rows_per_instance": int(expected_per_instance or 0),
        "has_perturbation_logs": int(
            bool(total_counts and isinstance(log_items, list) and len(log_items) > 0)
        ),
    }
    return rows_all, summary


def main() -> None:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else results_dir / "prediction_flip_analysis"
    )

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    result_files = discover_result_jsons(results_dir)
    if not result_files:
        raise ValueError(f"No *_result.json found under: {results_dir}")

    all_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for rf in result_files:
        rows, summary = analyse_file(rf, args.metric)
        all_rows.extend(rows)
        all_summary_rows.append(summary)

    # Split changed-instance outputs by sample group and perturbation.
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_perturbation: dict[str, list[dict]] = defaultdict(list)
    by_perturbation_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        by_group[str(row["sample_group"])].append(row)
        by_perturbation[str(row["perturbation"])].append(row)
        by_perturbation_group[
            (str(row["perturbation"]), str(row["sample_group"]))
        ].append(row)

    fieldnames_changed = [
        "sample_group",
        "perturbation",
        "instance_id",
        "total_perturbation_rows",
        "preserved_rows",
        "changed_rows",
        "changed_rate",
    ]
    fieldnames_summary = [
        "sample_group",
        "perturbation",
        "metric_used",
        "n_instances_with_total_counts",
        "n_instances_with_any_change",
        "expected_rows_per_instance",
        "has_perturbation_logs",
    ]

    write_csv(output_dir / "changed_instances_all.csv", all_rows, fieldnames_changed)
    write_csv(
        output_dir / "flip_summary_by_file.csv", all_summary_rows, fieldnames_summary
    )
    for group, rows in by_group.items():
        write_csv(
            output_dir / f"changed_instances_{group}.csv", rows, fieldnames_changed
        )
    for pert, rows in by_perturbation.items():
        safe_pert = str(pert).replace(" ", "_")
        write_csv(
            output_dir / f"changed_instances_{safe_pert}.csv",
            rows,
            fieldnames_changed,
        )
    for (pert, group), rows in by_perturbation_group.items():
        safe_pert = str(pert).replace(" ", "_")
        write_csv(
            output_dir / f"changed_instances_{safe_pert}_{group}.csv",
            rows,
            fieldnames_changed,
        )

    print(f"Analysed result files: {len(result_files)}")
    print(
        f"Files with any changed instances: {sum(1 for r in all_summary_rows if r['n_instances_with_any_change'] > 0)}"  # noqa: E501
    )
    print(f"Total instance rows (includes unchanged): {len(all_rows)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
