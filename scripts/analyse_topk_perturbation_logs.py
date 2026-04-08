from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse Top-K perturbation logs from experiment result JSON files and "
            "export per-instance and feature-frequency summaries."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory to search recursively for *topkfeatures_result.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=("Directory to write outputs. Default: <results-dir>/topk_analysis"),
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help=(
            "Deduplicate likely duplicated Top-K log rows per file/instance. "
            "Applies exact-row dedupe then caps rows per instance to the modal count per file."  # noqa: E501
        ),
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def discover_topk_jsons(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*topkfeatures_result.json"))


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_group_from_run_dir(run_dir: str) -> str | None:
    # Expected pattern includes _TP_ or _FP_ etc.
    parts = run_dir.split("_")
    if len(parts) >= 3:
        token = parts[2].upper()
        if token in {"TP", "FP", "TN", "FN"}:
            return token
    return None


def _features_key(selected_features) -> tuple[str, ...]:
    if not isinstance(selected_features, list):
        return tuple()
    cleaned = [str(x) for x in selected_features if x is not None]
    return tuple(sorted(set(cleaned)))


def load_logs(topk_files: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []

    for file_path in topk_files:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})
        logs = metrics.get("PerturbationLog_with_ids", [])
        if not isinstance(logs, list):
            continue

        run_dir = file_path.parent.name
        sample_group = _sample_group_from_run_dir(run_dir)

        for idx, log in enumerate(logs):
            if not isinstance(log, dict):
                continue

            selected = log.get("selected_features", [])
            if not isinstance(selected, list):
                selected = []

            rows.append(
                {
                    "result_file": str(file_path),
                    "run_dir": run_dir,
                    "sample_group": sample_group,
                    "row_index": idx,
                    "instance_id": log.get("instance_id"),
                    "strategy": log.get("strategy"),
                    "requested_k": _safe_int(log.get("requested_k")),
                    "eligible_perturbable": _safe_int(log.get("eligible_perturbable")),
                    "realised_k": _safe_int(log.get("realised_k")),
                    "perturbed_features_count": _safe_int(
                        log.get("perturbed_features_count")
                    ),
                    "skipped_no_eligible_features": bool(
                        log.get("skipped_no_eligible_features", False)
                    ),
                    "selected_features": selected,
                }
            )

    return pd.DataFrame(rows)


def build_feature_frequency(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    exploded = df[["sample_group", "instance_id", "selected_features"]].explode(
        "selected_features"
    )
    exploded = exploded.rename(columns={"selected_features": "feature"})
    exploded = exploded.dropna(subset=["feature"])
    if exploded.empty:
        return pd.DataFrame()

    total_rows = len(df)
    total_instances = int(df["instance_id"].nunique(dropna=True))
    out_rows: list[dict] = []

    overall_counts = Counter(exploded["feature"].tolist())
    overall_unique_instances = (
        exploded.groupby("feature")["instance_id"].nunique().to_dict()
    )

    for feature, count in sorted(overall_counts.items(), key=lambda x: (-x[1], x[0])):
        n_inst_selected = int(overall_unique_instances.get(feature, 0))
        out_rows.append(
            {
                "sample_group": "ALL",
                "feature": feature,
                "selection_count": int(count),
                "selection_rate_over_logs": float(count) / float(total_rows),
                "unique_instances_selected": n_inst_selected,
                "n_instances_in_group": total_instances,
                "instance_frequency": (
                    float(n_inst_selected) / float(total_instances)
                    if total_instances > 0
                    else 0.0
                ),
            }
        )

    for group in sorted(exploded["sample_group"].dropna().unique()):
        sub = exploded[exploded["sample_group"] == group]
        group_total_rows = int((df["sample_group"] == group).sum())
        group_total_instances = int(
            df.loc[df["sample_group"] == group, "instance_id"].nunique(dropna=True)
        )
        group_counts = Counter(sub["feature"].tolist())
        group_unique_instances = (
            sub.groupby("feature")["instance_id"].nunique().to_dict()
        )

        for feature, count in sorted(group_counts.items(), key=lambda x: (-x[1], x[0])):
            n_inst_selected = int(group_unique_instances.get(feature, 0))
            out_rows.append(
                {
                    "sample_group": group,
                    "feature": feature,
                    "selection_count": int(count),
                    "selection_rate_over_logs": (
                        float(count) / float(group_total_rows)
                        if group_total_rows > 0
                        else 0.0
                    ),
                    "unique_instances_selected": n_inst_selected,
                    "n_instances_in_group": group_total_instances,
                    "instance_frequency": (
                        float(n_inst_selected) / float(group_total_instances)
                        if group_total_instances > 0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(out_rows)


def build_per_instance_feature_list(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    grouped = df.groupby(
        ["result_file", "sample_group", "run_dir", "instance_id"], dropna=False
    )
    for keys, sub in grouped:
        result_file, sample_group, run_dir, instance_id = keys
        feature_sets = sub["selected_features"].apply(_features_key)
        first_key = feature_sets.iloc[0] if len(feature_sets) else tuple()
        out_rows.append(
            {
                "result_file": result_file,
                "sample_group": sample_group,
                "run_dir": run_dir,
                "instance_id": instance_id,
                "n_log_rows": int(len(sub)),
                "n_distinct_feature_sets": int(feature_sets.nunique()),
                "selected_features": ",".join(first_key),
            }
        )

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["sample_group", "run_dir", "instance_id"])
    return out


def build_per_instance_feature_list_by_sample_group(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    grouped = df.groupby(["sample_group", "instance_id"], dropna=False)
    for keys, sub in grouped:
        sample_group, instance_id = keys
        feature_sets = sub["selected_features"].apply(_features_key)
        first_key = feature_sets.iloc[0] if len(feature_sets) else tuple()
        out_rows.append(
            {
                "sample_group": sample_group,
                "instance_id": instance_id,
                "n_log_rows": int(len(sub)),
                "n_result_files": int(sub["result_file"].nunique()),
                "n_distinct_feature_sets": int(feature_sets.nunique()),
                "selected_features": ",".join(first_key),
            }
        )

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["sample_group", "instance_id"])
    return out


def find_feature_consistency_errors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    grouped = df.groupby(["result_file", "sample_group", "instance_id"], dropna=False)
    for keys, sub in grouped:
        result_file, sample_group, instance_id = keys
        feature_set_counts = (
            sub["selected_features"]
            .apply(_features_key)
            .value_counts()
            .sort_values(ascending=False)
        )
        if len(feature_set_counts) <= 1:
            continue
        rows.append(
            {
                "result_file": result_file,
                "sample_group": sample_group,
                "instance_id": instance_id,
                "n_log_rows": int(len(sub)),
                "n_distinct_feature_sets": int(len(feature_set_counts)),
                "feature_sets_with_counts": " | ".join(
                    [f"{list(k)}:{int(v)}" for k, v in feature_set_counts.items()]
                ),
            }
        )
    return pd.DataFrame(rows)


def dedupe_logs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame()

    work = df.copy()
    work["selected_features_key"] = work["selected_features"].apply(
        lambda xs: tuple(xs) if isinstance(xs, list) else tuple()
    )

    exact_subset = [
        "result_file",
        "instance_id",
        "strategy",
        "requested_k",
        "eligible_perturbable",
        "realised_k",
        "perturbed_features_count",
        "skipped_no_eligible_features",
        "selected_features_key",
    ]
    before_exact = len(work)
    work = work.drop_duplicates(subset=exact_subset, keep="first")
    after_exact = len(work)

    trimmed_parts: list[pd.DataFrame] = []
    report_rows: list[dict] = []
    for result_file, sub in work.groupby("result_file", dropna=False):
        counts = sub.groupby("instance_id", dropna=False).size()
        if counts.empty:
            continue
        target = int(counts.mode().iloc[0])

        keep_frames: list[pd.DataFrame] = []
        over_limit_instances = 0
        rows_trimmed = 0
        for _, inst_rows in sub.groupby("instance_id", dropna=False):
            inst_rows_sorted = inst_rows.sort_values("row_index", kind="stable")
            n = len(inst_rows_sorted)
            if n > target:
                over_limit_instances += 1
                rows_trimmed += n - target
                inst_rows_sorted = inst_rows_sorted.head(target)
            keep_frames.append(inst_rows_sorted)

        trimmed = (
            pd.concat(keep_frames, ignore_index=True) if keep_frames else sub.copy()
        )
        trimmed_parts.append(trimmed)
        report_rows.append(
            {
                "result_file": result_file,
                "target_rows_per_instance_mode": target,
                "instances_over_mode": over_limit_instances,
                "rows_trimmed_to_mode": rows_trimmed,
            }
        )

    deduped = (
        pd.concat(trimmed_parts, ignore_index=True) if trimmed_parts else work.copy()
    )
    deduped = deduped.drop(columns=["selected_features_key"], errors="ignore")

    report_df = pd.DataFrame(report_rows)
    report_df = (
        report_df.sort_values("result_file") if not report_df.empty else report_df
    )
    summary = pd.DataFrame(
        [
            {
                "stage": "exact_row_dedupe",
                "rows_before": before_exact,
                "rows_after": after_exact,
                "rows_removed": before_exact - after_exact,
            },
            {
                "stage": "cap_to_modal_rows_per_instance",
                "rows_before": after_exact,
                "rows_after": len(deduped),
                "rows_removed": after_exact - len(deduped),
            },
        ]
    )
    report_df = pd.concat([summary, report_df], ignore_index=True)
    return deduped, report_df


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        pd.DataFrame().to_csv(
            output_dir / "topk_unique_instances_by_sample_group.csv", index=False
        )
        pd.DataFrame().to_csv(output_dir / "topk_feature_frequency.csv", index=False)
        return

    per_instance_by_group = build_per_instance_feature_list_by_sample_group(df)
    per_instance_by_group.to_csv(
        output_dir / "topk_unique_instances_by_sample_group.csv", index=False
    )

    freq = build_feature_frequency(df)
    freq.to_csv(output_dir / "topk_feature_frequency.csv", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else results_dir / "topk_analysis"
    )

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    topk_files = discover_topk_jsons(results_dir)
    if not topk_files:
        raise ValueError(f"No *topkfeatures_result.json found under: {results_dir}")

    logs_df = load_logs(topk_files)
    if args.dedupe:
        logs_df, _ = dedupe_logs(logs_df)

    consistency_errors = find_feature_consistency_errors(logs_df)
    write_outputs(logs_df, output_dir)

    per_file_counts: list[tuple[str, int, int]] = []
    for fpath in (
        sorted(logs_df["result_file"].dropna().unique().tolist())
        if not logs_df.empty
        else []
    ):
        sub = logs_df[logs_df["result_file"] == fpath]
        per_file_counts.append(
            (
                fpath,
                len(sub),
                int(sub["instance_id"].nunique(dropna=True)),
            )
        )

    if per_file_counts:
        print("Per-file Top-K log rows:")
        for fpath, n_rows_file, n_inst_file in per_file_counts:
            print(f"  {fpath}: rows={n_rows_file}, unique_instance_ids={n_inst_file}")
    if not consistency_errors.empty:
        for _, row in consistency_errors.head(20).iterrows():
            logging.error(
                "Feature inconsistency: file=%s sample_group=%s instance_id=%s sets=%s",
                row["result_file"],
                row["sample_group"],
                row["instance_id"],
                row["feature_sets_with_counts"],
            )
        raise ValueError(
            "Found instance IDs with inconsistent selected_features across entries. "
            "Run aborted."
        )

    n_rows = len(logs_df)
    n_instances = int(logs_df["instance_id"].nunique(dropna=True)) if n_rows else 0
    print(f"Processed Top-K files: {len(topk_files)}")
    print(f"Total Top-K log rows: {n_rows}")
    print(f"Unique instance IDs: {n_instances}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
