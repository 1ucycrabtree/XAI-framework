import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import scipy.stats as ss
import csv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MetadataGenerator:
    def __init__(self):
        self.rules = {
            "sparse_cols": [],
            "redundant_cols": [],
            "all_dropped_cols": [],
            "correlation_groups": [],
        }
        self._redundant_set: set[str] = set()
        self._correlation_group_counter = 0
        self._corr_threshold = 0.90

    def identify_sparse_cols(self, df: pd.DataFrame) -> None:
        """Identify columns with >90% missing values."""
        sparsity = df.isnull().mean()
        self.rules["sparse_cols"] = sparsity[sparsity > 0.9].index.tolist()
        logging.info(f"Identified {len(self.rules['sparse_cols'])} sparse columns.")

    def _get_collinear(
        self, df: pd.DataFrame, cols: list[str], group_source: str
    ) -> None:
        """Internal method to identify collinear features."""
        if len(cols) < 2:
            return

        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

        # Analyse numeric features with Spearman correlation
        if len(num) > 1:
            self._prune_redundant(
                df=df,
                matrix=df[num].corr(method="spearman").abs(),
                method="spearman_abs",
                source=group_source,
            )

        # Analyse categorical features with Cramer's V
        if len(cat) > 1:
            n = len(cat)
            mat = pd.DataFrame(np.eye(n), index=cat, columns=cat)
            for i in range(n):
                for j in range(i + 1, n):
                    val = self.calculate_cramers_v(
                        df[cat[i]].fillna("NaN"), df[cat[j]].fillna("NaN")
                    )
                    mat.iloc[i, j] = mat.iloc[j, i] = val
            self._prune_redundant(
                df=df,
                matrix=mat,
                method="cramers_v",
                source=group_source,
            )

    def _prune_redundant(
        self, df: pd.DataFrame, matrix: pd.DataFrame, method: str, source: str
    ) -> None:
        """Identify correlated groups and keep a representative with less missingness."""
        logging.info(
            "Identifying redundant features with %s > %.2f...",
            method,
            self._corr_threshold,
        )

        cols = list(matrix.columns)
        adjacency: dict[str, set[str]] = {c: set() for c in cols}
        upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))

        for col in upper.columns:
            high_corr = upper[col][upper[col] > self._corr_threshold].index.tolist()
            for target in high_corr:
                adjacency[col].add(target)
                adjacency[target].add(col)

        visited = set()
        for col in cols:
            if col in visited or not adjacency[col]:
                continue

            stack = [col]
            component = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(adjacency[node] - visited)

            active_component = sorted(
                c
                for c in component
                if c not in self.rules["sparse_cols"] and c not in self._redundant_set
            )
            if len(active_component) < 2:
                continue

            representative = min(
                active_component,
                key=lambda c: self._representative_sort_key(df=df, col=c, method=method),
            )
            dropped = [c for c in active_component if c != representative]

            self._correlation_group_counter += 1
            self.rules["correlation_groups"].append(
                {
                    "group_id": f"corr_group_{self._correlation_group_counter:03d}",
                    "source": source,
                    "method": method,
                    "threshold": self._corr_threshold,
                    "features": active_component,
                    "representative_feature": representative,
                    "dropped_features": dropped,
                }
            )

            self._redundant_set.update(dropped)

        self.rules["redundant_cols"] = sorted(self._redundant_set)

    def _representative_sort_key(
        self, df: pd.DataFrame, col: str, method: str
    ) -> tuple[float, float, str]:
        """Rank representative candidates by missingness, then informational richness, then name.

        Lower tuple is better:
        1) lower missingness (ascending)
        2) richer signal (descending, encoded as negative value)
           - numerical groups: variance
           - categorical groups: cardinality (nunique)
        3) alphabetical order (ascending) as a tiebreaker
        """
        missingness = float(df[col].isnull().mean())

        if method == "spearman_abs":
            values = pd.to_numeric(df[col], errors="coerce")
            richness = float(values.var()) if values.notna().any() else 0.0
        elif method == "cramers_v":
            richness = float(df[col].nunique(dropna=True))
        else:
            # Safe fallback for any future association methods.
            richness = float(df[col].nunique(dropna=True))

        if np.isnan(richness):
            richness = 0.0

        return (missingness, -richness, col)

    def calculate_cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramer's V for categorical features."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        # Bias correction
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def _save_group_report(self, output_path: Path, timestamp: str) -> None:
        report_path = output_path / f"correlation_groups_{timestamp}.csv"
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "group_id",
                    "source",
                    "method",
                    "threshold",
                    "representative_feature",
                    "dropped_features",
                    "features",
                    "group_size",
                ],
            )
            writer.writeheader()
            for group in self.rules["correlation_groups"]:
                writer.writerow(
                    {
                        "group_id": group["group_id"],
                        "source": group["source"],
                        "method": group["method"],
                        "threshold": group["threshold"],
                        "representative_feature": group["representative_feature"],
                        "dropped_features": "|".join(group["dropped_features"]),
                        "features": "|".join(group["features"]),
                        "group_size": len(group["features"]),
                    }
                )
        logging.info("Saved correlation group report to %s", report_path)

    def _save_dropped_columns_report(self, output_path: Path, timestamp: str) -> None:
        dropped_path = output_path / f"dropped_columns_{timestamp}.txt"
        with open(dropped_path, "w") as f:
            f.write("# All removed columns (sparse + redundant)\n")
            for col in self.rules["all_dropped_cols"]:
                f.write(f"- {col}\n")
        logging.info("Saved dropped-columns report to %s", dropped_path)

    def run(self, data_path: Path, output_path: Path) -> None:
        logging.info("Loading IEEE-CIS dataset...")
        trans = pd.read_csv(data_path / "train_transaction.csv")
        ident = pd.read_csv(data_path / "train_identity.csv")
        df = pd.merge(
            trans, ident, on="TransactionID", how="left", validate="one_to_one"
        )
        del trans, ident

        logging.info("Applying temporal split (80% training)...")
        df = df.sort_values("TransactionDT")
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        logging.info(f"Training set size: {len(train_df)} rows.")

        self.identify_sparse_cols(train_df)

        null_groups = {}
        for col in train_df.columns:
            mask_key = tuple(train_df[col].isnull().values)
            if mask_key not in null_groups:
                null_groups[mask_key] = []
            null_groups[mask_key].append(col)

        logging.info(
            f"Identified {len(null_groups)} groups of columns with identical null patterns."
        )
        for i, group in enumerate(null_groups.values(), start=1):
            self._get_collinear(
                train_df, group, group_source=f"null_pattern_group_{i:03d}"
            )

        for prefix in ["C", "D", "M", "id_", "V"]:
            cols = [
                c
                for c in train_df.columns
                if c.startswith(prefix)
                and c not in self.rules["sparse_cols"] + self.rules["redundant_cols"]
            ]
            logging.info(
                f"Analysing {len(cols)} columns with prefix '{prefix}' for collinearity..."
            )
            self._get_collinear(train_df, cols, group_source=f"prefix_{prefix}")

        self.rules["redundant_cols"] = sorted(set(self.rules["redundant_cols"]))
        self.rules["all_dropped_cols"] = sorted(
            set(self.rules["sparse_cols"] + self.rules["redundant_cols"])
        )

        logging.info(
            "Identified {} columns to drop: {}".format(
                len(self.rules["all_dropped_cols"]),
                self.rules["all_dropped_cols"],
            )
        )

        logging.info("Saving metadata rules...")
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metadata_{timestamp}.json"
        with open(output_path / filename, "w") as f:
            json.dump(self.rules, f, indent=4)
        logging.info(f"Metadata saved to {output_path / filename}")
        self._save_group_report(output_path, timestamp)
        self._save_dropped_columns_report(output_path, timestamp)
