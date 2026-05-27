from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV_PATH = Path("outputs") / "song_feature_table_with_dicta.csv"
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures_dicta"
MEANS_BY_DECADE_PATH = OUTPUT_DIR / "dicta_enriched_feature_means_by_decade.csv"
TOP_MEAN_DIFF_PATH = OUTPUT_DIR / "dicta_enriched_top_decade_separating_features.csv"
ETA_SQUARED_PATH = OUTPUT_DIR / "dicta_enriched_feature_eta_squared.csv"
CORRELATION_MATRIX_PATH = OUTPUT_DIR / "dicta_enriched_feature_correlation_matrix.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "dicta_enriched_analysis_summary.json"

METADATA_COLUMNS = ["song_name", "artist_name", "decade"]
EXPECTED_DECADE_ORDER = ["1980s", "2000s", "2020s"]
BOUNDED_NUMERIC_FEATURES = {
    "lexical_diversity",
    "chorus_repetition_score",
    "direct_address_score",
    "lemma_diversity",
    "root_diversity",
}
PLOT_FEATURES = [
    ("pos_noun_ratio", "Average pos_noun_ratio by decade", "avg_pos_noun_ratio_by_decade.png"),
    ("pos_verb_ratio", "Average pos_verb_ratio by decade", "avg_pos_verb_ratio_by_decade.png"),
    ("pos_pron_ratio", "Average pos_pron_ratio by decade", "avg_pos_pron_ratio_by_decade.png"),
    ("tense_past_ratio", "Average tense_past_ratio by decade", "avg_tense_past_ratio_by_decade.png"),
    (
        "tense_future_ratio",
        "Average tense_future_ratio by decade",
        "avg_tense_future_ratio_by_decade.png",
    ),
    ("masculine_ratio", "Average masculine_ratio by decade", "avg_masculine_ratio_by_decade.png"),
    ("feminine_ratio", "Average feminine_ratio by decade", "avg_feminine_ratio_by_decade.png"),
    ("lemma_diversity", "Average lemma_diversity by decade", "avg_lemma_diversity_by_decade.png"),
    ("root_diversity", "Average root_diversity by decade", "avg_root_diversity_by_decade.png"),
    (
        "dicta_unknown_token_ratio",
        "Average dicta_unknown_token_ratio by decade",
        "avg_dicta_unknown_token_ratio_by_decade.png",
    ),
]


def load_enriched_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return pd.read_csv(input_path, encoding="utf-8-sig")


def get_decade_order(series: pd.Series) -> list[str]:
    observed = {str(value).strip() for value in series.dropna().tolist() if str(value).strip()}
    ordered = [decade for decade in EXPECTED_DECADE_ORDER if decade in observed]
    remaining = sorted(observed - set(ordered))
    return ordered + remaining


def coerce_numeric_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    working_df = df.copy()
    numeric_columns: list[str] = []
    non_numeric_columns: list[str] = []

    for column in working_df.columns:
        if column in METADATA_COLUMNS:
            continue
        converted = pd.to_numeric(working_df[column], errors="coerce")
        original_non_missing = working_df[column].notna().sum()
        converted_non_missing = converted.notna().sum()
        if original_non_missing == converted_non_missing and original_non_missing > 0:
            working_df[column] = converted.astype(float)
            numeric_columns.append(column)
        else:
            non_numeric_columns.append(column)

    return working_df, numeric_columns, non_numeric_columns


def summarize_missing_values(df: pd.DataFrame) -> dict[str, int]:
    return {column: int(df[column].isna().sum()) for column in df.columns}


def summarize_infinite_values(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, int]:
    infinite_counts: dict[str, int] = {}
    for column in numeric_columns:
        series = df[column]
        infinite_counts[column] = int(series.apply(lambda value: math.isinf(value) if pd.notna(value) else False).sum())
    return infinite_counts


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    finite_mask = series.notna() & ~series.apply(math.isinf)
    return series.loc[finite_mask]


def find_constant_columns(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    constant_columns: list[str] = []
    for column in numeric_columns:
        series = finite_series(df, column)
        if not series.empty and series.nunique(dropna=True) == 1:
            constant_columns.append(column)
    return constant_columns


def find_near_constant_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 0.95,
) -> list[dict[str, object]]:
    near_constant_columns: list[dict[str, object]] = []
    row_count = len(df)
    if row_count == 0:
        return near_constant_columns

    for column in numeric_columns:
        series = df[column]
        counts = Counter(series.dropna().tolist())
        if not counts:
            continue
        dominant_value, dominant_count = counts.most_common(1)[0]
        dominant_ratio = dominant_count / row_count
        if dominant_ratio > threshold:
            near_constant_columns.append(
                {
                    "column": column,
                    "dominant_value": float(dominant_value) if isinstance(dominant_value, (int, float)) else dominant_value,
                    "dominant_ratio": float(dominant_ratio),
                }
            )

    near_constant_columns.sort(key=lambda item: float(item["dominant_ratio"]), reverse=True)
    return near_constant_columns


def find_all_zero_columns(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    all_zero_columns: list[str] = []
    for column in numeric_columns:
        series = finite_series(df, column)
        if not series.empty and (series == 0).all():
            all_zero_columns.append(column)
    return all_zero_columns


def get_expected_upper_bound(column: str) -> float | None:
    if column == "we_vs_i_ratio":
        return None
    if column.endswith("_ratio") or column.endswith("_diversity") or column in BOUNDED_NUMERIC_FEATURES:
        return 1.0
    return None


def find_suspiciously_high_max_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> list[dict[str, object]]:
    suspicious_columns: list[dict[str, object]] = []

    for column in numeric_columns:
        series = finite_series(df, column)
        if series.empty:
            continue

        max_value = float(series.max())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        tukey_threshold = q3 + (3.0 * iqr)
        reasons: list[str] = []

        expected_upper_bound = get_expected_upper_bound(column)
        if expected_upper_bound is not None and max_value > expected_upper_bound + 1e-9:
            reasons.append(f"max exceeds expected upper bound {expected_upper_bound}")

        if max_value > tukey_threshold:
            reasons.append("max is above Q3 + 3*IQR")

        if reasons:
            suspicious_columns.append(
                {
                    "column": column,
                    "max_value": max_value,
                    "tukey_threshold": tukey_threshold,
                    "reasons": reasons,
                }
            )

    suspicious_columns.sort(key=lambda item: float(item["max_value"]), reverse=True)
    return suspicious_columns


def get_dicta_column_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    all_columns = list(columns)
    groups = {
        "dicta_prefix_columns": [column for column in all_columns if column.startswith("dicta_")],
        "pos_prefix_columns": [column for column in all_columns if column.startswith("pos_")],
        "tense_prefix_columns": [column for column in all_columns if column.startswith("tense_")],
        "verb_prefix_columns": [column for column in all_columns if column.startswith("verb_")],
        "binyan_prefix_columns": [column for column in all_columns if column.startswith("binyan_")],
        "morph_ratio_columns": [column for column in all_columns if column.endswith("_morph_ratio")],
        "lemma_root_diversity_columns": [
            column
            for column in [
                "lemma_count",
                "unique_lemma_count",
                "lemma_diversity",
                "root_count",
                "unique_root_count",
                "root_diversity",
            ]
            if column in all_columns
        ],
    }
    return groups


def compute_means_by_decade(
    df: pd.DataFrame,
    numeric_columns: list[str],
    decades: list[str],
) -> pd.DataFrame:
    means_df = df.groupby("decade", dropna=False)[numeric_columns].mean(numeric_only=True)
    return means_df.reindex(decades)


def build_mean_difference_table(
    means_by_decade: pd.DataFrame,
    numeric_columns: list[str],
    decades: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in numeric_columns:
        decade_means = means_by_decade[column]
        finite_means = decade_means.dropna()
        mean_difference = float(finite_means.max() - finite_means.min()) if not finite_means.empty else math.nan

        row: dict[str, float | str] = {
            "feature": column,
            "mean_difference": mean_difference,
        }
        for decade in decades:
            row[f"{decade}_mean"] = float(means_by_decade.loc[decade, column]) if decade in means_by_decade.index else math.nan
        rows.append(row)

    ranked_df = pd.DataFrame(rows)
    return ranked_df.sort_values("mean_difference", ascending=False, kind="stable").reset_index(drop=True)


def compute_eta_squared_table(
    df: pd.DataFrame,
    numeric_columns: list[str],
    decades: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for column in numeric_columns:
        clean_df = df[["decade", column]].dropna().copy()
        if clean_df.empty:
            eta_squared = math.nan
            between_ss = math.nan
            total_ss = math.nan
            group_means = {}
        else:
            overall_mean = float(clean_df[column].mean())
            total_ss = float(((clean_df[column] - overall_mean) ** 2).sum())
            between_ss = 0.0
            group_means = {}

            for decade in decades:
                group_series = clean_df.loc[clean_df["decade"] == decade, column]
                if group_series.empty:
                    group_means[decade] = math.nan
                    continue
                group_mean = float(group_series.mean())
                group_means[decade] = group_mean
                between_ss += float(len(group_series) * ((group_mean - overall_mean) ** 2))

            eta_squared = 0.0 if total_ss == 0.0 else between_ss / total_ss

        row: dict[str, float | str] = {
            "feature": column,
            "eta_squared": eta_squared,
            "between_group_sum_of_squares": between_ss,
            "total_sum_of_squares": total_ss,
        }
        for decade in decades:
            row[f"{decade}_mean"] = group_means.get(decade, math.nan)
        rows.append(row)

    eta_df = pd.DataFrame(rows)
    return eta_df.sort_values("eta_squared", ascending=False, kind="stable").reset_index(drop=True)


def compute_correlation_matrix(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    return df[numeric_columns].corr(method="pearson", numeric_only=True)


def extract_high_correlation_pairs(
    correlation_df: pd.DataFrame,
    threshold: float = 0.95,
) -> list[dict[str, float | str]]:
    pairs: list[dict[str, float | str]] = []
    columns = list(correlation_df.columns)

    for left_index, left_column in enumerate(columns):
        for right_column in columns[left_index + 1 :]:
            corr_value = correlation_df.loc[left_column, right_column]
            if pd.isna(corr_value):
                continue
            if abs(float(corr_value)) >= threshold:
                pairs.append(
                    {
                        "feature_1": left_column,
                        "feature_2": right_column,
                        "correlation": float(corr_value),
                        "abs_correlation": abs(float(corr_value)),
                    }
                )

    pairs.sort(key=lambda item: (float(item["abs_correlation"]), abs(float(item["correlation"]))), reverse=True)
    return pairs


def create_bar_plot(decades: list[str], values: list[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(decades, values, color="#2c7fb8")
    plt.title(title)
    plt.xlabel("Decade")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_requested_plots(means_by_decade: pd.DataFrame, decades: list[str]) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for feature_name, title, file_name in PLOT_FEATURES:
        if feature_name not in means_by_decade.columns:
            continue
        values = [float(means_by_decade.loc[decade, feature_name]) for decade in decades]
        output_path = FIGURES_DIR / file_name
        create_bar_plot(decades, values, title, output_path)
        saved_paths.append(output_path)

    return saved_paths


def print_missing_values_summary(missing_counts: dict[str, int]) -> None:
    total_missing = sum(missing_counts.values())
    non_zero_missing = {column: count for column, count in missing_counts.items() if count > 0}

    print(f"Missing values (total): {total_missing}")
    if non_zero_missing:
        print("Missing values by column:")
        for column, count in non_zero_missing.items():
            print(f"  - {column}: {count}")
    else:
        print("Missing values by column: none")


def print_infinite_values_summary(infinite_counts: dict[str, int]) -> None:
    total_infinite = sum(infinite_counts.values())
    non_zero_infinite = {column: count for column, count in infinite_counts.items() if count > 0}

    print(f"Infinite values (total): {total_infinite}")
    if non_zero_infinite:
        print("Infinite values by column:")
        for column, count in non_zero_infinite.items():
            print(f"  - {column}: {count}")
    else:
        print("Infinite values by column: none")


def print_songs_per_decade(df: pd.DataFrame, decades: list[str]) -> dict[str, int]:
    counts = Counter(str(value).strip() for value in df["decade"].dropna().tolist())
    print("Songs per decade:")
    for decade in decades:
        print(f"  - {decade}: {counts.get(decade, 0)}")
    return {decade: int(counts.get(decade, 0)) for decade in decades}


def print_constant_columns(columns: list[str]) -> None:
    print("Constant feature columns:")
    if columns:
        for column in columns:
            print(f"  - {column}")
    else:
        print("  - none")


def print_near_constant_columns(columns: list[dict[str, object]]) -> None:
    print("Near-constant feature columns (>95% one value):")
    if columns:
        for item in columns:
            print(
                "  - "
                f"{item['column']} | dominant_value={item['dominant_value']} "
                f"| share={float(item['dominant_ratio']):.3f}"
            )
    else:
        print("  - none")


def print_all_zero_columns(columns: list[str]) -> None:
    print("All-zero feature columns:")
    if columns:
        for column in columns:
            print(f"  - {column}")
    else:
        print("  - none")


def print_suspicious_columns(columns: list[dict[str, object]]) -> None:
    print("Numeric columns with suspiciously high max values:")
    if columns:
        for item in columns:
            print(
                "  - "
                f"{item['column']} | max={float(item['max_value']):.6f} "
                f"| tukey_threshold={float(item['tukey_threshold']):.6f} "
                f"| {'; '.join(item['reasons'])}"
            )
    else:
        print("  - none")


def print_column_group_inspection(groups: dict[str, list[str]]) -> None:
    print("Dicta-related column inspection:")
    for group_name, columns in groups.items():
        readable_name = re.sub(r"_+", " ", group_name).strip()
        print(f"  - {readable_name} ({len(columns)}): {', '.join(columns) if columns else 'none'}")


def print_top_feature_rows(
    ranked_df: pd.DataFrame,
    score_column: str,
    label: str,
    top_n: int = 20,
) -> list[dict[str, object]]:
    print(f"Top {top_n} features by {label}:")
    preview_rows: list[dict[str, object]] = []
    for _, row in ranked_df.head(top_n).iterrows():
        feature = str(row["feature"])
        score = float(row[score_column])
        print(f"  - {feature}: {score:.6f}")
        preview_rows.append({"feature": feature, score_column: score})
    return preview_rows


def print_high_correlation_pairs(pairs: list[dict[str, float | str]], limit: int = 30) -> list[dict[str, object]]:
    print("Highly correlated feature pairs (|correlation| >= 0.95):")
    if not pairs:
        print("  - none")
        return []

    preview: list[dict[str, object]] = []
    for item in pairs[:limit]:
        print(
            "  - "
            f"{item['feature_1']} <-> {item['feature_2']}: "
            f"{float(item['correlation']):.6f}"
        )
        preview.append(
            {
                "feature_1": str(item["feature_1"]),
                "feature_2": str(item["feature_2"]),
                "correlation": float(item["correlation"]),
                "abs_correlation": float(item["abs_correlation"]),
            }
        )

    if len(pairs) > limit:
        print(f"  ... truncated to first {limit} pairs")
    return preview


def serialize_near_constant_columns(columns: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for item in columns:
        serialized.append(
            {
                "column": str(item["column"]),
                "dominant_value": float(item["dominant_value"]) if isinstance(item["dominant_value"], (int, float)) else item["dominant_value"],
                "dominant_ratio": float(item["dominant_ratio"]),
            }
        )
    return serialized


def serialize_suspicious_columns(columns: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for item in columns:
        serialized.append(
            {
                "column": str(item["column"]),
                "max_value": float(item["max_value"]),
                "tukey_threshold": float(item["tukey_threshold"]),
                "reasons": [str(reason) for reason in item["reasons"]],
            }
        )
    return serialized


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_enriched_table(INPUT_CSV_PATH)
    working_df, numeric_columns, non_numeric_columns = coerce_numeric_feature_columns(df)
    decades = get_decade_order(working_df["decade"])

    missing_counts = summarize_missing_values(working_df)
    infinite_counts = summarize_infinite_values(working_df, numeric_columns)
    constant_columns = find_constant_columns(working_df, numeric_columns)
    near_constant_columns = find_near_constant_columns(working_df, numeric_columns)
    all_zero_columns = find_all_zero_columns(working_df, numeric_columns)
    suspicious_columns = find_suspiciously_high_max_columns(working_df, numeric_columns)
    dicta_column_groups = get_dicta_column_groups(numeric_columns)

    means_by_decade = compute_means_by_decade(working_df, numeric_columns, decades)
    mean_difference_df = build_mean_difference_table(means_by_decade, numeric_columns, decades)
    eta_squared_df = compute_eta_squared_table(working_df, numeric_columns, decades)
    correlation_df = compute_correlation_matrix(working_df, numeric_columns)
    high_correlation_pairs = extract_high_correlation_pairs(correlation_df, threshold=0.95)
    saved_plots = save_requested_plots(means_by_decade, decades)

    means_by_decade.to_csv(MEANS_BY_DECADE_PATH, encoding="utf-8-sig", index_label="decade")
    mean_difference_df.to_csv(TOP_MEAN_DIFF_PATH, encoding="utf-8-sig", index=False)
    eta_squared_df.to_csv(ETA_SQUARED_PATH, encoding="utf-8-sig", index=False)
    correlation_df.to_csv(CORRELATION_MATRIX_PATH, encoding="utf-8-sig", index_label="feature")

    print(f"Table shape: {working_df.shape}")
    print(f"Number of metadata columns: {sum(1 for column in working_df.columns if column in METADATA_COLUMNS)}")
    print(f"Number of numeric feature columns: {len(numeric_columns)}")
    if non_numeric_columns:
        print(f"Non-numeric feature columns skipped: {', '.join(non_numeric_columns)}")
    songs_per_decade = print_songs_per_decade(working_df, decades)
    print_missing_values_summary(missing_counts)
    print_infinite_values_summary(infinite_counts)
    print_constant_columns(constant_columns)
    print_near_constant_columns(near_constant_columns)
    print_all_zero_columns(all_zero_columns)
    print_suspicious_columns(suspicious_columns)
    print_column_group_inspection(dicta_column_groups)
    top_mean_difference_rows = print_top_feature_rows(
        mean_difference_df,
        score_column="mean_difference",
        label="raw mean difference",
        top_n=20,
    )
    top_eta_squared_rows = print_top_feature_rows(
        eta_squared_df,
        score_column="eta_squared",
        label="eta squared",
        top_n=20,
    )
    top_high_corr_rows = print_high_correlation_pairs(high_correlation_pairs, limit=30)
    print(f"Saved mean-by-decade table: {MEANS_BY_DECADE_PATH}")
    print(f"Saved raw mean-difference ranking: {TOP_MEAN_DIFF_PATH}")
    print(f"Saved eta-squared ranking: {ETA_SQUARED_PATH}")
    print(f"Saved correlation matrix: {CORRELATION_MATRIX_PATH}")
    print(f"Saved figures: {len(saved_plots)} in {FIGURES_DIR}")

    summary_payload = {
        "input_table_shape": [int(working_df.shape[0]), int(working_df.shape[1])],
        "metadata_column_count": int(sum(1 for column in working_df.columns if column in METADATA_COLUMNS)),
        "numeric_feature_column_count": int(len(numeric_columns)),
        "non_numeric_feature_columns": non_numeric_columns,
        "songs_per_decade": songs_per_decade,
        "missing_values_total": int(sum(missing_counts.values())),
        "missing_values_by_column": {column: int(count) for column, count in missing_counts.items()},
        "infinite_values_total": int(sum(infinite_counts.values())),
        "infinite_values_by_column": {column: int(count) for column, count in infinite_counts.items() if count > 0},
        "constant_columns": constant_columns,
        "near_constant_columns": serialize_near_constant_columns(near_constant_columns),
        "all_zero_columns": all_zero_columns,
        "suspicious_high_max_columns": serialize_suspicious_columns(suspicious_columns),
        "dicta_column_groups": dicta_column_groups,
        "top_20_mean_difference_features": top_mean_difference_rows,
        "top_20_eta_squared_features": top_eta_squared_rows,
        "top_high_correlation_pairs": top_high_corr_rows,
        "generated_files": {
            "means_by_decade_csv": str(MEANS_BY_DECADE_PATH),
            "top_mean_difference_csv": str(TOP_MEAN_DIFF_PATH),
            "eta_squared_csv": str(ETA_SQUARED_PATH),
            "correlation_matrix_csv": str(CORRELATION_MATRIX_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
            "figure_directory": str(FIGURES_DIR),
            "figure_files": [str(path) for path in saved_plots],
        },
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(summary_payload, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
