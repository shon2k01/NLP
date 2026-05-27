from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV_PATH = Path("outputs") / "song_feature_table.csv"
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
MEANS_BY_DECADE_PATH = OUTPUT_DIR / "feature_means_by_decade.csv"
TOP_FEATURES_PATH = OUTPUT_DIR / "top_decade_separating_features.csv"
CORRELATION_MATRIX_PATH = OUTPUT_DIR / "feature_correlation_matrix.csv"

METADATA_COLUMNS = ["song_name", "artist_name", "decade"]
EXPECTED_DECADE_ORDER = ["1980s", "2000s", "2020s"]
BOUNDED_NUMERIC_FEATURES = {
    "lexical_diversity",
    "chorus_repetition_score",
    "direct_address_score",
}
PLOT_FEATURES = [
    ("word_count", "Average word_count by decade", "avg_word_count_by_decade.png"),
    (
        "lexical_diversity",
        "Average lexical_diversity by decade",
        "avg_lexical_diversity_by_decade.png",
    ),
    (
        "repetition_ratio",
        "Average repetition_ratio by decade",
        "avg_repetition_ratio_by_decade.png",
    ),
    (
        "repeated_lines_ratio",
        "Average repeated_lines_ratio by decade",
        "avg_repeated_lines_ratio_by_decade.png",
    ),
    (
        "first_person_singular_ratio",
        "Average first_person_singular_ratio by decade",
        "avg_first_person_singular_ratio_by_decade.png",
    ),
    (
        "second_person_ratio",
        "Average second_person_ratio by decade",
        "avg_second_person_ratio_by_decade.png",
    ),
    (
        "love_words_ratio",
        "Average love_words_ratio by decade",
        "avg_love_words_ratio_by_decade.png",
    ),
    (
        "modern_slang_words_ratio",
        "Average modern_slang_words_ratio by decade",
        "avg_modern_slang_words_ratio_by_decade.png",
    ),
    (
        "old_israeli_words_ratio",
        "Average old_israeli_words_ratio by decade",
        "avg_old_israeli_words_ratio_by_decade.png",
    ),
]


def load_feature_table(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header row: {input_path}")
        rows = list(reader)
    return reader.fieldnames, rows


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def is_finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def detect_numeric_feature_columns(
    fieldnames: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    non_numeric_columns: list[str] = []

    for column in fieldnames:
        if column in METADATA_COLUMNS:
            continue
        non_empty_values = [
            row.get(column, "")
            for row in rows
            if not is_missing_value(row.get(column, ""))
        ]
        if non_empty_values and all(parse_float(value) is not None for value in non_empty_values):
            numeric_columns.append(column)
        else:
            non_numeric_columns.append(column)

    return numeric_columns, non_numeric_columns


def build_numeric_rows(
    rows: list[dict[str, str]], numeric_columns: list[str]
) -> list[dict[str, object]]:
    numeric_rows: list[dict[str, object]] = []
    for row in rows:
        converted: dict[str, object] = {
            "song_name": row.get("song_name", ""),
            "artist_name": row.get("artist_name", ""),
            "decade": row.get("decade", ""),
        }
        for column in numeric_columns:
            converted[column] = parse_float(row.get(column))
        numeric_rows.append(converted)
    return numeric_rows


def get_decade_order(rows: list[dict[str, str]]) -> list[str]:
    observed_decades = {row.get("decade", "") for row in rows if row.get("decade", "")}
    ordered = [decade for decade in EXPECTED_DECADE_ORDER if decade in observed_decades]
    remaining = sorted(observed_decades - set(ordered))
    return ordered + remaining


def summarize_missing_values(
    fieldnames: list[str], rows: list[dict[str, str]]
) -> dict[str, int]:
    return {
        column: sum(1 for row in rows if is_missing_value(row.get(column)))
        for column in fieldnames
    }


def column_values(
    numeric_rows: list[dict[str, object]], column: str
) -> list[float | None]:
    return [row.get(column) for row in numeric_rows]  # type: ignore[list-item]


def finite_values(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if is_finite_number(value)]


def find_constant_columns(
    numeric_rows: list[dict[str, object]], numeric_columns: list[str]
) -> list[str]:
    constant_columns: list[str] = []
    for column in numeric_columns:
        values = column_values(numeric_rows, column)
        if any(value is None or (isinstance(value, float) and not math.isfinite(value)) for value in values):
            continue
        unique_values = {value for value in values}
        if len(unique_values) == 1:
            constant_columns.append(column)
    return constant_columns


def find_near_constant_columns(
    numeric_rows: list[dict[str, object]], numeric_columns: list[str], threshold: float = 0.95
) -> list[dict[str, object]]:
    near_constant_columns: list[dict[str, object]] = []
    row_count = len(numeric_rows)
    if row_count == 0:
        return near_constant_columns

    for column in numeric_columns:
        values = column_values(numeric_rows, column)
        if any(value is None for value in values):
            continue
        counts = Counter(values)
        dominant_value, dominant_count = counts.most_common(1)[0]
        dominant_ratio = dominant_count / row_count
        if dominant_ratio > threshold:
            near_constant_columns.append(
                {
                    "column": column,
                    "dominant_value": dominant_value,
                    "dominant_ratio": dominant_ratio,
                }
            )
    return near_constant_columns


def find_infinite_value_columns(
    numeric_rows: list[dict[str, object]], numeric_columns: list[str]
) -> list[dict[str, object]]:
    columns_with_infinite_values: list[dict[str, object]] = []
    for column in numeric_columns:
        infinite_count = sum(
            1
            for value in column_values(numeric_rows, column)
            if isinstance(value, float) and math.isinf(value)
        )
        if infinite_count:
            columns_with_infinite_values.append({"column": column, "infinite_count": infinite_count})
    return columns_with_infinite_values


def get_expected_upper_bound(column: str) -> float | None:
    if column == "we_vs_i_ratio":
        return None
    if column.endswith("_ratio") or column in BOUNDED_NUMERIC_FEATURES:
        return 1.0
    return None


def get_quartiles(values: list[float]) -> tuple[float, float]:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def find_suspiciously_high_max_columns(
    numeric_rows: list[dict[str, object]], numeric_columns: list[str]
) -> list[dict[str, object]]:
    suspicious_columns: list[dict[str, object]] = []

    for column in numeric_columns:
        values = finite_values(column_values(numeric_rows, column))
        if not values:
            continue

        max_value = max(values)
        reasons: list[str] = []
        tukey_threshold: float | None = None

        expected_upper_bound = get_expected_upper_bound(column)
        if expected_upper_bound is not None and max_value > expected_upper_bound + 1e-9:
            reasons.append(f"max exceeds expected upper bound {expected_upper_bound}")

        if len(values) >= 4:
            q1, q3 = get_quartiles(sorted(values))
            iqr = q3 - q1
            tukey_threshold = q3 + (3.0 * iqr)
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


def safe_mean(values: Iterable[float | None]) -> float:
    clean_values = finite_values(values)
    return statistics.fmean(clean_values) if clean_values else math.nan


def compute_means_by_decade(
    numeric_rows: list[dict[str, object]], numeric_columns: list[str], decades: list[str]
) -> dict[str, dict[str, float]]:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in numeric_rows:
        grouped_rows[str(row.get("decade", ""))].append(row)

    means_by_decade: dict[str, dict[str, float]] = {}
    for decade in decades:
        means_by_decade[decade] = {
            column: safe_mean(row.get(column) for row in grouped_rows.get(decade, []))
            for column in numeric_columns
        }
    return means_by_decade


def write_means_by_decade(
    output_path: Path,
    means_by_decade: dict[str, dict[str, float]],
    numeric_columns: list[str],
    decades: list[str],
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["decade"] + numeric_columns)
        writer.writeheader()
        for decade in decades:
            row = {"decade": decade}
            row.update(means_by_decade[decade])
            writer.writerow(row)


def build_top_decade_feature_rows(
    means_by_decade: dict[str, dict[str, float]], numeric_columns: list[str], decades: list[str]
) -> list[dict[str, float | str]]:
    ranked_rows: list[dict[str, float | str]] = []
    for column in numeric_columns:
        decade_means = [means_by_decade[decade][column] for decade in decades]
        finite_means = [value for value in decade_means if math.isfinite(value)]
        mean_difference = max(finite_means) - min(finite_means) if finite_means else math.nan

        row: dict[str, float | str] = {
            "feature": column,
            "mean_difference": mean_difference,
        }
        for decade in decades:
            row[f"{decade}_mean"] = means_by_decade[decade][column]
        ranked_rows.append(row)

    ranked_rows.sort(
        key=lambda item: float(item["mean_difference"])
        if isinstance(item["mean_difference"], (int, float)) and math.isfinite(float(item["mean_difference"]))
        else float("-inf"),
        reverse=True,
    )
    return ranked_rows


def write_ranked_feature_table(
    output_path: Path, ranked_rows: list[dict[str, float | str]], decades: list[str]
) -> None:
    fieldnames = ["feature", "mean_difference"] + [f"{decade}_mean" for decade in decades]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)


def pearson_correlation(xs: list[float | None], ys: list[float | None]) -> float:
    paired_values = [
        (x, y)
        for x, y in zip(xs, ys)
        if is_finite_number(x) and is_finite_number(y)
    ]
    if len(paired_values) < 2:
        return math.nan

    x_values = [pair[0] for pair in paired_values]
    y_values = [pair[1] for pair in paired_values]

    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in paired_values)
    x_ss = sum((x - x_mean) ** 2 for x in x_values)
    y_ss = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_ss * y_ss)

    if denominator == 0:
        return math.nan
    return numerator / denominator


def compute_correlation_matrix(
    numeric_rows: list[dict[str, object]], numeric_columns: list[str]
) -> dict[str, dict[str, float]]:
    correlation_matrix: dict[str, dict[str, float]] = {}
    column_cache = {column: column_values(numeric_rows, column) for column in numeric_columns}

    for column_x in numeric_columns:
        correlation_matrix[column_x] = {}
        for column_y in numeric_columns:
            if column_x == column_y:
                correlation_matrix[column_x][column_y] = 1.0
            else:
                correlation_matrix[column_x][column_y] = pearson_correlation(
                    column_cache[column_x],
                    column_cache[column_y],
                )
    return correlation_matrix


def write_correlation_matrix(
    output_path: Path, correlation_matrix: dict[str, dict[str, float]], numeric_columns: list[str]
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["feature"] + numeric_columns)
        for row_feature in numeric_columns:
            writer.writerow(
                [row_feature] + [correlation_matrix[row_feature][column] for column in numeric_columns]
            )


def create_bar_plot(
    decades: list[str],
    values: list[float],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(decades, values, color="#2c7fb8")
    plt.title(title)
    plt.xlabel("Decade")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_requested_plots(
    means_by_decade: dict[str, dict[str, float]], decades: list[str], figures_dir: Path
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for feature_name, title, file_name in PLOT_FEATURES:
        values = [means_by_decade[decade][feature_name] for decade in decades]
        output_path = figures_dir / file_name
        create_bar_plot(decades, values, title, output_path)
        saved_paths.append(output_path)
    return saved_paths


def print_missing_values_summary(missing_values_by_column: dict[str, int]) -> None:
    total_missing = sum(missing_values_by_column.values())
    non_zero_missing = {
        column: count
        for column, count in missing_values_by_column.items()
        if count > 0
    }

    print(f"Missing values (total): {total_missing}")
    if non_zero_missing:
        print("Missing values by column:")
        for column, count in non_zero_missing.items():
            print(f"  - {column}: {count}")
    else:
        print("Missing values by column: none")


def print_songs_per_decade(rows: list[dict[str, str]], decades: list[str]) -> None:
    counts = Counter(row.get("decade", "") for row in rows)
    print("Songs per decade:")
    for decade in decades:
        print(f"  - {decade}: {counts.get(decade, 0)}")


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


def print_infinite_columns(columns: list[dict[str, object]]) -> None:
    print("Columns with infinite values:")
    if columns:
        for item in columns:
            print(f"  - {item['column']} | infinite_count={item['infinite_count']}")
    else:
        print("  - none")


def print_suspicious_columns(columns: list[dict[str, object]]) -> None:
    print("Numeric columns with suspiciously high max values:")
    if columns:
        for item in columns:
            threshold_text = (
                "n/a"
                if item["tukey_threshold"] is None
                else f"{float(item['tukey_threshold']):.6f}"
            )
            reason_text = "; ".join(item["reasons"])
            print(
                "  - "
                f"{item['column']} | max={float(item['max_value']):.6f} "
                f"| tukey_threshold={threshold_text} | {reason_text}"
            )
    else:
        print("  - none")


def print_top_decade_separating_features(ranked_rows: list[dict[str, float | str]], top_n: int = 15) -> None:
    print(f"Top {top_n} features by decade mean difference:")
    for row in ranked_rows[:top_n]:
        print(f"  - {row['feature']}: {float(row['mean_difference']):.6f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = load_feature_table(INPUT_CSV_PATH)
    numeric_columns, non_numeric_columns = detect_numeric_feature_columns(fieldnames, rows)
    numeric_rows = build_numeric_rows(rows, numeric_columns)
    decades = get_decade_order(rows)

    missing_values_by_column = summarize_missing_values(fieldnames, rows)
    constant_columns = find_constant_columns(numeric_rows, numeric_columns)
    near_constant_columns = find_near_constant_columns(numeric_rows, numeric_columns)
    infinite_columns = find_infinite_value_columns(numeric_rows, numeric_columns)
    suspicious_columns = find_suspiciously_high_max_columns(numeric_rows, numeric_columns)
    means_by_decade = compute_means_by_decade(numeric_rows, numeric_columns, decades)
    ranked_rows = build_top_decade_feature_rows(means_by_decade, numeric_columns, decades)
    correlation_matrix = compute_correlation_matrix(numeric_rows, numeric_columns)
    figure_paths = save_requested_plots(means_by_decade, decades, FIGURES_DIR)

    write_means_by_decade(MEANS_BY_DECADE_PATH, means_by_decade, numeric_columns, decades)
    write_ranked_feature_table(TOP_FEATURES_PATH, ranked_rows, decades)
    write_correlation_matrix(CORRELATION_MATRIX_PATH, correlation_matrix, numeric_columns)

    print(f"Table shape: ({len(rows)}, {len(fieldnames)})")
    print(f"Number of metadata columns: {sum(1 for column in fieldnames if column in METADATA_COLUMNS)}")
    print(f"Number of numeric feature columns: {len(numeric_columns)}")
    if non_numeric_columns:
        print(f"Non-numeric feature columns skipped: {', '.join(non_numeric_columns)}")
    print_songs_per_decade(rows, decades)
    print_missing_values_summary(missing_values_by_column)
    print_constant_columns(constant_columns)
    print_near_constant_columns(near_constant_columns)
    print_infinite_columns(infinite_columns)
    print_suspicious_columns(suspicious_columns)
    print_top_decade_separating_features(ranked_rows, top_n=15)
    print(f"Saved mean-by-decade table: {MEANS_BY_DECADE_PATH}")
    print(f"Saved ranked decade-separating features: {TOP_FEATURES_PATH}")
    print(f"Saved feature correlation matrix: {CORRELATION_MATRIX_PATH}")
    print(f"Saved figures: {len(figure_paths)} in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
