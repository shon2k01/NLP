"""Leakage-safe Hebrew song decade classification experiments.

This script repeats the previous classification experiment set, but keeps all
learned preprocessing inside sklearn Pipeline/ColumnTransformer objects so
cross-validation fits TF-IDF vectorizers and scalers only on each training fold.

Usage:
    python run_classification_experiments_corrected.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
FEATURE_TABLE_PATH = BASE_DIR / "outputs" / "song_feature_table_with_dicta.csv"
CORPUS_PATH = BASE_DIR / "israeli_songs_corpus.csv"
OLD_RESULTS_PATH = BASE_DIR / "outputs" / "classification_results_all.csv"
CORRECTED_RESULTS_PATH = BASE_DIR / "outputs" / "classification_results_corrected.csv"
CORRECTED_SUMMARY_PATH = BASE_DIR / "outputs" / "classification_summary_corrected.json"

RANDOM_STATE = 42
TARGET_COLUMN = "decade"
LYRICS_COLUMN = "lyrics"
METADATA_COLUMNS = ["song_name", "artist_name", TARGET_COLUMN]
NON_PREDICTIVE_COLUMNS = METADATA_COLUMNS + [LYRICS_COLUMN]

BASE_FEATURE_GROUPS: dict[str, list[str]] = {
    "basic_surface": [
        "word_count",
        "unique_word_count",
        "char_count",
        "line_count",
        "avg_words_per_line",
        "median_words_per_line",
        "max_words_per_line",
        "min_words_per_line",
        "avg_word_length",
        "lexical_diversity",
    ],
    "repetition": [
        "repetition_ratio",
        "repeated_lines_count",
        "repeated_lines_ratio",
        "unique_lines_count",
        "most_repeated_line_count",
        "chorus_repetition_score",
        "repeated_words_count",
        "repeated_words_ratio",
        "most_common_word_count",
        "most_common_word_ratio",
    ],
    "pronoun_person": [
        "first_person_singular_count",
        "first_person_singular_ratio",
        "first_person_plural_count",
        "first_person_plural_ratio",
        "second_person_count",
        "second_person_ratio",
        "third_person_count",
        "third_person_ratio",
        "we_vs_i_ratio",
        "direct_address_score",
    ],
    "semantic_lexicon": [
        "love_words_ratio",
        "army_words_ratio",
        "nature_words_ratio",
        "religion_words_ratio",
        "party_words_ratio",
        "sadness_words_ratio",
        "nostalgia_words_ratio",
        "family_words_ratio",
        "place_words_ratio",
        "time_words_ratio",
        "modern_slang_words_ratio",
        "old_israeli_words_ratio",
    ],
    "dicta_pos": [
        "pos_noun_ratio",
        "pos_verb_ratio",
        "pos_adj_ratio",
        "pos_adv_ratio",
        "pos_pron_ratio",
        "pos_propn_ratio",
        "pos_adp_ratio",
        "pos_det_ratio",
        "pos_cconj_ratio",
        "pos_sconj_ratio",
        "pos_aux_ratio",
        "pos_num_ratio",
        "pos_intj_ratio",
    ],
    "dicta_morphology": [
        "tense_past_ratio",
        "tense_present_ratio",
        "tense_future_ratio",
        "verb_inf_ratio",
        "verb_participle_ratio",
        "imperative_ratio",
        "masculine_ratio",
        "feminine_ratio",
        "singular_ratio",
        "plural_ratio",
        "first_person_morph_ratio",
        "second_person_morph_ratio",
        "third_person_morph_ratio",
    ],
    "dicta_binyan": [
        "binyan_paal_ratio",
        "binyan_piel_ratio",
        "binyan_hifil_ratio",
        "binyan_hitpael_ratio",
        "binyan_nifal_ratio",
        "binyan_pual_ratio",
        "binyan_hufal_ratio",
    ],
    "dicta_lemma_root": [
        "lemma_diversity",
        "root_diversity",
        "dicta_unknown_token_ratio",
    ],
    "punctuation_formatting": [
        "punctuation_ratio",
        "question_mark_count",
        "exclamation_mark_count",
        "comma_count",
        "quote_count",
        "parenthesis_count",
        "english_char_ratio",
        "digit_count",
    ],
    "last_word": [
        "last_word_unique_count",
        "last_word_repetition_ratio",
        "avg_last_word_length",
    ],
}

REDUNDANT_FEATURES = {
    "repetition_ratio",
    "dicta_x_pos_count",
    "dicta_x_pos_ratio",
    "dicta_token_count",
    "dicta_unknown_token_count",
    "char_count",
    "chorus_repetition_score",
    "unique_lemma_count",
}


def load_data() -> pd.DataFrame:
    feature_df = pd.read_csv(FEATURE_TABLE_PATH, encoding="utf-8-sig")
    corpus_df = pd.read_csv(CORPUS_PATH, encoding="utf-8-sig")
    corpus_subset = corpus_df[METADATA_COLUMNS + [LYRICS_COLUMN]]

    merged_df = feature_df.merge(
        corpus_subset,
        on=METADATA_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    merged_df[LYRICS_COLUMN] = merged_df[LYRICS_COLUMN].fillna("")
    return merged_df


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns: list[str] = []
    for column in df.columns:
        if column in NON_PREDICTIVE_COLUMNS:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == df[column].notna().sum():
            df[column] = converted.astype(float)
            numeric_columns.append(column)
    return numeric_columns


def find_constant_or_all_zero_columns(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    removable_columns: list[str] = []
    for column in numeric_columns:
        series = df[column].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            removable_columns.append(column)
            continue
        if series.nunique() <= 1 or (series == 0).all():
            removable_columns.append(column)
    return sorted(set(removable_columns))


def build_feature_groups(all_numeric_columns: list[str]) -> dict[str, list[str]]:
    feature_groups = {group: list(columns) for group, columns in BASE_FEATURE_GROUPS.items()}
    feature_groups["all_numeric"] = list(all_numeric_columns)
    feature_groups["reduced_non_redundant"] = [
        column for column in all_numeric_columns if column not in REDUNDANT_FEATURES
    ]
    return feature_groups


def numeric_lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def numeric_svm_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=10000, random_state=RANDOM_STATE)),
        ]
    )


def numeric_models() -> dict[str, Any]:
    return {
        "Majority Class": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Stratified Random": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Logistic Regression": numeric_lr_pipeline(),
        "Linear SVM": numeric_svm_pipeline(),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def text_pipeline(ngram_range: tuple[int, int], classifier: Any) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            ("clf", classifier),
        ]
    )


def text_models(ngram_range: tuple[int, int]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": text_pipeline(
            ngram_range,
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "Linear SVM": text_pipeline(
            ngram_range,
            LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
        ),
        "Multinomial NB": text_pipeline(ngram_range, MultinomialNB()),
    }


def combined_pipeline(numeric_columns: list[str], classifier: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                LYRICS_COLUMN,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline([("preprocessor", preprocessor), ("clf", classifier)])


def combined_models(numeric_columns: list[str]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": combined_pipeline(
            numeric_columns,
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "Linear SVM": combined_pipeline(
            numeric_columns,
            LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
        ),
    }


def evaluate_model(
    feature_group: str,
    model_name: str,
    model: Any,
    x_data: Any,
    y: pd.Series,
    cv: StratifiedKFold,
    n_features: int,
) -> dict[str, Any]:
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro"),
    }
    scores = cross_validate(
        model,
        x_data,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )
    return {
        "Feature Group": feature_group,
        "Model": model_name,
        "Accuracy": float(scores["test_accuracy"].mean()),
        "Acc Std": float(scores["test_accuracy"].std()),
        "Macro F1": float(scores["test_macro_f1"].mean()),
        "F1 Std": float(scores["test_macro_f1"].std()),
        "N Features": int(n_features),
    }


def load_old_best_result() -> dict[str, Any] | None:
    if not OLD_RESULTS_PATH.exists():
        return None
    old_df = pd.read_csv(OLD_RESULTS_PATH)
    if old_df.empty or "Accuracy" not in old_df.columns:
        return None
    best_row = old_df.sort_values("Accuracy", ascending=False).iloc[0]
    return {
        "feature_group": str(best_row["Feature Group"]),
        "model": str(best_row["Model"]),
        "accuracy": float(best_row["Accuracy"]),
        "macro_f1": float(best_row["Macro F1"]),
    }


def best_result(results_df: pd.DataFrame, sort_column: str = "Accuracy") -> dict[str, Any]:
    best_row = results_df.sort_values(sort_column, ascending=False).iloc[0]
    return {
        "feature_group": str(best_row["Feature Group"]),
        "model": str(best_row["Model"]),
        "accuracy": float(best_row["Accuracy"]),
        "macro_f1": float(best_row["Macro F1"]),
    }


def print_top_results(results_df: pd.DataFrame, sort_column: str, title: str) -> None:
    columns = ["Feature Group", "Model", "Accuracy", "Acc Std", "Macro F1", "F1 Std", "N Features"]
    print(f"\n{title}")
    print(
        results_df.sort_values(sort_column, ascending=False)
        .head(10)[columns]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def main() -> None:
    df = load_data()
    y = df[TARGET_COLUMN]

    numeric_columns = detect_numeric_columns(df)
    removed_columns = find_constant_or_all_zero_columns(df, numeric_columns)
    usable_numeric_columns = [column for column in numeric_columns if column not in removed_columns]
    feature_groups = build_feature_groups(usable_numeric_columns)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, Any]] = []

    print(f"Loaded dataset: {df.shape[0]} songs")
    print(f"Numeric feature columns before removal: {len(numeric_columns)}")
    print(f"Removed constant/all-zero columns: {removed_columns if removed_columns else 'none'}")
    print(f"Numeric feature columns after removal: {len(usable_numeric_columns)}")

    for group_name, feature_columns in feature_groups.items():
        valid_columns = [column for column in feature_columns if column in usable_numeric_columns]
        if not valid_columns:
            continue
        x_numeric = df[valid_columns]
        for model_name, model in numeric_models().items():
            results.append(
                evaluate_model(
                    feature_group=group_name,
                    model_name=model_name,
                    model=model,
                    x_data=x_numeric,
                    y=y,
                    cv=cv,
                    n_features=len(valid_columns),
                )
            )
        print(f"  done: {group_name}")

    lyrics = df[LYRICS_COLUMN].fillna("")
    for group_name, ngram_range in {
        "tfidf_unigrams": (1, 1),
        "tfidf_bigrams": (1, 2),
    }.items():
        for model_name, model in text_models(ngram_range).items():
            results.append(
                evaluate_model(
                    feature_group=group_name,
                    model_name=model_name,
                    model=model,
                    x_data=lyrics,
                    y=y,
                    cv=cv,
                    n_features=5000,
                )
            )
        print(f"  done: {group_name}")

    combined_x = df[usable_numeric_columns + [LYRICS_COLUMN]].copy()
    for model_name, model in combined_models(usable_numeric_columns).items():
        results.append(
            evaluate_model(
                feature_group="numeric+tfidf_bigrams",
                model_name=model_name,
                model=model,
                x_data=combined_x,
                y=y,
                cv=cv,
                n_features=len(usable_numeric_columns) + 5000,
            )
        )
    print("  done: numeric+tfidf_bigrams")

    results_df = pd.DataFrame(results)
    results_df.to_csv(CORRECTED_RESULTS_PATH, index=False, encoding="utf-8-sig")

    old_best = load_old_best_result()
    corrected_best = best_result(results_df, sort_column="Accuracy")
    corrected_best_macro = best_result(results_df, sort_column="Macro F1")

    summary = {
        "dataset": {
            "n_songs": int(len(df)),
            "decades": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "numeric_features_before_constant_removal": int(len(numeric_columns)),
            "numeric_features_after_constant_removal": int(len(usable_numeric_columns)),
            "removed_constant_or_all_zero_columns": removed_columns,
        },
        "evaluation": {
            "method": "5-fold StratifiedKFold cross-validation",
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "metrics": ["accuracy", "macro_f1"],
            "total_experiments": int(len(results_df)),
            "leakage_correction": [
                "TF-IDF vectorizers are inside sklearn Pipeline.",
                "StandardScaler is inside sklearn Pipeline.",
                "Combined numeric + TF-IDF experiments use ColumnTransformer.",
                "No full-dataset fit_transform is applied before cross-validation.",
            ],
        },
        "old_best_result": old_best,
        "corrected_best_by_accuracy": corrected_best,
        "corrected_best_by_macro_f1": corrected_best_macro,
        "difference_old_minus_corrected_best_accuracy": (
            None if old_best is None else float(old_best["accuracy"] - corrected_best["accuracy"])
        ),
        "difference_old_minus_corrected_best_macro_f1": (
            None if old_best is None else float(old_best["macro_f1"] - corrected_best["macro_f1"])
        ),
        "top_10_by_accuracy": results_df.sort_values("Accuracy", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "top_10_by_macro_f1": results_df.sort_values("Macro F1", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }

    with CORRECTED_SUMMARY_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)

    print(f"\nTotal experiments: {len(results_df)}")
    print_top_results(results_df, "Accuracy", "Top 10 by accuracy")
    print_top_results(results_df, "Macro F1", "Top 10 by macro-F1")

    if old_best is None:
        print("\nOld best result: not available")
    else:
        print(
            "\nOld best result: "
            f"{old_best['feature_group']} + {old_best['model']} | "
            f"accuracy={old_best['accuracy']:.4f}, macro_f1={old_best['macro_f1']:.4f}"
        )

    print(
        "Corrected best result: "
        f"{corrected_best['feature_group']} + {corrected_best['model']} | "
        f"accuracy={corrected_best['accuracy']:.4f}, macro_f1={corrected_best['macro_f1']:.4f}"
    )

    if old_best is not None:
        print(
            "Difference old - corrected best: "
            f"accuracy={old_best['accuracy'] - corrected_best['accuracy']:.4f}, "
            f"macro_f1={old_best['macro_f1'] - corrected_best['macro_f1']:.4f}"
        )

    print(f"\nSaved corrected results: {CORRECTED_RESULTS_PATH}")
    print(f"Saved corrected summary: {CORRECTED_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
