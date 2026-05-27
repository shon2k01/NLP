from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


FEATURE_TABLE_PATH = Path("outputs") / "song_feature_table.csv"

# Edit these paths here if the Dicta files move.
DICTA_FILE_PATHS = {
    "1980s": Path("DictaAnalysis") / "dicta-80s" / "dicta_all_1980s.ud.txt",
    "2000s": Path("DictaAnalysis") / "dicta-00s" / "dicta_all_2000s.ud.txt",
    "2020s": Path("DictaAnalysis") / "dicta-20s" / "dicta_all_2020s.ud.txt",
}

OUTPUT_DIR = Path("outputs")
OUTPUT_FEATURE_TABLE_PATH = OUTPUT_DIR / "song_feature_table_with_dicta.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "dicta_feature_summary.json"
OUTPUT_UNMATCHED_FEATURE_TABLE_PATH = OUTPUT_DIR / "dicta_unmatched_feature_table_songs.csv"
OUTPUT_UNMATCHED_DICTA_PATH = OUTPUT_DIR / "dicta_unmatched_dicta_songs.csv"

METADATA_COLUMNS = ["song_name", "artist_name", "decade"]
MATCH_KEY_COLUMNS = ["match_song_name", "match_artist_name", "match_decade"]

CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
VERB_LIKE_UPOS = {"VERB", "AUX"}
TRACKED_POS_TAGS = [
    ("noun", "NOUN"),
    ("verb", "VERB"),
    ("adj", "ADJ"),
    ("adv", "ADV"),
    ("pron", "PRON"),
    ("propn", "PROPN"),
    ("adp", "ADP"),
    ("det", "DET"),
    ("cconj", "CCONJ"),
    ("sconj", "SCONJ"),
    ("aux", "AUX"),
    ("num", "NUM"),
    ("intj", "INTJ"),
]
BINYAN_TAGS = [
    ("paal", "PAAL"),
    ("piel", "PIEL"),
    ("hifil", "HIFIL"),
    ("hitpael", "HITPAEL"),
    ("nifal", "NIFAL"),
    ("pual", "PUAL"),
    ("hufal", "HUFAL"),
]

TITLE_LINE_RE = re.compile(r"^#\s*text\s*=\s*###\s*(.+?)\s*$")
ARTIST_CONTINUATION_RE = re.compile(r"^#\s*text\s*=\s*/\s*(.+?)\s*$")
INTEGER_TOKEN_ID_RE = re.compile(r"^\d+$")
HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")
WHITESPACE_RE = re.compile(r"\s+")
NON_MATCH_TEXT_RE = re.compile(r"[^A-Za-z0-9\u05D0-\u05EA']+")


def normalize_hebrew_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = HEBREW_DIACRITICS_RE.sub("", normalized)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u05F3": "'",
        "`": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u05F4": '"',
        "\u2010": " ",
        "\u2011": " ",
        "\u2012": " ",
        "\u2013": " ",
        "\u2014": " ",
        "\u05BE": " ",
        "_": " ",
        "/": " ",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    normalized = NON_MATCH_TEXT_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.lower()


def parse_feats(feats_text: object) -> dict[str, set[str]]:
    if feats_text is None or pd.isna(feats_text):
        return {}

    feats = str(feats_text).strip()
    if not feats or feats == "_":
        return {}

    parsed: dict[str, set[str]] = {}
    for chunk in feats.split("|"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value_text = chunk.split("=", 1)
        values = {
            value.strip()
            for value in value_text.split(",")
            if value.strip() and value.strip() != "_"
        }
        if values:
            parsed.setdefault(key.strip(), set()).update(values)
    return parsed


def is_integer_token_id(token_id: object) -> bool:
    return bool(INTEGER_TOKEN_ID_RE.fullmatch(str(token_id).strip()))


def parse_song_title_line(line: str) -> tuple[str, str] | None:
    match = TITLE_LINE_RE.match(line.strip())
    if not match:
        return None

    payload = match.group(1).strip()
    if " / " in payload:
        song_name, artist_name = payload.rsplit(" / ", 1)
    elif "/" in payload:
        song_name, artist_name = payload.rsplit("/", 1)
    else:
        return payload, ""

    return song_name.strip(), artist_name.strip()


def feature_zero_template() -> dict[str, int | float]:
    feature_names = [
        "dicta_token_count",
        "dicta_non_punct_token_count",
        "dicta_content_token_count",
        "lemma_count",
        "unique_lemma_count",
        "lemma_diversity",
        "root_count",
        "unique_root_count",
        "root_diversity",
        "dicta_unknown_token_count",
        "dicta_unknown_token_ratio",
        "dicta_x_pos_count",
        "dicta_x_pos_ratio",
    ]

    for prefix, _ in TRACKED_POS_TAGS:
        feature_names.extend([f"pos_{prefix}_count", f"pos_{prefix}_ratio"])

    feature_names.extend(
        [
            "tense_past_count",
            "tense_past_ratio",
            "tense_present_count",
            "tense_present_ratio",
            "tense_future_count",
            "tense_future_ratio",
            "verb_inf_count",
            "verb_inf_ratio",
            "verb_participle_count",
            "verb_participle_ratio",
            "imperative_count",
            "imperative_ratio",
            "masculine_count",
            "masculine_ratio",
            "feminine_count",
            "feminine_ratio",
            "singular_count",
            "singular_ratio",
            "plural_count",
            "plural_ratio",
            "first_person_morph_count",
            "first_person_morph_ratio",
            "second_person_morph_count",
            "second_person_morph_ratio",
            "third_person_morph_count",
            "third_person_morph_ratio",
        ]
    )

    for prefix, _ in BINYAN_TAGS:
        feature_names.extend([f"binyan_{prefix}_count", f"binyan_{prefix}_ratio"])

    return {name: 0 for name in feature_names}


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def extract_song_dicta_features(tokens: list[dict[str, object]]) -> dict[str, int | float]:
    features = feature_zero_template()

    non_punct_tokens = [token for token in tokens if token["upos"] != "PUNCT"]
    verb_like_tokens = [token for token in non_punct_tokens if token["upos"] in VERB_LIKE_UPOS]
    lemma_values: list[str] = []
    root_values: list[str] = []

    total_token_count = len(tokens)
    non_punct_token_count = len(non_punct_tokens)
    content_token_count = sum(1 for token in non_punct_tokens if token["upos"] in CONTENT_UPOS)
    verb_like_token_count = len(verb_like_tokens)

    features["dicta_token_count"] = total_token_count
    features["dicta_non_punct_token_count"] = non_punct_token_count
    features["dicta_content_token_count"] = content_token_count

    for prefix, upos in TRACKED_POS_TAGS:
        count = sum(1 for token in non_punct_tokens if token["upos"] == upos)
        features[f"pos_{prefix}_count"] = count
        features[f"pos_{prefix}_ratio"] = safe_ratio(count, non_punct_token_count)

    tense_specs = [
        ("tense_past", "Tense", "Past"),
        ("tense_present", "Tense", "Pres"),
        ("tense_future", "Tense", "Fut"),
    ]
    for feature_prefix, feats_key, target_value in tense_specs:
        count = sum(1 for token in verb_like_tokens if target_value in token["feats"].get(feats_key, set()))
        features[f"{feature_prefix}_count"] = count
        features[f"{feature_prefix}_ratio"] = safe_ratio(count, verb_like_token_count)

    verb_form_specs = [
        ("verb_inf", "VerbForm", "Inf"),
        ("verb_participle", "VerbForm", "Part"),
        ("imperative", "Mood", "Imp"),
    ]
    for feature_prefix, feats_key, target_value in verb_form_specs:
        count = sum(1 for token in verb_like_tokens if target_value in token["feats"].get(feats_key, set()))
        features[f"{feature_prefix}_count"] = count
        features[f"{feature_prefix}_ratio"] = safe_ratio(count, verb_like_token_count)

    non_punct_feature_specs = [
        ("masculine", "Gender", "Masc"),
        ("feminine", "Gender", "Fem"),
        ("singular", "Number", "Sing"),
        ("plural", "Number", "Plur"),
        ("first_person_morph", "Person", "1"),
        ("second_person_morph", "Person", "2"),
        ("third_person_morph", "Person", "3"),
    ]
    for feature_prefix, feats_key, target_value in non_punct_feature_specs:
        count = sum(1 for token in non_punct_tokens if target_value in token["feats"].get(feats_key, set()))
        features[f"{feature_prefix}_count"] = count
        features[f"{feature_prefix}_ratio"] = safe_ratio(count, non_punct_token_count)

    for prefix, binyan_value in BINYAN_TAGS:
        count = sum(
            1
            for token in verb_like_tokens
            if binyan_value in token["feats"].get("HebBinyan", set())
        )
        features[f"binyan_{prefix}_count"] = count
        features[f"binyan_{prefix}_ratio"] = safe_ratio(count, verb_like_token_count)

    unknown_token_count = 0
    x_pos_count = 0
    for token in non_punct_tokens:
        if token["upos"] == "X":
            x_pos_count += 1

        if token["is_unknown"]:
            unknown_token_count += 1

        if token["upos"] not in {"PUNCT", "X"} and not token["is_unknown"]:
            lemma = normalize_hebrew_text(token["lemma"])
            if lemma and lemma != "_":
                lemma_values.append(lemma)

            for root in token["feats"].get("HebRoot", set()):
                normalized_root = normalize_hebrew_text(root)
                if normalized_root and normalized_root != "_":
                    root_values.append(normalized_root)

    features["lemma_count"] = len(lemma_values)
    features["unique_lemma_count"] = len(set(lemma_values))
    features["lemma_diversity"] = safe_ratio(
        int(features["unique_lemma_count"]),
        int(features["lemma_count"]),
    )
    features["root_count"] = len(root_values)
    features["unique_root_count"] = len(set(root_values))
    features["root_diversity"] = safe_ratio(
        int(features["unique_root_count"]),
        int(features["root_count"]),
    )
    features["dicta_unknown_token_count"] = unknown_token_count
    features["dicta_unknown_token_ratio"] = safe_ratio(unknown_token_count, non_punct_token_count)
    features["dicta_x_pos_count"] = x_pos_count
    features["dicta_x_pos_ratio"] = safe_ratio(x_pos_count, non_punct_token_count)

    return features


def parse_ud_file(ud_path: Path, decade: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not ud_path.exists():
        raise FileNotFoundError(f"Dicta UD file not found: {ud_path}")

    parsed_songs: list[dict[str, object]] = []
    stats = Counter()
    current_song: dict[str, object] | None = None
    current_tokens: list[dict[str, object]] = []
    inside_title_sentence = False
    title_sentence_has_tokens = False

    def finalize_current_song() -> None:
        nonlocal current_song, current_tokens
        if current_song is None:
            return

        song_row = dict(current_song)
        song_row.update(extract_song_dicta_features(current_tokens))
        parsed_songs.append(song_row)
        current_song = None
        current_tokens = []

    with ud_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            title_info = parse_song_title_line(line)
            if title_info is not None:
                finalize_current_song()
                song_name, artist_name = title_info
                current_song = {
                    "song_name": song_name,
                    "artist_name": artist_name,
                    "decade": decade,
                    "source_file": str(ud_path),
                    "source_line": line_number,
                }
                inside_title_sentence = True
                title_sentence_has_tokens = False
                stats["song_title_lines"] += 1
                continue

            artist_continuation_match = ARTIST_CONTINUATION_RE.match(line.strip())
            if (
                artist_continuation_match is not None
                and current_song is not None
                and not str(current_song.get("artist_name", "")).strip()
            ):
                current_song["artist_name"] = artist_continuation_match.group(1).strip()
                inside_title_sentence = True
                title_sentence_has_tokens = False
                stats["artist_continuation_lines"] += 1
                continue

            if line.startswith("#"):
                continue

            if not line.strip():
                if inside_title_sentence and title_sentence_has_tokens:
                    inside_title_sentence = False
                continue

            if current_song is None:
                stats["token_rows_before_first_song"] += 1
                continue

            if inside_title_sentence:
                title_sentence_has_tokens = True
                stats["title_sentence_token_rows_skipped"] += 1
                continue

            parts = line.split("\t")
            token_id = parts[0].strip() if parts else ""

            if "-" in token_id:
                stats["multi_token_rows_skipped"] += 1
                continue
            if "." in token_id:
                stats["decimal_token_rows_skipped"] += 1
                continue
            if not is_integer_token_id(token_id):
                stats["non_integer_token_rows_skipped"] += 1
                continue

            if len(parts) < 6:
                stats["short_token_rows"] += 1

            form = parts[1].strip() if len(parts) > 1 else ""
            lemma = parts[2].strip() if len(parts) > 2 else ""
            upos = parts[3].strip().upper() if len(parts) > 3 else ""
            xpos = parts[4].strip().upper() if len(parts) > 4 else ""
            feats_raw = parts[5].strip() if len(parts) > 5 else ""
            feats = parse_feats(feats_raw)
            dicta_note_values = feats.get("DictaNote", set())

            token = {
                "id": token_id,
                "form": form,
                "lemma": lemma,
                "upos": upos,
                "xpos": xpos,
                "feats": feats,
                "is_unknown": (
                    upos == "X"
                    or xpos == "X"
                    or "Unknown" in dicta_note_values
                    or (not lemma or lemma == "_") and upos not in {"PUNCT", ""}
                ),
            }
            current_tokens.append(token)
            stats["processed_token_rows"] += 1

    finalize_current_song()
    stats["songs_parsed"] = len(parsed_songs)
    return parsed_songs, dict(stats)


def build_match_key_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    keyed_df = df.copy()
    keyed_df["match_song_name"] = keyed_df["song_name"].map(normalize_hebrew_text)
    keyed_df["match_artist_name"] = keyed_df["artist_name"].map(normalize_hebrew_text)
    keyed_df["match_decade"] = keyed_df["decade"].astype(str).str.strip()
    return keyed_df


def make_console_safe(value: object) -> str:
    return str(value).encode("unicode_escape").decode("ascii")


def print_unmatched_rows(label: str, unmatched_df: pd.DataFrame, limit: int = 25) -> None:
    print(f"{label}: {len(unmatched_df)}")
    if unmatched_df.empty:
        return

    preview_columns = [
        column
        for column in ["song_name", "artist_name", "decade", "source_file", "source_line"]
        if column in unmatched_df.columns
    ]
    preview_df = unmatched_df[preview_columns].head(limit).copy()
    for column in preview_df.columns:
        preview_df[column] = preview_df[column].map(make_console_safe)
    print(preview_df.to_string(index=False))
    if len(unmatched_df) > limit:
        print(f"... truncated to first {limit} rows")


def merge_with_feature_table(
    feature_table_path: Path,
    dicta_song_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_df = pd.read_csv(feature_table_path, encoding="utf-8-sig")
    feature_df = build_match_key_dataframe(feature_df)

    dicta_df = pd.DataFrame(dicta_song_rows)
    if dicta_df.empty:
        raise ValueError("No Dicta songs were parsed.")
    dicta_df = build_match_key_dataframe(dicta_df)

    dicta_feature_columns = [
        column
        for column in dicta_df.columns
        if column not in METADATA_COLUMNS + MATCH_KEY_COLUMNS + ["source_file", "source_line"]
    ]

    feature_duplicate_count = int(feature_df.duplicated(MATCH_KEY_COLUMNS).sum())
    dicta_duplicate_count = int(dicta_df.duplicated(MATCH_KEY_COLUMNS).sum())
    if feature_duplicate_count:
        print(f"Warning: duplicate feature-table match keys: {feature_duplicate_count}")
    if dicta_duplicate_count:
        print(f"Warning: duplicate Dicta match keys: {dicta_duplicate_count}")

    if dicta_duplicate_count:
        dicta_df = dicta_df.drop_duplicates(MATCH_KEY_COLUMNS, keep="first").copy()

    merge_columns = MATCH_KEY_COLUMNS + dicta_feature_columns
    merged_df = feature_df.merge(
        dicta_df[merge_columns],
        on=MATCH_KEY_COLUMNS,
        how="left",
        indicator=True,
    )

    matched_song_count = int((merged_df["_merge"] == "both").sum())
    unmatched_feature_table_df = merged_df.loc[merged_df["_merge"] == "left_only", METADATA_COLUMNS + MATCH_KEY_COLUMNS].copy()

    feature_key_df = feature_df[MATCH_KEY_COLUMNS].drop_duplicates().copy()
    unmatched_dicta_df = dicta_df.merge(
        feature_key_df,
        on=MATCH_KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    unmatched_dicta_df = unmatched_dicta_df.loc[
        unmatched_dicta_df["_merge"] == "left_only",
        METADATA_COLUMNS + MATCH_KEY_COLUMNS + ["source_file", "source_line"],
    ].copy()

    print_unmatched_rows("Unmatched feature-table songs", unmatched_feature_table_df)
    print_unmatched_rows("Unmatched Dicta songs", unmatched_dicta_df)

    merged_df[dicta_feature_columns] = merged_df[dicta_feature_columns].fillna(0)
    final_df = merged_df.drop(columns=MATCH_KEY_COLUMNS + ["_merge"])

    summary = {
        "input_feature_table_shape": [int(feature_df.shape[0]), int(feature_df.shape[1] - len(MATCH_KEY_COLUMNS))],
        "final_feature_table_shape": [int(final_df.shape[0]), int(final_df.shape[1])],
        "number_of_dicta_feature_columns": len(dicta_feature_columns),
        "matched_song_count": matched_song_count,
        "unmatched_feature_table_song_count": int(len(unmatched_feature_table_df)),
        "unmatched_dicta_song_count": int(len(unmatched_dicta_df)),
        "dicta_feature_columns": dicta_feature_columns,
        "unmatched_feature_table_rows": unmatched_feature_table_df,
        "unmatched_dicta_rows": unmatched_dicta_df,
    }
    return final_df, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading feature table: {FEATURE_TABLE_PATH}")
    parsed_song_rows: list[dict[str, object]] = []
    songs_parsed_by_decade: dict[str, int] = {}
    file_level_stats: dict[str, dict[str, int]] = {}

    for decade, ud_path in DICTA_FILE_PATHS.items():
        songs, stats = parse_ud_file(ud_path, decade)
        parsed_song_rows.extend(songs)
        songs_parsed_by_decade[decade] = len(songs)
        file_level_stats[decade] = stats
        print(
            f"Parsed {len(songs)} songs from {ud_path} "
            f"(processed_token_rows={stats.get('processed_token_rows', 0)}, "
            f"title_sentence_token_rows_skipped={stats.get('title_sentence_token_rows_skipped', 0)}, "
            f"short_token_rows={stats.get('short_token_rows', 0)})"
        )

    total_parsed_song_count = len(parsed_song_rows)
    print(f"Total Dicta songs parsed: {total_parsed_song_count}")

    final_df, merge_summary = merge_with_feature_table(FEATURE_TABLE_PATH, parsed_song_rows)

    unmatched_feature_table_df = merge_summary.pop("unmatched_feature_table_rows")
    unmatched_dicta_df = merge_summary.pop("unmatched_dicta_rows")

    final_df.to_csv(OUTPUT_FEATURE_TABLE_PATH, index=False, encoding="utf-8-sig")
    unmatched_feature_table_df.to_csv(OUTPUT_UNMATCHED_FEATURE_TABLE_PATH, index=False, encoding="utf-8-sig")
    unmatched_dicta_df.to_csv(OUTPUT_UNMATCHED_DICTA_PATH, index=False, encoding="utf-8-sig")

    summary_payload = {
        **merge_summary,
        "songs_parsed_by_decade": songs_parsed_by_decade,
        "file_level_stats": file_level_stats,
    }
    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(summary_payload, json_file, ensure_ascii=False, indent=2)

    print(f"Matched songs: {merge_summary['matched_song_count']}")
    print(f"Final output shape: {tuple(merge_summary['final_feature_table_shape'])}")
    print(
        "First 10 Dicta feature columns: "
        + ", ".join(merge_summary["dicta_feature_columns"][:10])
    )
    print(f"Saved enriched feature table: {OUTPUT_FEATURE_TABLE_PATH}")
    print(f"Saved Dicta summary JSON: {OUTPUT_SUMMARY_PATH}")
    print(f"Saved unmatched feature-table songs: {OUTPUT_UNMATCHED_FEATURE_TABLE_PATH}")
    print(f"Saved unmatched Dicta songs: {OUTPUT_UNMATCHED_DICTA_PATH}")


if __name__ == "__main__":
    main()
