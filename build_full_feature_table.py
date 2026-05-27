from __future__ import annotations

import csv
import json
import re
import statistics
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


INPUT_CSV_PATH = Path("israeli_songs_corpus.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV_PATH = OUTPUT_DIR / "song_feature_table.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "feature_table_summary.json"

METADATA_COLUMNS = ["song_name", "artist_name", "decade"]

TOKEN_RE = re.compile(r"[A-Za-z\u05D0-\u05EA]+(?:[\"'׳״][A-Za-z\u05D0-\u05EA]+)*")
ENGLISH_CHAR_RE = re.compile(r"[A-Za-z]")

PUNCTUATION_CHARS = set(string.punctuation) | set("־–—…״׳“”‘’")
QUOTE_CHARS = set("\"'״׳“”‘’")
PARENTHESIS_CHARS = set("()[]{}")

FIRST_PERSON_SINGULAR = {
    "אני",
    "אותי",
    "לי",
    "שלי",
    "בי",
    "ממני",
    "אלי",
    "עלי",
    "אצלי",
    "בשבילי",
}

FIRST_PERSON_PLURAL = {
    "אנחנו",
    "אותנו",
    "לנו",
    "שלנו",
    "בנו",
    "מאיתנו",
    "עלינו",
    "אלינו",
    "אצלנו",
}

SECOND_PERSON = {
    "אתה",
    "את",
    "אתם",
    "אתן",
    "אותך",
    "לך",
    "לכם",
    "לכן",
    "שלך",
    "שלכם",
    "שלכן",
    "בך",
    "בכם",
    "בכן",
    "אליך",
    "אלייך",
    "עליך",
    "עלייך",
}

THIRD_PERSON = {
    "הוא",
    "היא",
    "הם",
    "הן",
    "אותו",
    "אותה",
    "אותם",
    "אותן",
    "שלו",
    "שלה",
    "שלהם",
    "שלהן",
    "לו",
    "לה",
    "להם",
    "להן",
    "אליו",
    "אליה",
    "עליו",
    "עליה",
}

SEMANTIC_LEXICONS = {
    "love_words_ratio": {
        "אהבה",
        "אהבתי",
        "אוהב",
        "אוהבת",
        "אוהבים",
        "אוהבות",
        "אהוב",
        "אהובה",
        "לב",
        "לבי",
        "נשיקה",
        "נשיקות",
        "חיבוק",
        "מחבק",
        "מחבקת",
        "תשוקה",
        "מותק",
        "יקירי",
        "יקירה",
    },
    "army_words_ratio": {
        "חייל",
        "חיילים",
        "צבא",
        "מלחמה",
        "נשק",
        "מדים",
        "מפקד",
        "בסיס",
        "קרב",
        "חזית",
        "גיוס",
        "מילואים",
        "חטיבה",
        "גדוד",
        "רובה",
        "קסדה",
        "טנק",
        "פקודה",
        "מארב",
    },
    "nature_words_ratio": {
        "ים",
        "שמש",
        "ירח",
        "כוכב",
        "כוכבים",
        "רוח",
        "אדמה",
        "שדה",
        "שדות",
        "פרח",
        "פרחים",
        "גשם",
        "עץ",
        "עצים",
        "הר",
        "נהר",
        "מדבר",
        "שמים",
        "ציפור",
        "ציפורים",
        "עלה",
        "עלים",
        "יער",
    },
    "religion_words_ratio": {
        "אלוהים",
        "אדוני",
        "תפילה",
        "תפילין",
        "שבת",
        "קודש",
        "קדוש",
        "מצווה",
        "מצוה",
        "רב",
        "ברכה",
        "אמונה",
        "נביא",
        "מלאך",
        "כיפה",
        "תורה",
        "מקדש",
    },
    "party_words_ratio": {
        "מסיבה",
        "מסיבות",
        "רוקד",
        "רוקדת",
        "לרקוד",
        "ריקוד",
        "מועדון",
        "בירה",
        "יין",
        "וודקה",
        "שיכור",
        "שיכורה",
        "דרינק",
        "קצב",
        "דיגיי",
        "לילה",
        "שמחה",
    },
    "sadness_words_ratio": {
        "עצוב",
        "עצובה",
        "עצב",
        "דמעות",
        "דמעה",
        "בכי",
        "בוכה",
        "לבכות",
        "כאב",
        "כאוב",
        "שבור",
        "שבורה",
        "בדידות",
        "לבד",
        "צער",
        "אובדן",
        "חסר",
    },
    "nostalgia_words_ratio": {
        "זיכרון",
        "זכרון",
        "זיכרונות",
        "זכרונות",
        "געגוע",
        "געגועים",
        "פעם",
        "עבר",
        "ילדות",
        "ישן",
        "ישנה",
        "אתמול",
        "תמול",
        "מזמן",
        "נוסטלגיה",
    },
    "family_words_ratio": {
        "אמא",
        "אבא",
        "אמי",
        "אבי",
        "אם",
        "אב",
        "אח",
        "אחות",
        "אחים",
        "אחיות",
        "ילד",
        "ילדה",
        "ילדים",
        "בן",
        "בת",
        "משפחה",
        "הורים",
        "סבא",
        "סבתא",
        "אשתי",
        "בעלי",
    },
    "place_words_ratio": {
        "עיר",
        "רחוב",
        "כפר",
        "שכונה",
        "בית",
        "ארץ",
        "מדינה",
        "תל",
        "אביב",
        "ירושלים",
        "כביש",
        "כיכר",
        "תחנה",
        "נמל",
        "מדבר",
        "גליל",
        "נגב",
    },
    "time_words_ratio": {
        "יום",
        "לילה",
        "בוקר",
        "ערב",
        "שעה",
        "שעות",
        "דקה",
        "דקות",
        "רגע",
        "זמן",
        "עכשיו",
        "מחר",
        "אתמול",
        "תמיד",
        "כבר",
        "עוד",
        "שנה",
        "שנים",
        "חודש",
        "קיץ",
        "חורף",
    },
    "modern_slang_words_ratio": {
        "סבבה",
        "מגניב",
        "אחלה",
        "וואי",
        "יאללה",
        "בקטנה",
        "בכיף",
        "וייב",
        "סטורי",
        "פאק",
        "קול",
        "דאחקה",
        "כפרה",
        "אחי",
        "סתלבט",
        "לייק",
        "פולואו",
        "טיקטוק",
    },
    "old_israeli_words_ratio": {
        "פלמח",
        "קומזיץ",
        "צבר",
        "חלוץ",
        "חלוצים",
        "קיבוץ",
        "קיבוצניק",
        "מולדת",
        "רעות",
        "הגשמה",
        "פרדס",
        "מעברה",
        "טרקטור",
        "התיישבות",
        "שיבולים",
    },
}

FEATURE_COLUMNS = [
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
    "repetition_ratio",
    "punctuation_count",
    "punctuation_ratio",
    "question_mark_count",
    "exclamation_mark_count",
    "comma_count",
    "quote_count",
    "parenthesis_count",
    "digit_count",
    "english_char_count",
    "english_char_ratio",
    "repeated_lines_count",
    "repeated_lines_ratio",
    "unique_lines_count",
    "most_repeated_line_count",
    "chorus_repetition_score",
    "repeated_words_count",
    "repeated_words_ratio",
    "most_common_word_count",
    "most_common_word_ratio",
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
    "last_word_unique_count",
    "last_word_repetition_ratio",
    "avg_last_word_length",
]


def strip_niqqud(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return strip_niqqud(normalized)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(label: object) -> str:
    cleaned = normalize_spaces(normalize_text(label)).lower()
    if not cleaned:
        return ""
    if "1980" in cleaned or re.search(r"\b80s\b|\b80's\b|\b80\b", cleaned):
        return "1980s"
    if "2000" in cleaned or re.search(r"\b00s\b|\b00's\b", cleaned):
        return "2000s"
    if "2020" in cleaned or re.search(r"\b20s\b|\b20's\b", cleaned):
        return "2020s"
    raise ValueError(f"Unsupported decade label: {label!r}")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    return [token.lower() for token in tokens]


def get_non_empty_lines(text: str) -> list[str]:
    return [normalize_spaces(line) for line in text.split("\n") if normalize_spaces(line)]


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def normalize_token_set(words: Iterable[str]) -> set[str]:
    normalized_words: set[str] = set()
    for word in words:
        token_text = normalize_text(word)
        for token in tokenize(token_text):
            normalized_words.add(token)
    return normalized_words


FIRST_PERSON_SINGULAR = normalize_token_set(FIRST_PERSON_SINGULAR)
FIRST_PERSON_PLURAL = normalize_token_set(FIRST_PERSON_PLURAL)
SECOND_PERSON = normalize_token_set(SECOND_PERSON)
THIRD_PERSON = normalize_token_set(THIRD_PERSON)
SEMANTIC_LEXICONS = {
    feature_name: normalize_token_set(words)
    for feature_name, words in SEMANTIC_LEXICONS.items()
}


def count_tokens_in_lexicon(tokens: list[str], lexicon: set[str]) -> int:
    return sum(1 for token in tokens if token in lexicon)


def extract_basic_features(text: str, tokens: list[str], lines: list[str]) -> dict[str, float | int]:
    word_count = len(tokens)
    unique_word_count = len(set(tokens))
    char_count = len(text)
    line_count = len(lines)
    words_per_line = [len(tokenize(line)) for line in lines]
    avg_word_length = safe_divide(sum(len(token) for token in tokens), word_count)

    return {
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "char_count": char_count,
        "line_count": line_count,
        "avg_words_per_line": safe_divide(sum(words_per_line), line_count),
        "median_words_per_line": statistics.median(words_per_line) if words_per_line else 0,
        "max_words_per_line": max(words_per_line, default=0),
        "min_words_per_line": min(words_per_line, default=0),
        "avg_word_length": avg_word_length,
        "lexical_diversity": safe_divide(unique_word_count, word_count),
        "repetition_ratio": safe_divide(word_count - unique_word_count, word_count),
    }


def extract_punctuation_features(text: str) -> dict[str, float | int]:
    punctuation_count = sum(1 for char in text if char in PUNCTUATION_CHARS)
    question_mark_count = text.count("?")
    exclamation_mark_count = text.count("!")
    comma_count = text.count(",")
    quote_count = sum(1 for char in text if char in QUOTE_CHARS)
    parenthesis_count = sum(1 for char in text if char in PARENTHESIS_CHARS)
    digit_count = sum(1 for char in text if char.isdigit())
    english_char_count = len(ENGLISH_CHAR_RE.findall(text))
    char_count = len(text)

    return {
        "punctuation_count": punctuation_count,
        "punctuation_ratio": safe_divide(punctuation_count, char_count),
        "question_mark_count": question_mark_count,
        "exclamation_mark_count": exclamation_mark_count,
        "comma_count": comma_count,
        "quote_count": quote_count,
        "parenthesis_count": parenthesis_count,
        "digit_count": digit_count,
        "english_char_count": english_char_count,
        "english_char_ratio": safe_divide(english_char_count, char_count),
    }


def extract_line_repetition_features(lines: list[str]) -> dict[str, float | int]:
    line_count = len(lines)
    line_counter = Counter(lines)
    unique_lines_count = len(line_counter)
    repeated_lines_count = sum(count - 1 for count in line_counter.values() if count > 1)
    repeated_line_coverage = sum(count for count in line_counter.values() if count > 1)

    return {
        "repeated_lines_count": repeated_lines_count,
        "repeated_lines_ratio": safe_divide(repeated_lines_count, line_count),
        "unique_lines_count": unique_lines_count,
        "most_repeated_line_count": max(line_counter.values(), default=0),
        "chorus_repetition_score": safe_divide(repeated_line_coverage, line_count),
    }


def extract_word_repetition_features(tokens: list[str]) -> dict[str, float | int]:
    word_count = len(tokens)
    word_counter = Counter(tokens)
    unique_word_count = len(word_counter)
    repeated_words_count = sum(1 for count in word_counter.values() if count > 1)
    most_common_word_count = max(word_counter.values(), default=0)

    return {
        "repeated_words_count": repeated_words_count,
        "repeated_words_ratio": safe_divide(repeated_words_count, unique_word_count),
        "most_common_word_count": most_common_word_count,
        "most_common_word_ratio": safe_divide(most_common_word_count, word_count),
    }


def extract_pronoun_features(tokens: list[str]) -> dict[str, float | int]:
    word_count = len(tokens)
    first_person_singular_count = count_tokens_in_lexicon(tokens, FIRST_PERSON_SINGULAR)
    first_person_plural_count = count_tokens_in_lexicon(tokens, FIRST_PERSON_PLURAL)
    second_person_count = count_tokens_in_lexicon(tokens, SECOND_PERSON)
    third_person_count = count_tokens_in_lexicon(tokens, THIRD_PERSON)
    total_person_reference_count = (
        first_person_singular_count
        + first_person_plural_count
        + second_person_count
        + third_person_count
    )

    return {
        "first_person_singular_count": first_person_singular_count,
        "first_person_singular_ratio": safe_divide(first_person_singular_count, word_count),
        "first_person_plural_count": first_person_plural_count,
        "first_person_plural_ratio": safe_divide(first_person_plural_count, word_count),
        "second_person_count": second_person_count,
        "second_person_ratio": safe_divide(second_person_count, word_count),
        "third_person_count": third_person_count,
        "third_person_ratio": safe_divide(third_person_count, word_count),
        "we_vs_i_ratio": safe_divide(first_person_plural_count, max(first_person_singular_count, 1)),
        "direct_address_score": safe_divide(second_person_count, total_person_reference_count),
    }


def extract_semantic_features(tokens: list[str]) -> dict[str, float]:
    word_count = len(tokens)
    feature_values: dict[str, float] = {}
    for feature_name, lexicon in SEMANTIC_LEXICONS.items():
        match_count = count_tokens_in_lexicon(tokens, lexicon)
        feature_values[feature_name] = safe_divide(match_count, word_count)
    return feature_values


def extract_last_word_features(lines: list[str]) -> dict[str, float | int]:
    last_words: list[str] = []
    for line in lines:
        line_tokens = tokenize(line)
        if line_tokens:
            last_words.append(line_tokens[-1])

    last_word_count = len(last_words)
    unique_last_words = len(set(last_words))

    return {
        "last_word_unique_count": unique_last_words,
        "last_word_repetition_ratio": safe_divide(last_word_count - unique_last_words, last_word_count),
        "avg_last_word_length": safe_divide(sum(len(word) for word in last_words), last_word_count),
    }


def build_feature_row(record: dict[str, object]) -> dict[str, object]:
    lyrics = normalize_text(record.get("lyrics"))
    song_name = normalize_spaces(normalize_text(record.get("song_name")))
    artist_name = normalize_spaces(normalize_text(record.get("artist_name")))
    decade = normalize_label(record.get("decade"))

    tokens = tokenize(lyrics)
    lines = get_non_empty_lines(lyrics)

    row: dict[str, object] = {
        "song_name": song_name,
        "artist_name": artist_name,
        "decade": decade,
    }
    row.update(extract_basic_features(lyrics, tokens, lines))
    row.update(extract_punctuation_features(lyrics))
    row.update(extract_line_repetition_features(lines))
    row.update(extract_word_repetition_features(tokens))
    row.update(extract_pronoun_features(tokens))
    row.update(extract_semantic_features(tokens))
    row.update(extract_last_word_features(lines))
    return row


def read_corpus(input_path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**7)
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = {"song_name", "artist_name", "decade", "lyrics"}
        if not reader.fieldnames or set(reader.fieldnames) != expected_columns:
            raise ValueError(
                f"Unexpected corpus columns: {reader.fieldnames}. "
                f"Expected exactly {sorted(expected_columns)}."
            )
        return list(reader)


def write_feature_table(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = METADATA_COLUMNS + FEATURE_COLUMNS
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    all_columns = METADATA_COLUMNS + FEATURE_COLUMNS
    missing_values_by_column = {
        column: sum(1 for row in rows if is_missing(row.get(column)))
        for column in all_columns
    }
    songs_per_decade = dict(sorted(Counter(row["decade"] for row in rows).items()))

    return {
        "number_of_songs": len(rows),
        "number_of_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "missing_values_by_column": missing_values_by_column,
        "songs_per_decade": songs_per_decade,
    }


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_rows = read_corpus(INPUT_CSV_PATH)
    feature_rows = [build_feature_row(row) for row in corpus_rows]

    write_feature_table(feature_rows, OUTPUT_CSV_PATH)
    summary = build_summary(feature_rows)
    write_summary(summary, OUTPUT_SUMMARY_PATH)

    print("Feature table build complete.")
    print(f"Songs processed: {summary['number_of_songs']}")
    print(f"Feature count: {summary['number_of_features']}")
    print(f"Output table: {OUTPUT_CSV_PATH}")
    print(f"Summary JSON: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
