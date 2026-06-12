"""Hebrew Song Decade Classification - Full Experiment Pipeline.

Runs 80 classification experiments: 12 feature groups x 7 models.
Outputs results CSV and JSON summary.

Usage: python run_classification_experiments.py
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
RANDOM_STATE = 42

FEATURE_GROUPS = {
    "basic_surface": ["word_count","unique_word_count","char_count","line_count","avg_words_per_line","median_words_per_line","max_words_per_line","min_words_per_line","avg_word_length","lexical_diversity"],
    "repetition": ["repetition_ratio","repeated_lines_count","repeated_lines_ratio","unique_lines_count","most_repeated_line_count","chorus_repetition_score","repeated_words_count","repeated_words_ratio","most_common_word_count","most_common_word_ratio"],
    "pronoun_person": ["first_person_singular_count","first_person_singular_ratio","first_person_plural_count","first_person_plural_ratio","second_person_count","second_person_ratio","third_person_count","third_person_ratio","we_vs_i_ratio","direct_address_score"],
    "semantic_lexicon": ["love_words_ratio","army_words_ratio","nature_words_ratio","religion_words_ratio","party_words_ratio","sadness_words_ratio","nostalgia_words_ratio","family_words_ratio","place_words_ratio","time_words_ratio","modern_slang_words_ratio","old_israeli_words_ratio"],
    "dicta_pos": ["pos_noun_ratio","pos_verb_ratio","pos_adj_ratio","pos_adv_ratio","pos_pron_ratio","pos_propn_ratio","pos_adp_ratio","pos_det_ratio","pos_cconj_ratio","pos_sconj_ratio","pos_aux_ratio","pos_num_ratio","pos_intj_ratio"],
    "dicta_morphology": ["tense_past_ratio","tense_future_ratio","verb_inf_ratio","verb_participle_ratio","imperative_ratio","masculine_ratio","feminine_ratio","singular_ratio","plural_ratio","first_person_morph_ratio","second_person_morph_ratio","third_person_morph_ratio"],
    "dicta_binyan": ["binyan_paal_ratio","binyan_piel_ratio","binyan_hifil_ratio","binyan_hitpael_ratio","binyan_nifal_ratio","binyan_pual_ratio","binyan_hufal_ratio"],
    "dicta_lemma_root": ["lemma_diversity","root_diversity","dicta_unknown_token_ratio"],
    "punctuation_formatting": ["punctuation_ratio","question_mark_count","exclamation_mark_count","comma_count","quote_count","parenthesis_count","english_char_ratio","digit_count"],
    "last_word": ["last_word_unique_count","last_word_repetition_ratio","avg_last_word_length"],
}

def main():
    ft = pd.read_csv(BASE / "outputs/song_feature_table_with_dicta.csv")
    corpus = pd.read_csv(BASE / "israeli_songs_corpus.csv")
    df = ft.merge(corpus[["song_name","artist_name","decade","lyrics"]], on=["song_name","artist_name","decade"], how="left")
    df = df.drop(columns=["tense_present_count","tense_present_ratio"], errors="ignore")
    y = df["decade"]
    all_num = [c for c in df.columns if c not in ["song_name","artist_name","decade","lyrics"] and df[c].dtype in ["int64","float64"]]
    FEATURE_GROUPS["all_numeric"] = all_num
    redundant = ["repetition_ratio","dicta_x_pos_count","dicta_x_pos_ratio","dicta_token_count","dicta_unknown_token_count","char_count","chorus_repetition_score","unique_lemma_count"]
    FEATURE_GROUPS["reduced_non_redundant"] = [c for c in all_num if c not in redundant]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "macro_f1": make_scorer(f1_score, average="macro")}
    models = {
        "Majority Class": DummyClassifier(strategy="most_frequent"),
        "Stratified Random": DummyClassifier(strategy="stratified"),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]),
        "Linear SVM": Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(max_iter=5000, random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    results = []
    for gname, fcols in FEATURE_GROUPS.items():
        valid = [c for c in fcols if c in df.columns]
        if not valid: continue
        X = df[valid].fillna(0).values
        for mname, model in models.items():
            s = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            results.append({"Feature Group": gname, "Model": mname, "Accuracy": s["test_accuracy"].mean(), "Acc Std": s["test_accuracy"].std(), "Macro F1": s["test_macro_f1"].mean(), "F1 Std": s["test_macro_f1"].std(), "N Features": len(valid)})
        print(f"  done: {gname}")

    lyrics = df["lyrics"].fillna("")
    for tname, ngram in [("tfidf_unigrams",(1,1)),("tfidf_bigrams",(1,2))]:
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=ngram, sublinear_tf=True)
        Xt = tfidf.fit_transform(lyrics)
        for mname, model in [("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)), ("Linear SVM", LinearSVC(max_iter=5000, random_state=42)), ("Multinomial NB", MultinomialNB())]:
            s = cross_validate(model, Xt, y, cv=cv, scoring=scoring, n_jobs=-1)
            results.append({"Feature Group": tname, "Model": mname, "Accuracy": s["test_accuracy"].mean(), "Acc Std": s["test_accuracy"].std(), "Macro F1": s["test_macro_f1"].mean(), "F1 Std": s["test_macro_f1"].std(), "N Features": Xt.shape[1]})
        print(f"  done: {tname}")

    tfidf_bi = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    Xt = tfidf_bi.fit_transform(lyrics)
    Xn = StandardScaler().fit_transform(df[all_num].fillna(0).values)
    Xc = hstack([csr_matrix(Xn), Xt])
    for mname, model in [("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)), ("Linear SVM", LinearSVC(max_iter=5000, random_state=42))]:
        s = cross_validate(model, Xc, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({"Feature Group": "numeric+tfidf_bigrams", "Model": mname, "Accuracy": s["test_accuracy"].mean(), "Acc Std": s["test_accuracy"].std(), "Macro F1": s["test_macro_f1"].mean(), "F1 Std": s["test_macro_f1"].std(), "N Features": Xc.shape[1]})

    rdf = pd.DataFrame(results)
    rdf.to_csv(BASE / "outputs/classification_results_all.csv", index=False)
    print(f"\nTotal experiments: {len(rdf)}")
    print(rdf.sort_values("Accuracy", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
