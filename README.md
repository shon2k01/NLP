# Hebrew Israeli Songs Decade Classification - NLP Project

## 1. Project Overview

This project analyzes Hebrew Israeli song lyrics from three decades:

- 1980s
- 2000s
- 2020s

The goal is to study stylistic, lexical, morphological, and semantic differences between decades, and eventually train models that classify a new song into its likely decade based only on its lyrics and extracted linguistic features.

Main research question:

> Can Hebrew Israeli songs be classified by decade using linguistic, stylistic, morphological, semantic, and machine-learning features?

The project currently contains a full feature-engineering and exploratory-analysis pipeline. Classification experiments are planned as the next major step.

## 2. Dataset

Main corpus file:

- `israeli_songs_corpus.csv`

Corpus size:

- 597 songs total
- 1980s: 200 songs
- 2000s: 200 songs
- 2020s: 197 songs

Columns:

- `song_name`
- `artist_name`
- `decade`
- `lyrics`

The lyrics are in Hebrew. The metadata columns `song_name` and `artist_name` are preserved for traceability and matching, but they must not be used as predictive features. The `decade` column is the target label.

## 3. Project Pipeline

The project pipeline is:

1. Collect Hebrew song lyrics by decade.
2. Build a per-song feature table.
3. Run sanity checks and exploratory feature analysis.
4. Enrich each song with Dicta morphological analysis.
5. Analyze the enriched feature table.
6. Train classification models using different feature groups.
7. Compare classical machine-learning models to AI/LLM-based approaches.
8. Write a final research report.

## 4. Step 1 - Initial Per-Song Feature Table

Script:

- `build_full_feature_table.py`

Inputs:

- `israeli_songs_corpus.csv`

Outputs:

- `outputs/song_feature_table.csv`
- `outputs/feature_table_summary.json`

Result:

- 597 rows
- 58 columns total
- 3 metadata columns: `song_name`, `artist_name`, `decade`
- 55 feature columns
- 0 missing values
- Decade counts: `1980s=200`, `2000s=200`, `2020s=197`

Purpose:

The goal of this step was to convert raw Hebrew lyrics into a structured tabular representation where each song is represented by measurable features that can later be used for analysis and classification.

Feature groups:

- Basic stylometric features: `word_count`, `unique_word_count`, `char_count`, `line_count`, `avg_words_per_line`, `median_words_per_line`, `max_words_per_line`, `min_words_per_line`, `avg_word_length`, `lexical_diversity`, `repetition_ratio`
- Punctuation and formatting: `punctuation_count`, `punctuation_ratio`, `question_mark_count`, `exclamation_mark_count`, `comma_count`, `quote_count`, `parenthesis_count`, `digit_count`, `english_char_count`, `english_char_ratio`
- Line and word repetition: `repeated_lines_count`, `repeated_lines_ratio`, `unique_lines_count`, `most_repeated_line_count`, `chorus_repetition_score`, `repeated_words_count`, `repeated_words_ratio`, `most_common_word_count`, `most_common_word_ratio`
- Pronoun/person features: `first_person_singular_count`, `first_person_singular_ratio`, `first_person_plural_count`, `first_person_plural_ratio`, `second_person_count`, `second_person_ratio`, `third_person_count`, `third_person_ratio`, `we_vs_i_ratio`, `direct_address_score`
- Manual semantic lexicons: `love_words_ratio`, `army_words_ratio`, `nature_words_ratio`, `religion_words_ratio`, `party_words_ratio`, `sadness_words_ratio`, `nostalgia_words_ratio`, `family_words_ratio`, `place_words_ratio`, `time_words_ratio`, `modern_slang_words_ratio`, `old_israeli_words_ratio`
- Rhyme-like/simple ending features: `last_word_unique_count`, `last_word_repetition_ratio`, `avg_last_word_length`

Implementation notes:

- Feature names are English for code compatibility.
- Hebrew niqqud/diacritics are stripped during tokenization.
- Missing or empty lyrics are handled safely.
- `song_name` and `artist_name` are kept only as metadata.

## 5. Step 2 - Initial Feature Table Quality Analysis

Script:

- `analyze_feature_table.py`

Inputs:

- `outputs/song_feature_table.csv`

Outputs:

- `outputs/feature_means_by_decade.csv`
- `outputs/top_decade_separating_features.csv`
- `outputs/feature_correlation_matrix.csv`
- `outputs/figures/`

Checks performed:

- Table shape
- Number of metadata and numeric columns
- Songs per decade
- Missing values
- Infinite values
- Constant columns
- Near-constant columns
- Numeric columns with suspiciously high maximum values
- Feature means by decade
- Top decade-separating features
- Feature correlation matrix
- Exploratory plots

Findings:

- 0 missing values
- 0 infinite values
- No constant features
- One near-constant feature: `old_israeli_words_ratio`

Top raw decade-separating features included:

- `char_count`
- `word_count`
- `unique_word_count`
- `english_char_count`
- `repeated_words_count`
- `line_count`
- `repeated_lines_count`
- `second_person_count`
- `first_person_singular_count`
- `unique_lines_count`
- `most_common_word_count`
- `last_word_unique_count`
- `max_words_per_line`
- `parenthesis_count`

Interpretation:

- 2020s songs in the corpus tend to be longer.
- 2020s songs tend to contain more repeated words and repeated lines.
- 2020s songs tend to contain more first-person and second-person language.
- 1980s and 2000s are closer to each other than either is to 2020s.
- Some technical features such as `english_char_count`, `parenthesis_count`, and `digit_count` may reflect real stylistic changes, but may also reflect source formatting artifacts. These should be tested carefully later.

Why this step matters:

Before training models, feature quality must be validated to avoid training on broken, constant, missing, or misleading features.

## 6. Step 3 - Dicta Morphological Feature Enrichment

Script:

- `add_dicta_features.py`

Inputs:

- `outputs/song_feature_table.csv`
- `DictaAnalysis/dicta-80s/dicta_all_1980s.ud.txt`
- `DictaAnalysis/dicta-00s/dicta_all_2000s.ud.txt`
- `DictaAnalysis/dicta-20s/dicta_all_2020s.ud.txt`

Outputs:

- `outputs/song_feature_table_with_dicta.csv`
- `outputs/dicta_feature_summary.json`
- `outputs/dicta_unmatched_feature_table_songs.csv`
- `outputs/dicta_unmatched_dicta_songs.csv`

Results:

- Dicta songs parsed: 597
- 1980s: 200
- 2000s: 200
- 2020s: 197
- Songs matched to feature table: 597
- Unmatched feature-table songs: 0
- Unmatched Dicta songs: 0
- Final enriched table shape: 597 rows x 137 columns

Parser details:

- Dicta files are in UD-style tabular format.
- Songs are detected using title/comment lines such as `# text = ### song name / artist name`.
- The parser skips comments, empty lines, range token IDs, decimal token IDs, malformed rows, and punctuation where appropriate.
- Matching is done using normalized `song_name`, normalized `artist_name`, and `decade`.
- One edge case was handled where a 2020s Dicta title split the song and artist across two metadata sentences.
- Duplicate normalized match keys were reported in the 2020s data, but they still matched cleanly and did not create unmatched rows.

Why this step matters:

The initial features mostly capture surface-level structure. Dicta adds deeper Hebrew linguistic information such as POS tags, morphology, tense, person, gender, number, binyan, lemmas, roots, and unknown-token behavior.

Dicta feature groups:

- Token features: `dicta_token_count`, `dicta_non_punct_token_count`, `dicta_content_token_count`
- POS counts and ratios: `pos_noun_count`, `pos_noun_ratio`, `pos_verb_count`, `pos_verb_ratio`, `pos_adj_count`, `pos_adj_ratio`, `pos_adv_count`, `pos_adv_ratio`, `pos_pron_count`, `pos_pron_ratio`, `pos_propn_count`, `pos_propn_ratio`, `pos_adp_count`, `pos_adp_ratio`, `pos_det_count`, `pos_det_ratio`, `pos_cconj_count`, `pos_cconj_ratio`, `pos_sconj_count`, `pos_sconj_ratio`, `pos_aux_count`, `pos_aux_ratio`, `pos_num_count`, `pos_num_ratio`, `pos_intj_count`, `pos_intj_ratio`
- Tense: `tense_past_count`, `tense_past_ratio`, `tense_present_count`, `tense_present_ratio`, `tense_future_count`, `tense_future_ratio`
- Verb forms: `verb_inf_count`, `verb_inf_ratio`, `verb_participle_count`, `verb_participle_ratio`, `imperative_count`, `imperative_ratio`
- Gender and number: `masculine_count`, `masculine_ratio`, `feminine_count`, `feminine_ratio`, `singular_count`, `singular_ratio`, `plural_count`, `plural_ratio`
- Person morphology: `first_person_morph_count`, `first_person_morph_ratio`, `second_person_morph_count`, `second_person_morph_ratio`, `third_person_morph_count`, `third_person_morph_ratio`
- Binyan: `binyan_paal_count`, `binyan_paal_ratio`, `binyan_piel_count`, `binyan_piel_ratio`, `binyan_hifil_count`, `binyan_hifil_ratio`, `binyan_hitpael_count`, `binyan_hitpael_ratio`, `binyan_nifal_count`, `binyan_nifal_ratio`, `binyan_pual_count`, `binyan_pual_ratio`, `binyan_hufal_count`, `binyan_hufal_ratio`
- Lemma/root diversity: `lemma_count`, `unique_lemma_count`, `lemma_diversity`, `root_count`, `unique_root_count`, `root_diversity`
- Unknown/X-token features: `dicta_unknown_token_count`, `dicta_unknown_token_ratio`, `dicta_x_pos_count`, `dicta_x_pos_ratio`

## 7. Step 4 - Dicta-Enriched Feature Table Analysis

Script:

- `analyze_dicta_enriched_table.py`

Inputs:

- `outputs/song_feature_table_with_dicta.csv`

Outputs:

- `outputs/dicta_enriched_feature_means_by_decade.csv`
- `outputs/dicta_enriched_top_decade_separating_features.csv`
- `outputs/dicta_enriched_feature_eta_squared.csv`
- `outputs/dicta_enriched_feature_correlation_matrix.csv`
- `outputs/dicta_enriched_analysis_summary.json`
- `outputs/figures_dicta/`

Quality results:

- Table shape: 597 x 137
- Missing values: 0
- Infinite values: 0
- Constant/all-zero columns: `tense_present_count`, `tense_present_ratio`
- Near-constant columns: `tense_present_count`, `tense_present_ratio`, `old_israeli_words_ratio`

Present-tense issue:

The present-tense Dicta features are all zero, likely because Hebrew present tense is often encoded by Dicta as `VerbForm=Part` rather than `Tense=Present`. Therefore these columns should be removed or ignored during model training.

Top eta-squared features:

- `repeated_words_count`: 0.244335
- `repeated_lines_count`: 0.174687
- `second_person_morph_count`: 0.166422
- `word_count`: 0.164920
- `second_person_count`: 0.158091
- `pos_adv_count`: 0.153480
- `lemma_diversity`: 0.150684
- `char_count`: 0.148174
- `pos_pron_count`: 0.148108
- `dicta_non_punct_token_count`: 0.142184
- `pos_verb_count`: 0.141302
- `pos_sconj_count`: 0.137647
- `line_count`: 0.134612
- `lemma_count`: 0.133287
- `first_person_morph_count`: 0.128029
- `repeated_lines_ratio`: 0.125817
- `root_count`: 0.124385
- `chorus_repetition_score`: 0.123351
- `lexical_diversity`: 0.122662
- `repetition_ratio`: 0.122662

Interpretation:

- Repetition remains the strongest signal.
- Length remains important.
- Second-person language is strongly associated with decade differences, both through manual counts and through Dicta morphology.
- Dicta adds meaningful linguistic signal through POS distributions, lemma diversity, root counts, and person morphology.
- The most useful Dicta features appear to complement the surface-level features.

Highly correlated feature pairs:

- `lexical_diversity` <-> `repetition_ratio`: -1.000
- `dicta_unknown_token_ratio` <-> `dicta_x_pos_ratio`: about 0.994
- `dicta_token_count` <-> `dicta_non_punct_token_count`: about 0.993
- `dicta_unknown_token_count` <-> `dicta_x_pos_count`: about 0.993
- `word_count` <-> `char_count`: about 0.990
- `repeated_lines_ratio` <-> `chorus_repetition_score`: about 0.979
- `unique_word_count` <-> `unique_lemma_count`: about 0.971

Modeling implication:

These correlations are expected and not necessarily errors. However, interpretable models should consider reducing redundant features. For regularized models, scaling and regularization can handle much of this redundancy.

## 8. Current Interpretation So Far

Current research interpretation:

- Songs from the 2020s are structurally different from the earlier decades in this corpus.
- The strongest differences are length, repetition, and personal/direct address.
- 2020s songs are generally longer and more repetitive.
- 2020s songs appear to use more first-person and second-person language.
- Dicta confirms and strengthens some of these findings, especially around person morphology and pronoun usage.
- The 1980s and 2000s are more similar to each other in many features.
- Some features may reflect data-source artifacts, so model experiments should include versions with and without suspicious technical features.
- Manual semantic lexicons are useful but currently sparse; some lexicons such as `old_israeli_words_ratio` should likely be expanded or treated as experimental.

## 9. Classification Experiments - Results

Script: `run_classification_experiments.py`

### Setup

- Evaluation: 5-fold stratified cross-validation
- Total experiments: 80 (12 feature groups x 6-7 models)
- Random state: 42

### Top 10 Results

| Rank | Accuracy | Macro F1 | Model | Feature Group |
|------|----------|----------|-------|---------------|
| 1 | **65.5%** | 65.1% | Linear SVM | TF-IDF bigrams |
| 2 | 64.5% | 64.3% | Logistic Regression | TF-IDF bigrams |
| 3 | 64.3% | 63.8% | Logistic Regression | TF-IDF unigrams |
| 4 | 63.7% | 63.2% | Linear SVM | TF-IDF unigrams |
| 5 | 63.2% | 63.3% | Multinomial NB | TF-IDF bigrams |
| 6 | 61.1% | 60.0% | Multinomial NB | TF-IDF unigrams |
| 7 | 60.5% | 60.2% | Gradient Boosting | Reduced numeric (124) |
| 8 | 60.0% | 59.8% | Logistic Regression | Numeric + TF-IDF |
| 9 | 59.8% | 59.7% | Logistic Regression | All numeric (132) |
| 10 | 59.6% | 59.5% | Logistic Regression | Reduced numeric (124) |

Baseline: Majority class = 33.5%, Random = 33.3%

### Per-Decade Difficulty

| Decade | Recall | Interpretation |
|--------|--------|----------------|
| 2020s | 75-82% | Easiest - distinct modern style |
| 1980s | 60-80% | Medium - recognizable |
| 2000s | 33-48% | Hardest - transitional decade |

### Top Features (Random Forest Importance)

1. `punctuation_ratio` (0.022)
2. `repeated_words_count` (0.021)
3. `pos_sconj_ratio` (0.017)
4. `char_count` / `word_count` (0.015-0.016)
5. `second_person_count` (0.015)
6. `chorus_repetition_score` (0.015)

### Key Findings

1. **TF-IDF bigrams dominate** - actual words classify better than numeric features
2. **Bigrams > unigrams** (+2% avg) - word pairs carry decade-specific patterns
3. **2020s most distinct**, 2000s hardest (transitional decade)
4. **Best numeric-only**: Gradient Boosting + reduced features (60.5%)
5. **Most valuable Dicta features**: morphology (+4.9%) and POS (+3.7%)
6. **Combined numeric + TF-IDF hurts** (60%) vs pure TF-IDF (65.5%)


## 9a. Corrected Classification Experiment

Script: `run_classification_experiments_corrected.py`

Outputs:

- `outputs/classification_results_corrected.csv`
- `outputs/classification_summary_corrected.json`

This experiment is a methodological correction and validation step for the first classification experiment. The original experiment remains documented above because it is useful as the first modeling pass, but its TF-IDF evaluations had a possible data leakage issue: `TfidfVectorizer` was fit on the full dataset before cross-validation. In the combined numeric + TF-IDF setup, numeric scaling was also fit before cross-validation.

The corrected script fixes this by putting all learned preprocessing inside scikit-learn objects that are passed directly to `cross_validate`:

- Text-only experiments use `Pipeline(TfidfVectorizer, classifier)`.
- Numeric-only Logistic Regression and Linear SVM use `Pipeline(SimpleImputer, StandardScaler, classifier)`.
- Combined numeric + TF-IDF experiments use `ColumnTransformer`, with numeric columns scaled in one branch and lyrics vectorized with TF-IDF in the other branch.
- No `fit_transform` is applied to the full dataset before cross-validation.

Corrected evaluation setup:

- Evaluation: 5-fold `StratifiedKFold`
- Shuffle: `True`
- Random state: 42
- Metrics: accuracy and macro-F1
- Total experiments: 80
- Removed constant/all-zero columns: `tense_present_count`, `tense_present_ratio`

### Old vs Corrected Best Result

| Version | Feature Group | Model | Accuracy | Macro-F1 |
|---------|---------------|-------|----------|----------|
| Old first experiment | `tfidf_bigrams` | Linear SVM | 0.6549 | 0.6513 |
| Corrected experiment | `tfidf_bigrams` | Linear SVM | 0.6332 | 0.6271 |

Difference:

- Accuracy drop: 0.0217
- Macro-F1 drop: 0.0242

Interpretation:

- The corrected results confirm that the previous experiment was slightly optimistic.
- The drop is meaningful but not catastrophic.
- The best model remains TF-IDF bigrams + Linear SVM.
- This suggests that word-pair patterns are still the strongest signal.
- However, the corrected result is not strong enough for us to consider the research question fully solved.
- Therefore, the next stage is to expand the dataset and rerun the experiment using the corrected evaluation pipeline.


## 9b. Hyperparameter Tuning Results

Tuning was performed on the top configurations from the baseline experiments.

| Config | Accuracy | Notes |
|--------|----------|-------|
| LR C=5, trigrams, 5000feat | 65.3% | Best tuned TF-IDF |
| LR C=10, bigrams, 3000feat, min_df=3 | 65.2% | Lowest variance |
| SVM C=2, trigrams, 3000feat, min_df=3 | 64.5% | |
| SVM C=1, bigrams, 10000feat | 63.8% | More features doesn't help |
| GB 300trees lr=0.1 d=5 (numeric) | 60.8% | Best numeric tuned |
| RF 500trees depth=20 (numeric) | 59.5% | |
| **Baseline: SVM + TF-IDF bigrams** | **65.5%** | **Still competitive** |

**Conclusion**: Hyperparameter tuning yields marginal improvement (+0.3-1.3pp). The default LinearSVC + TF-IDF bigrams (65.5%) remains the top performer. Key tuning insights: `min_df=3` and `max_df=0.9` help by removing rare/overly-common words. Trigrams add slight benefit but the gain is within noise.

## 9c. Feature Selection Analysis

Using SelectKBest (ANOVA F-statistic) with Logistic Regression to find the minimal optimal feature set:

| K features | Accuracy | vs All (132) |
|-----------|----------|--------------|
| 5 | 53.1% | -7.0pp |
| 10 | 51.4% | -8.7pp |
| 20 | 56.5% | -3.7pp |
| 30 | 58.8% | -1.3pp |
| 50 | 60.1% | +0.0pp |
| **80** | **61.3%** | **+1.2pp** |
| 100 | 60.5% | +0.3pp |
| 132 (all) | 60.1% | baseline |

**Best**: K=80 features achieves 61.3% — better than using all 132 features.

Top 10 features by F-statistic:

1. `repeated_words_count` (F=96.0)
2. `repeated_lines_count` (F=62.9)
3. `second_person_morph_count` (F=59.3)
4. `word_count` (F=58.7)
5. `second_person_count` (F=55.8)
6. `pos_adv_count` (F=53.8)
7. `lemma_diversity` (F=52.7)
8. `char_count` (F=51.7)
9. `pos_pron_count` (F=51.6)
10. `dicta_non_punct_token_count` (F=49.2)

**Conclusion**: ~40% of features can be dropped without accuracy loss. The strongest signals are repetition patterns, second-person language (both manual and Dicta-based), and song length. This aligns with the RF importance analysis.


## 10. Remaining Work

Completed:

1. ~~Run classification experiments~~ Done
2. ~~Compare feature groups~~ Done
3. ~~Compare models~~ Done
4. ~~Analyze confusion matrices~~ Done
5. ~~Identify which decade is easiest/hardest to classify~~ Done
6. ~~Correct evaluation leakage by moving TF-IDF and scaling into Pipeline / ColumnTransformer~~ Done

Remaining:

7. Expand the dataset beyond the current 597 songs.
8. Rerun corrected experiments on the expanded dataset.
9. Optionally add DictaBERT / newer Dicta analyzer features after the expanded dataset is ready.
10. Perform error analysis on misclassified songs.
11. Test model performance with and without suspicious technical features (`english_char_count`, `parenthesis_count`, `digit_count`).
12. Re-run or extend hyperparameter tuning using the corrected evaluation pipeline.
13. Add LLM-based analysis: LLM as feature extractor and LLM as direct classifier.
14. Compare classical ML results to LLM/AI results.
15. Add literature review.
16. Write final research report.

## 11. LLM / AI Extension Plan

The project should also include an AI-based approach.

### A. LLM as feature extractor

For each song, provide only the lyrics and ask the LLM to output structured JSON features such as:

- `main_theme`
- `emotional_tone`
- `nostalgia_level`
- `romantic_theme_level`
- `military_or_national_theme_level`
- `party_or_dance_energy`
- `poetic_language_level`
- `slang_or_spoken_language_level`
- `personal_confession_level`
- `direct_address_level`
- `social_or_historical_context_level`

These features can then be merged into the feature table and tested with the same ML models.

### B. LLM as direct classifier

Provide only the lyrics, without song name, artist, or year, and ask the LLM to classify the song as `1980s`, `2000s`, or `2020s` based on style. Then compare its accuracy and mistakes against the classical ML models.

Important leakage rule:

Never give the LLM `song_name`, `artist_name`, year, or `decade` when testing classification, because that would create data leakage.

## 12. Repository Structure

```text
.
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- israeli_songs_corpus.csv
|-- build_full_feature_table.py
|-- analyze_feature_table.py
|-- add_dicta_features.py
|-- analyze_dicta_enriched_table.py
|-- run_classification_experiments.py      <-- NEW
|-- run_classification_experiments_corrected.py
|-- classifier.py                          (deprecated)
|-- classifier_balanced_set.py             (deprecated)
|-- classifier_full_set.py                 (deprecated)
|-- [Hebrew project PDF]
|-- notebooks/
|   |-- song_decade_classification_experiments.ipynb   <-- NEW
|-- DictaAnalysis/
|   |-- dicta_analysis_code.ipynb
|   |-- data analysis.docx
|   |-- dicta-80s/
|   |-- dicta-00s/
|   |-- dicta-20s/
|   |-- dicta-output/
|-- outputs/
|   |-- song_feature_table.csv
|   |-- song_feature_table_with_dicta.csv
|   |-- classification_results_all.csv     <-- NEW
|   |-- classification_summary.json        <-- NEW
|   |-- classification_results_corrected.csv
|   |-- classification_summary_corrected.json
|   |-- feature_means_by_decade.csv
|   |-- top_decade_separating_features.csv
|   |-- feature_correlation_matrix.csv
|   |-- dicta_enriched_*.csv
|   |-- figures/
|   |-- figures_dicta/
```
