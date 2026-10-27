# SpamShield — SMS Spam Classification

A local machine learning project that trains text classifiers, compares them against a majority baseline, and evaluates the selected model on an untouched test set. No API key or GPU required.

## Run on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python spamshield.py train
.\.venv\Scripts\python spamshield.py predict "Congratulations! Claim your free prize now"
.\.venv\Scripts\python spamshield.py predict "Please call me when you get home"
```

If environment creation reports an ensurepip error, bootstrap packages using the system pip:

```powershell
python -m pip --python .venv\Scripts\python.exe install -r requirements.txt
```

The first training run downloads the public dataset. Subsequent runs use the local copy. Predictions run locally after training.

## Experiment design

1. Validate records, normalize whitespace and case for duplicate detection, remove duplicate messages and conflicting label groups.
2. Create stratified 60% training, 20% validation, and 20% test partitions with seed 42.
3. Fit TF-IDF word unigrams and bigrams inside each model pipeline, using only training text.
4. Compare a majority predictor, Multinomial Naive Bayes, and class-balanced logistic regression on validation spam F1.
5. Refit the winning pipeline on training plus validation data; evaluate once on the held-out test set.
6. Save the model, metrics, dataset hash, runtime versions, confusion matrix and misclassified examples.

`reports/metrics.json` contains measured results. `reports/errors.csv` supports qualitative error analysis. The model is saved to `models/spamshield.joblib`; load only artifacts you trust.

## Why this is useful on a resume

This demonstrates dataset cleaning, NLP feature extraction, imbalanced classification, baseline comparison, leakage prevention, model selection, reproducibility and inference. The strongest evidence is a reproducible experiment plus an explanation of its mistakes.

## Learning walkthrough

1. Read `load_data()` and explain why duplicate messages crossing partitions inflate evaluation scores.
2. Read `train()` and distinguish training, validation, and test data.
3. Explain TF-IDF: words that distinguish a message receive more weight than common words. Bigrams capture pairs such as “free prize”.
4. Explain precision (how many flagged messages are spam), recall (how much spam is caught), and F1 (their harmonic mean). Accuracy alone can hide poor spam detection because ham is more common.
5. Inspect at least ten errors in `reports/errors.csv`; record patterns and suggest improvements without tuning against the test set.
6. Run your own examples and discuss why predictions can fail outside the dataset.

## Limitations and next experiments

This is an educational benchmark, not a production spam filter. The historical English SMS corpus does not establish performance on current scams, email, multilingual messages or Indian SMS traffic. Deduplication removes exact normalized matches, but near-duplicate templates may remain. The random split is not a temporal evaluation. Batch latency is a local measurement and excludes model loading.

For a stronger second version, predefine a cross-validation protocol on development data, compare character n-grams, inspect precision/recall tradeoffs, and collect a separate recent consented evaluation set. Avoid repeatedly using this test set to select improvements.

## Dataset and attribution

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CC84.

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Dataset license: CC BY 4.0. Downloaded raw data is kept out of Git. Do not treat benchmark messages as real contacts or redistribute personal details in demo screenshots.

## Resume use

Use the measured results in `reports/metrics.json`, not invented numbers. Review and understand the implementation before presenting it as your work. Describe your own experiments and contributions accurately, including AI assistance if asked.

### Verified first run

After deduplication, this experiment used 5,159 messages: 3,095 training, 1,032 validation and 1,032 test. Logistic regression won validation model selection. On the held-out test set it achieved **98.26% accuracy**, **91.67% spam precision**, **94.53% spam recall** and **93.08% spam F1**. There were 11 false positives and 7 false negatives. See the full metrics file for baseline comparisons and environment versions.

Example resume bullet, after you can explain and reproduce the project:

> Developed an SMS spam classification pipeline using TF-IDF and scikit-learn; compared a majority baseline, Naive Bayes and logistic regression, achieving 93.08% spam F1 on 1,032 held-out messages after deduplication.

`requirements-lock.txt` records the installed packages from this run. To reproduce this environment, use `python -m pip --python .venv\Scripts\python.exe install -r requirements-lock.txt`.

## Verification

Run `python -m unittest discover -s tests -v`. The offline suite checks deduplication, conflicting labels, invalid input and the train/save/reload/predict workflow. GitHub Actions runs these checks on pushes and pull requests. Synthetic test fixtures are used only for software checks; reported benchmark metrics come from the UCI data.

Raw data, local model artifacts and message-level error exports are excluded from Git. The aggregate benchmark report is included in `reports/metrics.json`.
