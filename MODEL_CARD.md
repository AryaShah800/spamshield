# Model card: SpamShield primary model (character n-gram SVM)

| | |
| --- | --- |
| Model | TF-IDF character 2-5-grams (`char_wb`, min_df 2, sublinear tf) -> LinearSVC (class-balanced, C = 0.1 chosen on validation) -> `CalibratedClassifierCV` (3-fold, sigmoid) |
| Artifact | `models/spamshield_svm.joblib` (1.4 MB, not in git). Rebuild: `python benchmark.py --export-primary` |
| Data | UCI SMS Spam Collection (2011, English, mostly UK/Singapore SMS), 5,159 messages after removing duplicates and conflicting labels. 60/20/20 stratified split, seed 42; final model fit on train+validation (4,127) |
| Intended use | Learning/portfolio demonstration of a spam classifier with an honest benchmark. Scoring a short English SMS and showing which character n-grams drove the score |
| Not intended for | Production filtering, other languages, email, modern scam traffic (UPI/KYC/delivery phishing), Indian SMS formats, any decision about a person |

## Measured performance (1,032-message test split, 128 spam, threshold 0.5)

Source: `reports/benchmark.json` (`python benchmark.py`), `reports/calibration.json` and `reports/error_analysis.md` (`python scripts/error_analysis.py`).

| Metric | Value |
| --- | --- |
| Spam precision / recall / F1 | 0.976 / 0.938 / 0.956 (bootstrap 95% CI for F1: 0.927-0.980) |
| False positives / false negatives | 3 / 8 |
| PR-AUC | 0.992 |
| Calibration | ECE 0.0105 (10 equal-width bins), Brier 0.0074. Most mass is in the 0-0.1 and 0.9-1.0 bins; middle bins hold 1-5 messages each, so the reliability curve between them is noisy (`reports/calibration.png`) |
| Obfuscation recall (spam only) | clean 0.938, leetspeak 0.945, look-alike letters 0.836, dotted words 0.859, link tricks 0.930 |
| Batch-1 latency | about 4 ms median on a shared laptop CPU (indicative) |

Comparison: the fine-tuned MiniLM transformer scored F1 0.961 (single seed); McNemar p = 1.0 between the two, so they are not statistically distinguishable on this data, but the transformer's recall fell to 0.28 / 0.38 on look-alike letters / dotted words. That is why the simpler model ships (README).

## Errors (details in `reports/error_analysis.md`)

- 3 false positives: short, friendly messages that contain a word common in spam ("reply", "text", "free"). Probabilities 0.64-0.81, so a higher threshold would remove them, at a recall cost that was not evaluated.
- 8 false negatives: spam that reads like conversation or has no promotional vocabulary (a phone number appended to an apology, a "dating" message, a stock-tip pitch at 0.49) or that is a link-only pitch. Four of the eight sit between 0.25 and 0.49, close to the threshold.
- These are 11 errors in total; any statement about error *patterns* rests on very small counts.

## Limits and risks

- 2011 data: no knowledge of current scam styles. The corpus is easy (a bag-of-words model scores 0.93 F1), so these numbers will not transfer.
- One split, one seed for the transformer; the SVM is deterministic given the split. No out-of-domain evaluation was possible (no clearly licensed second corpus).
- Obfuscation tests are fixed, non-adaptive perturbations. A targeted attacker can do better.
- Explanations are linear-SVM contributions of character n-grams in margin units, not probabilities and not causal.
- False positives block legitimate messages; any real deployment needs a human-reviewable threshold and monitoring.
- The UCI collection contains real messages and some phone numbers; the repo includes only the 11 error messages shown in `reports/`. Messages sent to the API are not stored.
