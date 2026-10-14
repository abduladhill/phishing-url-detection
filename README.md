# Raw-URL phishing detection

This is a deliberately constrained research workflow: it trains from the `URL`
and `label` columns only. It does **not** use `URLSimilarityIndex`, TLD
probabilities, webpage/DOM values, titles, redirects, or any other source field
that a pasted raw URL cannot faithfully reproduce.

## Setup and training

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --data "C:\path\to\PhiUSIIL_Phishing_URL_Dataset.csv"
```

This writes to `artifacts/`:

- `raw_url_extra_trees.joblib` — trained classifier
- `feature_names.json` — exact, ordered model input list
- `metrics.json` — held-out evaluation results and split details

The evaluation uses `GroupShuffleSplit` with normalized hostname as the group.
No hostname appears in both train and test sets. Labels are interpreted as the
dataset defines them: `0 = phishing`, `1 = legitimate`.

## Run the interface

```powershell
streamlit run app.py
```

The interface calculates the exact same deterministic URL features before
prediction. It never fetches or visits entered links.

Before feature extraction, root URLs are canonicalized so that an optional
trailing slash does not change the result (`https://example.com` and
`https://example.com/` are treated equivalently).

## Features

The model uses URL/host/path/query lengths; punctuation counts (`.`, `/`, `-`,
`_`, `@`, `?`, `=`, `&`, `%`, `:`); digit, letter, and special-character ratios;
HTTPS; IP-literal host; a conservative subdomain count; port; punycode; and
percent-encoding presence. See `feature_names.json` for the authoritative order.

## Interpretation

Metrics printed after training are the only reported model performance. They are
not evidence of real-world protection: domain grouping reduces a key leakage
path, but an independent, recent test set is still needed before using this for
security decisions.
