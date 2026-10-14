"""Train a phishing classifier using only features computable from a URL string.

The source dataset contains many pre-engineered and page-content fields.  This
script deliberately ignores them: only URL and label are read.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length", "fragment_length",
    "dot_count", "slash_count", "hyphen_count", "underscore_count", "at_count",
    "question_count", "equals_count", "ampersand_count", "percent_count", "colon_count",
    "digit_count", "digit_ratio", "letter_ratio", "special_char_ratio", "https",
    "has_ip_host", "subdomain_count", "has_port", "has_punycode", "has_encoded_char",
]


def normalise_url(value: object) -> str:
    """Canonicalize harmless URL spelling differences used by browsers.

    In particular, https://example.com and https://example.com/ identify the
    same root resource. Treating them differently makes a lexical model brittle.
    """
    url = str(value).strip()
    if url and "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc and not parsed.path:
        url = parsed._replace(path="/").geturl()
    return url


def host_from_url(value: object) -> str:
    parsed = urlparse(normalise_url(value))
    return (parsed.hostname or "").lower().strip(".")


def is_ip_address(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return 0


def subdomain_count(host: str) -> int:
    if not host or is_ip_address(host):
        return 0
    # A fully correct public-suffix calculation needs an external suffix list.
    # This conservative count is intentionally raw-string-only and deterministic.
    labels = [part for part in host.split(".") if part]
    return max(0, len(labels) - 2)


def extract_features(value: object) -> dict[str, float | int]:
    """Extract the deployment-time feature vector from one raw URL."""
    url = normalise_url(value)
    parsed = urlparse(url)
    host = host_from_url(url)
    raw_no_scheme = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
    total = max(len(url), 1)
    letters = sum(char.isalpha() for char in url)
    digits = sum(char.isdigit() for char in url)
    specials = sum(not char.isalnum() for char in url)
    try:
        port_present = int(parsed.port is not None)
    except ValueError:  # e.g. an invalid or out-of-range literal port
        port_present = 1
    return {
        "url_length": len(url), "host_length": len(host), "path_length": len(parsed.path),
        "query_length": len(parsed.query), "fragment_length": len(parsed.fragment),
        "dot_count": url.count("."), "slash_count": raw_no_scheme.count("/"),
        "hyphen_count": url.count("-"), "underscore_count": url.count("_"), "at_count": url.count("@"),
        "question_count": url.count("?"), "equals_count": url.count("="), "ampersand_count": url.count("&"),
        "percent_count": url.count("%"), "colon_count": url.count(":"), "digit_count": digits,
        "digit_ratio": digits / total, "letter_ratio": letters / total,
        "special_char_ratio": specials / total, "https": int(parsed.scheme.lower() == "https"),
        "has_ip_host": is_ip_address(host), "subdomain_count": subdomain_count(host),
        "has_port": port_present, "has_punycode": int("xn--" in host),
        "has_encoded_char": int("%" in url),
    }


def feature_frame(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_features(url) for url in urls], columns=FEATURE_NAMES)


def evaluate(model: ExtraTreesClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    # In this source dataset label 0 is phishing and label 1 legitimate.
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "phishing_precision": float(precision_score(y_test, predictions, pos_label=0, zero_division=0)),
        "phishing_recall": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
        "phishing_f1": float(f1_score(y_test, predictions, pos_label=0, zero_division=0)),
        "legitimate_precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "legitimate_recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "legitimate_f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, predictions, target_names=["phishing", "legitimate"], zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV containing URL and label columns")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=["URL", "label"])
    data = data.dropna(subset=["URL", "label"]).copy()
    data["URL"] = data["URL"].astype(str)
    data["domain_group"] = data["URL"].map(host_from_url)
    # Keep malformed/no-host URLs isolated by URL rather than putting them all in one group.
    data.loc[data["domain_group"] == "", "domain_group"] = "__no_host__" + data.index.astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(data, data["label"], groups=data["domain_group"]))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    overlap = set(train["domain_group"]) & set(test["domain_group"])
    if overlap:
        raise RuntimeError(f"Domain leakage detected: {len(overlap)} groups overlap")

    x_train, x_test = feature_frame(train["URL"]), feature_frame(test["URL"])
    y_train, y_test = train["label"].astype(int), test["label"].astype(int)
    model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=args.random_state,
    )
    model.fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test)
    metrics.update({
        "dataset_rows": int(len(data)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "unique_domain_groups": int(data["domain_group"].nunique()), "overlapping_domain_groups": 0,
        "split_method": "GroupShuffleSplit grouped by normalized URL hostname", "random_state": args.random_state,
    })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "raw_url_extra_trees.joblib")
    (output / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
