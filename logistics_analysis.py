"""Exploratory logistics analysis with conservative data cleaning and optional ML."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "shipment_date": (
        "shipment date",
        "ship date",
        "order date",
        "purchase date",
        "date",
    ),
    "delivery_status": ("delivery status", "shipment status", "delivery outcome"),
    "shipping_mode": ("shipping mode", "shipment mode", "shipping method", "ship mode"),
    "shipping_cost": ("shipping cost", "freight cost", "freight", "shipping expense"),
    "delivery_time_days": (
        "days for shipping (real)",
        "actual delivery time",
        "delivery time days",
        "delivery time",
        "days to deliver",
    ),
    "scheduled_delivery_days": (
        "days for shipment (scheduled)",
        "scheduled delivery days",
        "estimated delivery days",
        "promised delivery days",
    ),
    "delivery_delay_days": (
        "delivery delay days",
        "delivery delay",
        "delay days",
        "late delivery days",
    ),
}

NUMERIC_NAME_HINTS = (
    "cost",
    "freight",
    "quantity",
    "amount",
    "price",
    "profit",
    "weight",
    "days",
    "delay",
    "time",
)
IDENTIFIER_HINTS = ("id", "identifier", "tracking", "customer", "product", "postal", "zip")
LEAKAGE_HINTS = (
    "delivery",
    "delay",
    "arrival",
    "actual",
    "status",
    "outcome",
    "target",
    "label",
)


def _normalized_name(value: Any) -> str:
    """Normalize column labels for tolerant, case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def identify_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    """Map supported canonical fields to original dataset column names."""
    normalized = {_normalized_name(column): str(column) for column in frame.columns}
    result: dict[str, str | None] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        result[canonical] = next(
            (normalized[_normalized_name(alias)] for alias in aliases
             if _normalized_name(alias) in normalized),
            None,
        )
    return result


def inspect_data(frame: pd.DataFrame, columns: dict[str, str | None]) -> str:
    """Build a readable, factual inspection summary without changing the frame."""
    lines = [
        f"Dataset shape: {frame.shape[0]} rows x {frame.shape[1]} columns",
        f"Column names: {', '.join(map(str, frame.columns)) or '(none)'}",
        "Data types:",
    ]
    lines.extend(f"  - {name}: {dtype}" for name, dtype in frame.dtypes.items())
    lines.append("Sample records (up to 5):")
    lines.append(frame.head().to_string(index=False) if not frame.empty else "  (no rows)")
    lines.append("Missing values by column:")
    missing = frame.isna().sum()
    lines.extend(f"  - {name}: {count}" for name, count in missing.items())
    lines.append(f"Duplicate rows: {int(frame.duplicated().sum())}")
    lines.append("Recognized logistics fields:")
    lines.extend(f"  - {name}: {value or 'not available'}" for name, value in columns.items())
    return "\n".join(lines) + "\n"


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standardize relevant types while retaining all original rows and columns."""
    cleaned = frame.copy()
    notes: list[str] = []
    columns = identify_columns(cleaned)

    for canonical in ("shipment_date",):
        column = columns[canonical]
        if column is not None:
            original_nonempty = cleaned[column].notna()
            converted = pd.to_datetime(cleaned[column], errors="coerce", format="mixed")
            invalid_count = int((original_nonempty & converted.isna()).sum())
            cleaned[column] = converted
            notes.append(f"Parsed {column!r} as a date; {invalid_count} non-empty value(s) were invalid.")

    # Convert recognizable operational measures, not arbitrary numeric-looking fields.
    numeric_columns: set[str] = {
        value
        for canonical, value in columns.items()
        if value is not None
        and canonical in {
            "shipping_cost",
            "delivery_time_days",
            "scheduled_delivery_days",
            "delivery_delay_days",
        }
    }
    for column in cleaned.columns:
        if any(hint in _normalized_name(column) for hint in NUMERIC_NAME_HINTS):
            numeric_columns.add(str(column))
    for column in numeric_columns:
        if column not in cleaned.columns:
            continue
        original_nonempty = cleaned[column].notna()
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        invalid_count = int((original_nonempty & converted.isna()).sum())
        if invalid_count or not pd.api.types.is_numeric_dtype(cleaned[column].dtype):
            cleaned[column] = converted
            notes.append(
                f"Converted {column!r} to numeric; {invalid_count} non-empty value(s) "
                "could not be parsed and are now missing."
            )

    duplicate_count = int(cleaned.duplicated().sum())
    notes.append(
        f"Retained all {len(cleaned)} rows; {duplicate_count} duplicate row(s) were identified "
        "but not removed."
    )
    notes.append("No rows or columns were silently dropped.")
    return cleaned, notes


def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def _delivery_delay(frame: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series | None:
    delay = _numeric_series(frame, columns["delivery_delay_days"])
    actual = _numeric_series(frame, columns["delivery_time_days"])
    scheduled = _numeric_series(frame, columns["scheduled_delivery_days"])
    if actual is not None and scheduled is not None:
        derived = actual - scheduled
        return delay.combine_first(derived) if delay is not None else derived
    return delay


def late_delivery_target(
    frame: pd.DataFrame, columns: dict[str, str | None] | None = None
) -> pd.Series:
    """Return 1 for late, 0 for known on-time/early, and NaN when unknown."""
    columns = columns or identify_columns(frame)
    target = pd.Series(np.nan, index=frame.index, dtype="float64")

    delay = _delivery_delay(frame, columns)
    if delay is not None:
        target.loc[delay.notna()] = (delay.loc[delay.notna()] > 0).astype(float)

    status_column = columns["delivery_status"]
    if status_column is not None:
        status = frame[status_column].astype("string").str.strip().str.lower()
        late = status.str.contains(r"\blate\b|\bdelay(?:ed)?\b", regex=True, na=False)
        on_time = status.str.contains(
            r"on[\s-]*time|advance|early|delivered", regex=True, na=False
        )
        known_status = (late | on_time) & target.isna()
        target.loc[known_status] = late.loc[known_status].astype(float)

    return target


def calculate_kpis(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute available shipment KPIs; unsupported values are None, not invented."""
    columns = identify_columns(frame)
    target = late_delivery_target(frame, columns)
    known = target.notna()
    late_rate = float(target.loc[known].mean()) if known.any() else None
    delay = _delivery_delay(frame, columns)
    cost = _numeric_series(frame, columns["shipping_cost"])
    delivery_time = _numeric_series(frame, columns["delivery_time_days"])
    mode_column = columns["shipping_mode"]
    mode_counts = (
        frame[mode_column].fillna("Unknown").astype(str).value_counts().sort_index().to_dict()
        if mode_column is not None
        else {}
    )
    return {
        "total_shipments": int(len(frame)),
        "on_time_delivery_rate": 1.0 - late_rate if late_rate is not None else None,
        "late_delivery_rate": late_rate,
        "delivery_rate_known_count": int(known.sum()),
        "average_shipping_cost": (
            float(cost.mean()) if cost is not None and cost.notna().any() else None
        ),
        "average_delivery_time_days": (
            float(delivery_time.mean())
            if delivery_time is not None and delivery_time.notna().any()
            else None
        ),
        "average_delivery_delay_days": (
            float(delay.mean()) if delay is not None and delay.notna().any() else None
        ),
        "shipment_count_by_shipping_mode": mode_counts,
        "delivery_status_counts": (
            frame[columns["delivery_status"]]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
            if columns["delivery_status"] is not None
            else {}
        ),
        "recognized_columns": columns,
    }


def _safe_numeric_features(
    frame: pd.DataFrame, columns: dict[str, str | None]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    excluded = {
        value
        for canonical, value in columns.items()
        if value is not None
        and canonical in {
            "delivery_status",
            "delivery_time_days",
            "scheduled_delivery_days",
            "delivery_delay_days",
        }
    }
    numeric_features: list[str] = []
    for column in frame.select_dtypes(include=["number", "bool"]).columns:
        normalized = _normalized_name(column)
        if column in excluded or any(hint in normalized for hint in IDENTIFIER_HINTS):
            continue
        if any(hint in normalized for hint in LEAKAGE_HINTS):
            continue
        if frame[column].nunique(dropna=True) > 1:
            numeric_features.append(str(column))

    categorical_features: list[str] = []
    mode = columns["shipping_mode"]
    if mode is not None and frame[mode].nunique(dropna=True) > 1:
        categorical_features.append(mode)

    features = frame[numeric_features + categorical_features].copy()
    date_column = columns["shipment_date"]
    if date_column is not None:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if dates.notna().sum() > 1:
            features["shipment_month"] = dates.dt.month
            features["shipment_day_of_week"] = dates.dt.dayofweek
            numeric_features.extend(["shipment_month", "shipment_day_of_week"])
    return features, numeric_features, categorical_features


def evaluate_classifier(
    frame: pd.DataFrame, columns: dict[str, str | None] | None = None
) -> tuple[str, dict[str, float] | None]:
    """Train and evaluate a leakage-conscious late-delivery classifier when feasible."""
    columns = columns or identify_columns(frame)
    target = late_delivery_target(frame, columns)
    labeled = target.notna()
    y = target.loc[labeled].astype(int)
    class_counts = y.value_counts()
    if len(y) < 20 or len(class_counts) != 2 or int(class_counts.min()) < 2:
        return (
            "Model not trained: at least 20 labeled shipments and at least two examples "
            "of each class are required for a holdout evaluation.",
            None,
        )

    features, numeric_features, categorical_features = _safe_numeric_features(frame, columns)
    features = features.loc[labeled]
    usable = features.notna().any(axis=1)
    features, y = features.loc[usable], y.loc[usable]
    if len(features) < 20 or y.nunique() != 2 or y.value_counts().min() < 2:
        return (
            "Model not trained: too few labeled rows have usable, non-leaking predictor features.",
            None,
        )
    if not numeric_features and not categorical_features:
        return "Model not trained: no suitable non-leaking predictor features were found.", None

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        return (
            "Model not trained: scikit-learn is not installed. Install requirements.txt "
            "to enable optional predictive modeling.",
            None,
        )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
    except ValueError as exc:
        return f"Model not trained: the available data could not support a safe split ({exc}).", None

    details = [
        "Model: Logistic Regression with median/mode imputation, numeric scaling, and one-hot encoding.",
        "Target: late delivery (1) versus known on-time/early delivery (0).",
        "Evaluation: stratified 75/25 train/test split; random_state=42.",
        "Leakage controls: delivery outcome, actual delivery time, schedule, and delay fields are excluded from predictors.",
        f"Predictors: {', '.join(numeric_features + categorical_features) or '(none)'}.",
        f"Train rows: {len(X_train)}; test rows: {len(X_test)}.",
        "Holdout metrics:",
    ]
    details.extend(f"  - {name}: {value:.4f}" for name, value in metrics.items())
    details.append("Metrics describe this dataset and split; they are not a guarantee of future performance.")
    return "\n".join(details), metrics


def evaluate_clusters(frame: pd.DataFrame) -> str:
    """Run exploratory K-means only when there are enough usable numeric measures."""
    columns = identify_columns(frame)
    blocked = {
        value
        for canonical, value in columns.items()
        if value is not None
        and canonical in {"delivery_status", "shipment_date"}
    }
    feature_names = [
        str(name)
        for name in frame.select_dtypes(include=["number"]).columns
        if name not in blocked
        and not any(hint in _normalized_name(name) for hint in IDENTIFIER_HINTS)
        and frame[name].nunique(dropna=True) > 1
    ]
    if len(feature_names) < 2 or len(frame) < 10:
        return (
            "Clustering not performed: at least 10 records and two varying numeric "
            "operational features are required."
        )
    try:
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return "Clustering not performed: scikit-learn is not installed."

    usable = frame[feature_names].notna().sum(axis=1) >= 2
    values = frame.loc[usable, feature_names]
    if len(values) < 10:
        return "Clustering not performed: fewer than 10 rows have usable operational measurements."
    cluster_count = min(3, len(values) // 3)
    if cluster_count < 2:
        return "Clustering not performed: too few usable records to form interpretable groups."
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KMeans(n_clusters=cluster_count, random_state=42, n_init=10),
    )
    labels = pipeline.fit_predict(values)
    summary = pd.DataFrame(values).assign(cluster=labels).groupby("cluster").mean(numeric_only=True)
    lines = [
        f"Exploratory K-means clustering used {len(values)} rows and {len(feature_names)} numeric feature(s): "
        f"{', '.join(feature_names)}.",
        "Features were median-imputed and standardized. Cluster IDs are descriptive, not quality rankings.",
        "Cluster mean profiles:",
        "\n".join(
            line.rstrip()
            for line in summary.to_string(float_format=lambda number: f"{number:.2f}").splitlines()
        ),
        "Validate whether these groupings are operationally meaningful before making decisions.",
    ]
    return "\n".join(lines)


def _save_charts(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    columns = identify_columns(frame)
    generated: list[str] = []

    def save_figure(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(name)

    status_column = columns["delivery_status"]
    target = late_delivery_target(frame, columns)
    if status_column is not None and frame[status_column].notna().any():
        counts = frame[status_column].fillna("Unknown").astype(str).value_counts()
        title = "Delivery status distribution"
    elif target.notna().any():
        counts = target.map({1.0: "Late", 0.0: "On time / early"}).fillna("Unknown").value_counts()
        title = "Delivery outcome (derived from available delay fields)"
    else:
        counts = None
    if counts is not None:
        counts.plot(kind="bar", color="#2878B5")
        plt.title(title)
        plt.xlabel("Status")
        plt.ylabel("Shipment count")
        plt.xticks(rotation=25, ha="right")
        save_figure("delivery_status.png")

    mode_column = columns["shipping_mode"]
    if mode_column is not None and frame[mode_column].notna().any():
        frame[mode_column].fillna("Unknown").astype(str).value_counts().plot(
            kind="bar", color="#55A868"
        )
        plt.title("Shipment count by shipping mode")
        plt.xlabel("Shipping mode")
        plt.ylabel("Shipment count")
        plt.xticks(rotation=25, ha="right")
        save_figure("shipping_mode_analysis.png")

    delay = _delivery_delay(frame, columns)
    if delay is not None and delay.notna().any():
        delay.dropna().plot(kind="hist", bins=min(20, max(5, int(np.sqrt(delay.notna().sum())))), color="#C44E52")
        plt.title("Delivery delay distribution")
        plt.xlabel("Delay in days (positive = later than schedule)")
        plt.ylabel("Shipment count")
        save_figure("delivery_delay_analysis.png")

    cost = _numeric_series(frame, columns["shipping_cost"])
    if cost is not None and cost.notna().any():
        cost.dropna().plot(kind="hist", bins=min(20, max(5, int(np.sqrt(cost.notna().sum())))), color="#8172B2")
        plt.title("Shipping cost distribution")
        plt.xlabel("Shipping cost")
        plt.ylabel("Shipment count")
        save_figure("shipping_cost_analysis.png")
    return generated


def _format_value(value: Any, percent: bool = False) -> str:
    if value is None:
        return "Not available in this dataset"
    if percent:
        return f"{value:.1%}"
    return f"{value:.2f}"


def _write_report(
    path: Path,
    source_name: str,
    data_kind: str,
    kpis: dict[str, Any],
    chart_names: list[str],
    model_text: str,
    cluster_text: str,
    clean_notes: list[str],
    chart_path_prefix: str = "",
) -> None:
    synthetic_note = (
        " **This is a synthetic testing dataset; its values are not real-world findings.**"
        if data_kind == "synthetic"
        else ""
    )
    mode_counts = kpis["shipment_count_by_shipping_mode"]
    mode_text = (
        ", ".join(f"{mode}: {count}" for mode, count in mode_counts.items())
        if mode_counts
        else "Not available"
    )
    status_counts = kpis["delivery_status_counts"]
    status_text = (
        ", ".join(f"{status}: {count}" for status, count in status_counts.items())
        if status_counts
        else "Not available"
    )
    chart_interpretations = {
        "delivery_status.png": (
            f"Observed delivery-status counts: {status_text}."
            if status_counts
            else "Displays the distribution of known delivery outcomes."
        ),
        "shipping_mode_analysis.png": (
            f"Observed shipment counts by mode: {mode_text}."
            if mode_counts
            else "Displays the shipment count for each recognized mode."
        ),
        "delivery_delay_analysis.png": (
            f"The plotted delay values are centered on an observed mean of "
            f"{_format_value(kpis['average_delivery_delay_days'])} days; positive values mean late."
        ),
        "shipping_cost_analysis.png": (
            f"The plotted costs have an observed mean of "
            f"{_format_value(kpis['average_shipping_cost'])} in the source currency."
        ),
    }
    chart_lines = (
        "\n".join(
            f"### {name.replace('_', ' ').replace('.png', '').title()}\n\n"
            f"![{name.replace('_', ' ').replace('.png', '')}]"
            f"({chart_path_prefix}{name})\n\n"
            f"{chart_interpretations[name]}"
            for name in chart_names
        )
        if chart_names
        else "No charts were produced because the required fields were unavailable or empty."
    )
    lines = [
        "# Logistics Analysis Report",
        "",
        "## Introduction and background",
        "This project explores shipment delivery performance, shipping mix, delivery delays, "
        "and shipping costs to support evidence-based logistics planning.",
        "",
        "## Problem statement and objectives",
        "Operational teams need a repeatable way to inspect shipment data, measure service and "
        "cost KPIs, and identify improvement opportunities. Objectives are to validate incoming "
        "data, calculate available KPIs, visualize supported measures, and evaluate a cautious "
        "late-delivery model only when data is sufficient.",
        "",
        "## Dataset description and source",
        f"- Input file: `{source_name}`",
        f"- Dataset designation: **{data_kind}**.{synthetic_note}",
        "- For real data, obtain the DataCo Smart Supply Chain dataset from the Kaggle dataset "
        "page listed in the project README and data guide. Follow the current listing's license "
        "and terms; this project does not bundle or redistribute that dataset.",
        "",
        "## Data cleaning methodology",
        "Column names are matched through documented aliases. Recognized dates and operational "
        "numeric measures are parsed with invalid non-empty values reported as missing. Original "
        "rows and columns are retained; duplicates are reported, not removed. No cleaned source "
        "CSV is exported by default to avoid publishing potentially sensitive records.",
        "",
        "Cleaning and data-quality notes:",
        *[f"- {note}" for note in clean_notes],
        "",
        "## KPI definitions and results",
        "- **Total shipments:** number of input rows; duplicate records are retained.",
        "- **On-time rate:** known on-time/early deliveries divided by shipments with a classifiable outcome.",
        "- **Late rate:** known late deliveries divided by shipments with a classifiable outcome.",
        "- **Average shipping cost:** arithmetic mean of parseable recognized shipping-cost values.",
        "- **Average delivery time:** arithmetic mean of parseable actual delivery-time values in days.",
        "- **Shipment count by mode:** rows per recognized shipping-mode category.",
        "",
        f"| KPI | Result |",
        "|---|---:|",
        f"| Total shipments | {kpis['total_shipments']} |",
        f"| On-time delivery rate | {_format_value(kpis['on_time_delivery_rate'], percent=True)} |",
        f"| Late delivery rate | {_format_value(kpis['late_delivery_rate'], percent=True)} |",
        f"| Outcome records with known status | {kpis['delivery_rate_known_count']} |",
        f"| Average shipping cost | {_format_value(kpis['average_shipping_cost'])} |",
        f"| Average delivery time (days) | {_format_value(kpis['average_delivery_time_days'])} |",
        f"| Average delay (days; positive means late) | {_format_value(kpis['average_delivery_delay_days'])} |",
        f"| Shipment count by mode | {mode_text} |",
        "",
        "## Exploratory data analysis and charts",
        chart_lines,
        "",
        "Interpretations are descriptive, not causal. Charts that could not be generated are "
        "omitted because their source fields were unavailable or empty.",
        "",
        "## Machine learning methodology and evaluation",
        model_text,
        "",
        "## Exploratory clustering",
        cluster_text,
        "",
        "## Key findings",
        (
            "The results above describe only this synthetic test sample and must not be interpreted "
            "as findings about real operations."
            if data_kind == "synthetic"
            else "The KPI table and available charts are the observed results for the supplied dataset. "
            "No causal claims are made."
        ),
        "",
        "## Recommendations",
        "- Review delay and on-time rates by route, mode, and service commitment before adjusting schedules.",
        "- Compare like-for-like shipping modes and lanes to identify cost/service trade-offs.",
        "- Investigate recurring late-delivery causes with operational context before changing staffing or capacity.",
        "- Validate any model on a later time period and monitor performance before operational use.",
        "- Protect customer identifiers, addresses, and other sensitive fields when handling real datasets.",
        "",
        "## Limitations",
        "- Source datasets may use different names, units, label definitions, and outcome coverage.",
        "- Unrecognized or absent fields cannot support the associated KPI or chart.",
        "- Duplicate rows are intentionally retained pending domain review.",
        "- Holdout model scores are dataset-specific and may not generalize; small datasets may not support a model.",
        "- Synthetic dataset results are for software verification only.",
        "",
        "## Conclusion",
        "The analysis is designed to adapt to available columns, expose data-quality limitations, "
        "and avoid presenting unavailable measures as facts. Replace the test sample with an "
        "authorized operational dataset for business conclusions.",
        "",
        "## References",
        "- DataCo Smart Supply Chain dataset listing: "
        "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
        "- pandas documentation: https://pandas.pydata.org/docs/",
        "- scikit-learn documentation: https://scikit-learn.org/stable/",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path, delimiter: str | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place an authorized CSV at this path, "
            "or pass --data with the CSV location."
        )
    if delimiter is None:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
            sample = source.read(8192)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    try:
        return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=delimiter, encoding="latin-1")


def run_analysis(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
    data_kind: str = "real",
    project_report_path: str | Path | None = "reports/logistics_analysis_report.md",
    delimiter: str | None = None,
) -> list[Path]:
    """Run the full analysis and write available results beneath output_dir."""
    path = Path(data_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = _read_csv(path, delimiter)
    columns = identify_columns(raw)
    summary = inspect_data(raw, columns)
    cleaned, notes = clean_data(raw)
    kpis = calculate_kpis(cleaned)

    (output / "data_summary.txt").write_text(
        summary + "\nCleaning notes:\n" + "\n".join(f"  - {note}" for note in notes) + "\n",
        encoding="utf-8",
    )
    kpi_rows = [
        ("total_shipments", kpis["total_shipments"], "shipments"),
        ("on_time_delivery_rate", kpis["on_time_delivery_rate"], "fraction"),
        ("late_delivery_rate", kpis["late_delivery_rate"], "fraction"),
        ("delivery_rate_known_count", kpis["delivery_rate_known_count"], "shipments"),
        ("average_shipping_cost", kpis["average_shipping_cost"], "source currency"),
        ("average_delivery_time_days", kpis["average_delivery_time_days"], "days"),
        ("average_delivery_delay_days", kpis["average_delivery_delay_days"], "days"),
    ]
    kpi_rows.extend(
        (f"shipments_by_mode:{mode}", count, "shipments")
        for mode, count in kpis["shipment_count_by_shipping_mode"].items()
    )
    pd.DataFrame(kpi_rows, columns=["kpi", "value", "unit"]).to_csv(
        output / "kpi_summary.csv", index=False
    )

    chart_names = _save_charts(cleaned, output)
    model_text, _ = evaluate_classifier(cleaned, identify_columns(cleaned))
    (output / "model_evaluation.txt").write_text(model_text + "\n", encoding="utf-8")
    cluster_text = evaluate_clusters(cleaned)
    (output / "clustering_summary.txt").write_text(cluster_text + "\n", encoding="utf-8")
    report_path = output / "analysis_report.md"
    _write_report(report_path, path.name, data_kind, kpis, chart_names, model_text, cluster_text, notes)
    created = [
        output / "data_summary.txt",
        output / "kpi_summary.csv",
        output / "model_evaluation.txt",
        output / "clustering_summary.txt",
        *[output / name for name in chart_names],
        report_path,
    ]
    if project_report_path is not None:
        project_path = Path(project_report_path)
        _write_report(
            project_path,
            path.name,
            data_kind,
            kpis,
            chart_names,
            model_text,
            cluster_text,
            notes,
            chart_path_prefix=Path(
                os.path.relpath(output.resolve(), project_path.parent.resolve())
            ).as_posix().rstrip("/")
            + "/",
        )
        created.append(project_path)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/logistics_data.csv",
        help="Input CSV path (default: data/logistics_data.csv)",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output folder (default: outputs)")
    parser.add_argument(
        "--data-kind",
        choices=("real", "synthetic"),
        default="real",
        help="Label the dataset provenance in reports; choose synthetic only for a test sample.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="CSV delimiter; auto-detected by default (for example: ';' or ',').",
    )
    args = parser.parse_args(argv)
    try:
        created = run_analysis(
            args.data,
            args.output_dir,
            args.data_kind,
            delimiter=args.delimiter,
        )
    except (FileNotFoundError, pd.errors.ParserError, OSError, ValueError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Analysis completed using {args.data} ({args.data_kind} dataset).")
    print("Generated files:")
    for file_path in created:
        print(f"  - {file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
