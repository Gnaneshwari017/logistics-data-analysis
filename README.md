# Strategic Planning and Data Exploration in Logistics Using Python

An adaptable, beginner-friendly Python project for exploring logistics and
supply-chain shipment data. It checks data quality, calculates delivery and
cost KPIs, creates charts when the input contains the needed columns, and
evaluates an optional late-delivery model without using delivery outcomes as
predictors.

> **Data note:** This repository includes a small, clearly labeled synthetic
> dataset for validating the code. Its records and generated results are not
> real logistics observations. No real shipment dataset was available in the
> project folder when this project was prepared.

## Problem statement

Logistics teams need to understand delivery reliability, shipping-mode usage,
delivery delays, and shipping costs. Data sources often differ in their column
names and completeness, so an analysis must explain what is unavailable and
avoid silently discarding or misrepresenting records.

## Objectives

- Inspect dataset structure, types, missing values, and duplicate rows.
- Clean recognized dates and operational numeric fields conservatively.
- Calculate total shipments, known on-time and late rates, available cost/time
  averages, and counts by shipping mode.
- Generate descriptive charts only when supported by the data.
- Train and evaluate a late-delivery classifier only when the data is adequate;
  exclude delivery outcomes and post-delivery measures from its predictors.
- Report whether clustering is meaningful and skip it when requirements are
  not met.
- Provide practical, appropriately qualified logistics recommendations.

## Technologies

Python 3.10+, pandas, NumPy, Matplotlib, scikit-learn, Jupyter, and pytest.

## Repository structure

```text
logistics-data-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── logistics_analysis.py
├── data/
│   ├── README.md
│   └── synthetic_logistics_sample.csv
├── outputs/
│   └── README.md
├── notebooks/
│   └── logistics_exploration.ipynb
├── reports/
│   └── logistics_analysis_report.md
└── tests/
    └── test_logistics_analysis.py
```

Generated charts, summaries, and reports are kept in `outputs/` and are not
ignored by Git. Real datasets and downloaded archives are ignored to reduce the
risk of publishing restricted or sensitive shipment information.

## Dataset source and setup

A suitable real-data source is the [DataCo Smart Supply Chain for Big Data
Analysis Kaggle listing](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
Kaggle may require an account. Read the current dataset description, license,
and usage terms before downloading or using it. This repository does not
redistribute the DataCo data.

1. Download the CSV using the Kaggle listing's download option and extract it.
2. Rename the CSV to `logistics_data.csv`.
3. Put it in `data\logistics_data.csv`, or pass its path with `--data`.
4. Keep real files private unless you have confirmed that you may share them.
   The program does not export a cleaned source CSV by default.

See [data/README.md](data/README.md) for full dataset and privacy notes. If no
real data is available yet, use the synthetic sample command below. Synthetic
outputs must not be reported as actual operational findings.

## Install on Windows

The commands below work in PowerShell and Windows Command Prompt from the
repository root. Use Python 3.10 or later.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable, install Python from [python.org](https://www.python.org/downloads/windows/)
or use the `python` command in the same commands above.

## Run the Python analysis

From the repository root, run the sample for a complete, clearly labeled test
of the pipeline:

```powershell
.\.venv\Scripts\python.exe logistics_analysis.py --data data\synthetic_logistics_sample.csv --data-kind synthetic
```

When the authorized real dataset is in the default location:

```powershell
.\.venv\Scripts\python.exe logistics_analysis.py
```

You can specify a different CSV and output folder:

```powershell
.\.venv\Scripts\python.exe logistics_analysis.py --data "C:\data\shipments.csv" --output-dir outputs --data-kind real
```

The CSV delimiter is detected when possible. For a semicolon-delimited file,
pass `--delimiter ";"`. Missing files, unreadable input, and unavailable
analysis fields are reported clearly. The script does not fabricate records
when the input is missing.

## Run the notebook

Install the dependencies above, then launch Jupyter from the repository root:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

Open `notebooks/logistics_exploration.ipynb` and run its cells in order. It uses
the labeled synthetic sample by default. Edit `DATA_PATH` and `DATA_KIND` in the
notebook to use a locally downloaded real CSV; do not label a real dataset as
synthetic or vice versa. Notebook code does not contain fabricated execution
results.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## KPIs and outputs

The program calculates total shipments, known on-time and late delivery rates,
average shipping cost and actual delivery time when recognized, average delay
when measurable, and shipment counts by shipping mode. Column matching accepts
common aliases rather than requiring one exact schema.

### Verified synthetic test run

The completed sample run processed 40 synthetic rows. It reported a 62.5%
on-time rate, 37.5% late rate, average shipping cost of 13.47 (source currency
unspecified), and average delivery time of 3.58 days. Shipment counts were 16
Standard Class, 10 Second Class, 8 First Class, and 6 Same Day. The optional
holdout classifier was trainable on this sample (accuracy 0.80; ROC AUC 0.79).
**These are results from synthetic test data only, not real operational
findings.** The generated report in `outputs/analysis_report.md` records the
sample designation and evaluation details.

The following files are generated when the input supports them:

- `outputs/data_summary.txt` — shape, column names and types, sample rows,
  missing values, duplicates, and recognized fields.
- `outputs/kpi_summary.csv` — KPI values and units.
- `outputs/delivery_status.png` — delivery outcomes/status distribution.
- `outputs/shipping_mode_analysis.png` — shipment counts by mode.
- `outputs/delivery_delay_analysis.png` — delay distribution.
- `outputs/shipping_cost_analysis.png` — shipping-cost distribution.
- `outputs/model_evaluation.txt` — holdout metrics or the reason modeling was
  skipped.
- `outputs/clustering_summary.txt` — cluster profiles or the reason clustering
  was skipped.
- `outputs/analysis_report.md` — run-specific analysis report.
- `reports/logistics_analysis_report.md` — project report, updated by each run.

Charts that require missing or empty fields are not created. No
`cleaned_data.csv` is exported by default, to avoid accidentally publishing
personal or sensitive source data. Details are in [outputs/README.md](outputs/README.md).

## Findings and limitations

The included sample is **synthetic**. Its KPIs, charts, and any model scores
serve only to demonstrate and test the implementation, not to describe actual
operations. Use the run-specific report to see the currently generated results
and their provenance. Replace the sample with an authorized real dataset before
making decisions.

Supported columns vary across datasets. Rows and duplicates are retained, but
invalid values in recognized numeric/date columns become missing and are
reported. A model requires enough labeled examples from both classes and uses a
stratified holdout split; it may be skipped for small or unsuitable data.
Holdout metrics do not guarantee future performance. Clustering is exploratory
and must be reviewed for operational meaning.

## Future improvements

- Validate performance on time-based splits and monitor model drift.
- Add route, carrier, warehouse, and service-level analyses when those
  attributes are available and safe to use.
- Add data validation rules and domain-specific unit/currency configuration.
- Confirm causal drivers with operational investigation before changing plans.
