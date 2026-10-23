# Strategic Planning and Data Exploration in Logistics Using Python

An adaptable, beginner-friendly Python project for exploring logistics and
supply-chain shipment data. It checks data quality, calculates delivery and
cost KPIs, creates charts when the input contains the needed columns, and
evaluates an optional late-delivery model without using delivery outcomes as
predictors.

> **Data note:** The current committed analysis outputs are generated from the
> DataCo Smart Supply Chain dataset. The raw source CSV remains local and is
> excluded from Git because it contains customer-identifying columns. A
> clearly labeled synthetic sample is included for tests only.

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

A suitable source is the [DataCo Smart Supply Chain for Big Data Analysis
Kaggle listing](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
Its Kaggle dataset API currently reports **CC0: Public Domain**. The downloaded
shipment CSV is about 95.9 MB and includes customer names, addresses, and
identifiers, so this project keeps it local and does not publish it.

The dataset files used for this analysis are `DataCoSupplyChainDataset.csv`
(the main shipment data), `DescriptionDataCoSupplyChain.csv` (field
definitions), and `tokenized_access_logs.csv` (clickstream data, not used as
the shipment dataset). To reproduce locally, download and extract the Kaggle
archive, then put the primary file at `data\logistics_data.csv`. The program
uses that path by default. Downloaded CSV files and archives are excluded by
`.gitignore`; generated reports suppress row-level identifying values.

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

Run the real DataCo analysis (after putting the CSV at the default path):

```powershell
.\.venv\Scripts\python.exe logistics_analysis.py
```

You can specify a different CSV and output folder:

```powershell
.\.venv\Scripts\python.exe logistics_analysis.py --data "C:\data\shipments.csv" --output-dir outputs --data-kind real
```

The program recognizes DataCo's actual column names, parses the order date,
uses the delivery status for outcomes, and derives delay as actual shipping
days minus scheduled shipping days. Canceled records are excluded from
on-time/late rates and delay averages. DataCo does not provide shipping-cost
values, so shipping cost is reported as unavailable, never estimated from
product revenue. The CSV delimiter is detected when possible. For a
semicolon-delimited file, pass `--delimiter ";"`.

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

## Real DataCo findings

The current outputs were generated from **180,519 DataCo line-item rows**.
They are not counts of unique physical shipments. No complete duplicate rows
were found; all rows were retained. The order dates parsed successfully.
The status field contained 98,977 late deliveries, 41,592 advance-shipping
records, 32,196 on-time records, and 7,754 canceled records. Canceled records
are excluded from service-rate denominators, leaving 172,765 known delivery
outcomes:

- **Late:** 98,977 / 172,765 = **57.29%**.
- **On time or early:** 73,788 / 172,765 = **42.71%**.
- **Average actual delivery time:** **3.50 days**.
- **Average delivery delay:** **0.57 days** (actual shipping days minus
  scheduled days; positive means late).
- **Average shipping cost:** unavailable in this source.

Shipment records by mode: Standard Class 107,752; Second Class 35,216;
First Class 27,814; Same Day 9,737. Among known outcomes, observed late rates
were First Class **100.0%** (26,513), Second Class **79.8%** (33,806), Same Day
**47.9%** (9,293), and Standard Class **39.8%** (103,153). These are
descriptive results; the striking First Class rate should be validated against
source status coding and operational definitions before decisions.

The Logistic Regression model used a **chronological 75/25 holdout by order
date**, with delivery status, the explicit late-risk label, actual/scheduled
delivery duration, and delay excluded from predictors. It evaluated 43,193
later records: accuracy **0.6953**, balanced accuracy **0.7217**, precision
**0.8839**, recall **0.5377**, F1 **0.6686**, and ROC AUC **0.7421**. These
scores are dataset-specific and do not guarantee future performance.

Data quality: 8 missing customer surnames, 3 missing customer ZIP codes,
155,679 missing order ZIP codes, and a product-description field missing in
all 180,519 rows. Identifying sample values are suppressed in the saved data
summary. See the run-specific [analysis report](outputs/analysis_report.md)
and [KPI summary](outputs/kpi_summary.csv).

## KPIs and outputs

The program calculates total shipments, known on-time and late delivery rates,
average shipping cost and actual delivery time when recognized, average delay
when measurable, and shipment counts by shipping mode. Column matching accepts
common aliases rather than requiring one exact schema.

The following files are generated when the input supports them:

- `outputs/data_summary.txt` — shape, column names and types, sample rows,
  missing values, duplicates, and recognized fields.
- `outputs/kpi_summary.csv` — KPI values and units.
- `outputs/delivery_performance_by_mode.csv` — shipment counts and observed
  delivery metrics by mode.
- `outputs/delivery_status.png` — delivery outcomes/status distribution.
- `outputs/shipping_mode_analysis.png` — shipment counts by mode.
- `outputs/late_delivery_rate_by_mode.png` — late rates by shipping mode.
- `outputs/delivery_delay_analysis.png` — delay distribution.
- `outputs/shipping_cost_analysis.png` — generated only when a shipping-cost
  column is present. DataCo has no such field.
- `outputs/model_evaluation.txt` — holdout metrics or the reason modeling was
  skipped.
- `outputs/clustering_summary.txt` — cluster profiles or the reason clustering
  was skipped.
- `outputs/analysis_report.md` — run-specific analysis report.
- `reports/logistics_analysis_report.md` — project report, updated by each run.

Charts that require missing or empty fields are not created. The previous
synthetic shipping-cost chart was removed because shipping cost is unavailable
in DataCo. No
`cleaned_data.csv` is exported by default, to avoid accidentally publishing
personal or sensitive source data. Details are in [outputs/README.md](outputs/README.md).

## Findings and limitations

Supported columns vary across datasets. Rows and duplicates are retained, but
invalid values in recognized numeric/date columns become missing and are
reported. For the DataCo run, all source rows were retained, and canceled
outcomes were left out of service rates. A model requires enough labeled
examples from both classes; where dates are available it uses a chronological
holdout. Holdout metrics do not guarantee future performance. Clustering used
a deterministic 20,000-row sample for speed and is exploratory. The bundled
synthetic sample and its test results are separate and must not be presented as
real-world findings.

## Future improvements

- Validate performance on time-based splits and monitor model drift.
- Add route, carrier, warehouse, and service-level analyses when those
  attributes are available and safe to use.
- Add data validation rules and domain-specific unit/currency configuration.
- Confirm causal drivers with operational investigation before changing plans.
