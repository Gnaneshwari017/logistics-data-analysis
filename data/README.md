# Dataset guide

## Real dataset source

The project is configured to read `data/logistics_data.csv` by default. A suitable
public source is the [DataCo Smart Supply Chain for Big Data Analysis dataset on
Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).

To obtain it:

1. Open the Kaggle dataset listing and sign in to Kaggle if required.
2. Read the current dataset description and license/terms before downloading or
   using the data. This repository does not redistribute the original dataset.
3. Use the listing's **Download** option and extract the CSV locally.
4. Rename the CSV to `logistics_data.csv` and place it in this `data` folder.
   Alternatively, pass its path to `logistics_analysis.py --data`.
5. The DataCo file may contain personal or sensitive information and may use a
   different delimiter/encoding. Do not publish the original CSV. The analysis
   does not export a cleaned copy by default.

The pipeline detects common comma, semicolon, tab, and pipe delimiters. If
autodetection fails, specify one explicitly, for example
`--delimiter ";"`.

## Synthetic test data

`synthetic_logistics_sample.csv` is invented test data, clearly marked as
synthetic. It is included only to validate code, tests, charts, and report
generation. Its records, KPIs, plots, and model scores are **not real logistics
observations** and must not be used to make operational decisions.

## Privacy

The real dataset should remain local unless you have verified the license and
privacy requirements for redistribution. `data/logistics_data.csv` and common
archive formats are excluded by `.gitignore`.
