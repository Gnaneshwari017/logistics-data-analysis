# Dataset guide

## Real dataset source and local copy

The project is configured to read `data/logistics_data.csv` by default. A suitable
public source is the [DataCo Smart Supply Chain for Big Data Analysis dataset on
Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
The Kaggle dataset API currently lists its license as **CC0: Public Domain**.
The downloaded shipment CSV is approximately 95.9 MB. This copy includes
customer names, addresses, and identifiers, so the project keeps it local and
does not include it in the public Git repository despite the listing's CC0
label.

To obtain and prepare the source:

1. Obtain the dataset from its Kaggle listing and review the current license
   and terms yourself.
2. Use the listing's **Download** option and extract the archive locally.
3. Place `DataCoSupplyChainDataset.csv` in this folder as `logistics_data.csv`.
   The corresponding `DescriptionDataCoSupplyChain.csv` documents source
   columns. `tokenized_access_logs.csv` is clickstream data, not the primary
   shipment dataset.
4. Alternatively, pass the actual CSV path to `logistics_analysis.py --data`.
   Keep the raw file local; it is excluded by `.gitignore`.

The DataCo shipment CSV is comma-delimited. The pipeline also detects common
semicolon, tab, and pipe delimiters. If autodetection fails, specify one
explicitly, for example `--delimiter ";"`.

## Synthetic test data

`synthetic_logistics_sample.csv` is invented test data, clearly marked as
synthetic. It is included only to validate code, tests, charts, and report
generation. Its records, KPIs, plots, and model scores are **not real logistics
observations** and must not be used to make operational decisions.

## Privacy

`data/logistics_data.csv` and all downloaded CSVs and common archive formats
are excluded by `.gitignore`. Output samples suppress identifying columns, and
the analysis does not export the cleaned input data.
