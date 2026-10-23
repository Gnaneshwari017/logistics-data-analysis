# Analysis outputs

These committed outputs were generated from the real DataCo shipment CSV
available locally as `data/logistics_data.csv`. The Kaggle listing's dataset
API reports CC0, but the approximately 95.9 MB raw CSV includes customer
identifying fields and is deliberately not committed.

Run `logistics_analysis.py` from the repository root to regenerate the data
summary, KPI CSV, delivery-by-mode summary, supported charts, model/clustering
evaluation files, and reports. The saved summary suppresses personal field
values. The raw dataset is never copied into this folder.

DataCo has no shipping-cost column, so there is no shipping-cost chart. Charts
are omitted when their source fields are unavailable or empty. All committed
metrics are real-dataset observations; synthetic sample results are not mixed
into these outputs.
