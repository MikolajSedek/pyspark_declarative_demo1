# PySpark Declarative Pipelines Demo

A demo project showcasing **Apache Spark Declarative Pipelines** (introduced in
Spark 4.0) with a clean, testable project structure.

> Reference: <https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html>

---

## Overview

Spark Declarative Pipelines let you define a data pipeline as a graph of
datasets using Python decorators (`@table`, `@materialized_view`,
`@temporary_view`).  The pipeline runner resolves dependencies automatically and
executes transformations in the correct order.

This repo contains **two complete pipelines**, both following the
[Palantir PySpark Style Guide](https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide).

---

## Pipeline 1 – User CSV Pipeline (`pipeline/demo_pipeline.py`)

A **Bronze → Silver → Gold** medallion pipeline for a fictional user dataset
loaded from CSV.

| Layer  | Dataset                      | Description                          |
|--------|------------------------------|--------------------------------------|
| Bronze | `bronze_users`               | Raw CSV ingestion                    |
| Silver | `silver_active_users`        | Filtered (active only) + `full_name` |
| Gold   | `gold_user_count_by_country` | Aggregated user count per country    |

---

## Pipeline 2 – Customer Parquet + SCD Pipeline (`pipeline/parquet_scd_pipeline.py`)

A **Bronze → Silver → Gold + Dimension** pipeline for a customer dataset loaded
from Parquet files, extended with Slowly Changing Dimension (SCD) tables.

| Layer     | Dataset                      | Description                                      |
|-----------|------------------------------|--------------------------------------------------|
| Bronze    | `bronze_customers`           | Raw Parquet ingestion                            |
| Silver    | `silver_valid_customers`     | Filtered (valid + active) + `full_name` + `revenue_tier` |
| Gold      | `gold_revenue_by_country`    | Customer count & revenue by country              |
| Gold      | `gold_revenue_by_segment`    | Customer count & revenue by country + tier       |
| Dimension | `dim_customers_scd1`         | SCD Type 1 – latest record per customer          |
| Dimension | `dim_customers_scd2`         | SCD Type 2 – full history with effective dates   |

### SCD Type 1 – Overwrite

Each `customer_id` has exactly **one row** representing the latest attribute
values.  Previous values are overwritten and not retained.  Implemented via a
window-function `ROW_NUMBER()` deduplicate on `updated_at`.

### SCD Type 2 – Track History

All historical attribute changes are preserved.  Each row carries:

| Column          | Description                               |
|-----------------|-------------------------------------------|
| `effective_from` | Timestamp when this version became active |
| `effective_to`   | Timestamp when it was superseded (`NULL` if current) |
| `is_current`     | `True` for the active version, `False` for historical |

The merge algorithm (implemented in pure PySpark without Delta Lake `MERGE`):

1. Deduplicate the incoming snapshot by `customer_id`, keeping the latest row.
2. Join against currently active rows in the existing SCD2 table.
3. Compute a SHA-256 hash of tracked attribute columns to detect changes.
4. **Close** existing active rows for changed customers (`effective_to`, `is_current=False`).
5. **Insert** new rows for new and changed customers (`effective_from`, `is_current=True`).
6. **Retain** all historical (non-current) rows unchanged.
7. **Carry forward** unchanged current rows.

---

## Project Layout

```
pyspark_declarative_demo1/
├── pipeline/
│   ├── __init__.py
│   ├── demo_pipeline.py          # Pipeline 1 – CSV users (Bronze→Silver→Gold)
│   ├── transformations.py        # Pure transformation functions for pipeline 1
│   ├── parquet_scd_pipeline.py   # Pipeline 2 – Parquet customers + SCD dims
│   └── scd_transformations.py    # Pure transformation functions for pipeline 2
├── tests/
│   ├── conftest.py               # Shared pytest fixtures (SparkSession)
│   ├── test_pipeline.py          # Unit tests for pipeline 1 transformations
│   └── test_scd_transformations.py  # Unit tests for pipeline 2 transformations
├── notebooks/
│   └── demo_pipeline.ipynb       # Interactive walkthrough of pipeline 1
├── pyproject.toml                # Project metadata, pytest & ruff configuration
├── requirements.txt              # Runtime + development dependencies
└── .pre-commit-config.yaml       # Pre-commit hooks (ruff lint + format)
```

---

## Requirements

- Python ≥ 3.10
- Java 11 or 17 (required by Apache Spark)
- PySpark ≥ 4.0

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/MikolajSedek/pyspark_declarative_demo1.git
cd pyspark_declarative_demo1

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Tests

The test suite uses [pytest](https://docs.pytest.org/) with a session-scoped
`SparkSession` fixture defined in `tests/conftest.py`.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pipeline --cov-report=term-missing
```

### What the tests cover

**`tests/test_pipeline.py`** (Pipeline 1 – CSV users)

- **`filter_active_users`** – inactive rows are excluded; active row values are unchanged; empty input yields empty output.
- **`enrich_with_full_name`** – `full_name` column is added and concatenated correctly.
- **`aggregate_user_count_by_country`** – correct columns and per-country counts.

**`tests/test_scd_transformations.py`** (Pipeline 2 – Parquet customers + SCD)

- **`filter_valid_customers`** – rows with null `customer_id` or `updated_at` are excluded.
- **`filter_active_customers`** – inactive rows are excluded.
- **`enrich_with_full_name`** – `full_name` column added and correct.
- **`enrich_with_revenue_tier`** – revenue tier boundaries (Platinum/Gold/Silver/Bronze) are all covered with parametrize.
- **`deduplicate_by_latest`** – latest row per key is retained; empty input is safe.
- **`aggregate_revenue_by_country`** – correct columns and per-country counts / totals.
- **`aggregate_revenue_by_segment`** – correct columns and group count.
- **`apply_scd_type1`** – one row per customer, latest values; empty input is safe.
- **`apply_scd_type2`** – first-run inserts, unchanged rows not duplicated, changed rows close old version and insert new, history is preserved, new customers are inserted, only one current row per customer.

---

## Running the Demo Notebook

```bash
pip install jupyter
jupyter notebook notebooks/demo_pipeline.ipynb
```

The notebook walks through each pipeline layer interactively using a local
`SparkSession`.

---

## Running the Declarative Pipelines

To run the pipelines via the Spark Pipelines CLI (requires a Spark Connect
server):

```bash
# Pipeline 1 – CSV users
spark-pipelines run pipeline/demo_pipeline.py

# Pipeline 2 – Parquet customers + SCD
spark-pipelines run pipeline/parquet_scd_pipeline.py
```

---

## Code Quality & Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) with
[ruff](https://docs.astral.sh/ruff/) for linting and formatting, following the
[Palantir PySpark Style Guide](https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide).

```bash
# Install hooks (one-time setup)
pre-commit install

# Run hooks manually against all files
pre-commit run --all-files
```

Hooks configured in `.pre-commit-config.yaml`:

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with a newline |
| `check-yaml` / `check-toml` | Validate config files |
| `check-merge-conflict` | Detect unresolved merge markers |
| `debug-statements` | Catch leftover `breakpoint()` / `pdb` calls |
| `ruff` | Fast Python linting (with auto-fix) |
| `ruff-format` | Opinionated Python formatting |

---

## License

Apache License 2.0

