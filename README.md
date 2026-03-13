# PySpark Declarative Pipelines Demo

A demo project showcasing **Apache Spark Declarative Pipelines** (introduced in
Spark 4.0) with a clean, testable, and well-structured project architecture.

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

## Pipeline 1 – User CSV Pipeline

A **Bronze → Silver → Gold** medallion pipeline for a fictional user dataset
loaded from CSV.

| Layer  | Dataset                      | Description                          |
|--------|------------------------------|--------------------------------------|
| Bronze | `bronze_users`               | Raw CSV ingestion                    |
| Silver | `silver_active_users`        | Filtered (active only) + `full_name` |
| Gold   | `gold_user_count_by_country` | Aggregated user count per country    |

---

## Pipeline 2 – Customer Parquet + SCD Pipeline

A **Bronze → Silver → Gold + Dimension** pipeline for a customer dataset loaded
from Parquet files, extended with Slowly Changing Dimension (SCD) tables.

| Layer     | Dataset                      | Description                                                      |
|-----------|------------------------------|------------------------------------------------------------------|
| Bronze    | `bronze_customers`           | Raw Parquet ingestion                                            |
| Silver    | `silver_valid_customers`     | Filtered (valid + active) + `full_name` + `revenue_tier`         |
| Gold      | `gold_revenue_by_country`    | Customer count & revenue by country                              |
| Gold      | `gold_revenue_by_segment`    | Customer count & revenue by country + tier                       |
| Dimension | `dim_customers_scd1`         | SCD Type 1 – latest record per customer                          |
| Dimension | `dim_customers_scd2`         | SCD Type 2 – full history with effective dates                   |

### SCD Type 1 – Overwrite

Each `customer_id` has exactly **one row** representing the latest attribute
values.  Previous values are overwritten and not retained.  Implemented via a
window-function `ROW_NUMBER()` deduplicate on `updated_at`.

### SCD Type 2 – Track History

All historical attribute changes are preserved.  Each row carries:

| Column           | Description                                            |
|------------------|--------------------------------------------------------|
| `effective_from` | Timestamp when this version became active              |
| `effective_to`   | Timestamp when it was superseded (`NULL` if current)   |
| `is_current`     | `True` for the active version, `False` for historical  |

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
├── src/
│   ├── python/
│   │   ├── __init__.py
│   │   ├── io.py               # I/O helpers – read CSV / Parquet sources
│   │   ├── transformations.py  # All pure DataFrame transformations (both pipelines)
│   │   ├── utils.py            # Shared utilities: deduplication, SCD2 helpers
│   │   └── pipelines.py        # Pipeline definitions (register_user_pipeline,
│   │                           #   register_customer_pipeline)
│   └── diagrams/
│       ├── c4_context.drawio   # C4 Level 1 – System Context diagram
│       ├── c4_container.drawio # C4 Level 2 – Container diagram
│       └── c4_pipelines.drawio # C4 Level 3 – Component / Pipelines diagram
├── tests/
│   ├── conftest.py             # Shared pytest fixtures (SparkSession)
│   ├── test_pipeline.py        # Unit tests for Pipeline 1 transformations
│   └── test_scd_transformations.py  # Unit tests for Pipeline 2 transformations
├── notebooks/
│   └── demo_pipeline.ipynb     # Interactive walkthrough of Pipeline 1
├── pyproject.toml              # Project metadata, pytest & ruff configuration
└── .pre-commit-config.yaml     # Pre-commit hooks (ruff lint + format)
```

### Module responsibilities

| Module                          | Responsibility                                                         |
|---------------------------------|------------------------------------------------------------------------|
| `src/python/io.py`              | Read raw data from CSV and Parquet sources                             |
| `src/python/transformations.py` | All pure filtering, enrichment, aggregation, and SCD functions         |
| `src/python/utils.py`           | Generic helpers: deduplication, SCD2 building-blocks                   |
| `src/python/pipelines.py`       | Pipeline registration using `@table` / `@materialized_view` decorators |

---

## Architecture Diagrams (C4)

C4 draw.io diagrams are available in `src/diagrams/`.  Open them with
[draw.io](https://app.diagrams.net/) or the VS Code draw.io extension.

| File                   | Level      | Description                               |
|------------------------|------------|-------------------------------------------|
| `c4_context.drawio`    | C4 Level 1 | System context – actors and data sources  |
| `c4_container.drawio`  | C4 Level 2 | Containers – Python modules and their roles |
| `c4_pipelines.drawio`  | C4 Level 3 | Component – both pipelines layer by layer |

---

## Requirements

- Python ≥ 3.12
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

# 3. Install dependencies (including dev tools)
pip install -e ".[dev]"
```

---

## Running the Tests

The test suite uses [pytest](https://docs.pytest.org/) with a session-scoped
`SparkSession` fixture defined in `tests/conftest.py`.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/python --cov-report=term-missing
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
spark-pipelines run src/python/pipelines.py --pipeline register_user_pipeline

# Pipeline 2 – Parquet customers + SCD
spark-pipelines run src/python/pipelines.py --pipeline register_customer_pipeline
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

| Hook                        | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `trailing-whitespace`       | Remove trailing whitespace                   |
| `end-of-file-fixer`         | Ensure files end with a newline              |
| `check-yaml` / `check-toml` | Validate config files                        |
| `check-merge-conflict`      | Detect unresolved merge markers              |
| `debug-statements`          | Catch leftover `breakpoint()` / `pdb` calls  |
| `ruff`                      | Fast Python linting (with auto-fix)          |
| `ruff-format`               | Opinionated Python formatting                |

---

## Continuous Integration (GitHub Actions)

Every push and pull request triggers the CI pipeline defined in
`.github/workflows/ci.yml`.  The workflow contains two parallel jobs:

### `pre-commit` job

Runs all hooks from `.pre-commit-config.yaml` against the entire codebase using
[`pre-commit/action@v3.0.1`](https://github.com/pre-commit/action).
Hook environments are cached automatically on the `.pre-commit-config.yaml`
hash, so subsequent runs are fast.

| Check           | Tool            | Purpose                                 |
|-----------------|-----------------|-----------------------------------------|
| Whitespace/EOF  | pre-commit-hooks | Formatting hygiene                     |
| YAML / TOML     | pre-commit-hooks | Config file validation                  |
| Merge conflicts | pre-commit-hooks | Detect unresolved markers               |
| Lint + isort    | ruff            | PEP 8 compliance, import ordering       |
| Format          | ruff-format     | Black-compatible formatting             |
| Security        | bandit          | Common security anti-patterns           |
| Docstrings      | interrogate     | ≥ 80 % docstring coverage               |
| Secrets         | detect-secrets  | Prevent accidental credential commits   |

### `tests` job

Runs the full pytest suite on **Python 3.12** on `ubuntu-latest`.  Java 17
(Temurin) is installed because Apache Spark requires a JVM.  Coverage XML is
uploaded as a workflow artifact for review.

```
ubuntu-latest × Python 3.12 ── pytest --cov=src/python
                                       (coverage XML uploaded as artifact)
```

The two jobs run **in parallel** for fastest total feedback.  Each job cancels
any previous in-progress run for the same branch (`concurrency` group), saving
CI minutes on rapid successive pushes.

---

## License

Apache License 2.0
