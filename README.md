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

This demo implements a classic **Bronze → Silver → Gold** medallion architecture
for a fictional user dataset:

| Layer  | Dataset                  | Description                              |
|--------|--------------------------|------------------------------------------|
| Bronze | `bronze_users`           | Raw CSV ingestion                        |
| Silver | `silver_active_users`    | Filtered (active only) + `full_name`     |
| Gold   | `gold_user_count_by_country` | Aggregated user count per country    |

---

## Project Layout

```
pyspark_declarative_demo1/
├── pipeline/
│   ├── __init__.py
│   ├── demo_pipeline.py      # Declarative pipeline definitions (@table / @materialized_view)
│   └── transformations.py    # Pure transformation functions (testable business logic)
├── tests/
│   ├── conftest.py           # Shared pytest fixtures (SparkSession)
│   └── test_pipeline.py      # Unit tests for transformation functions
├── notebooks/
│   └── demo_pipeline.ipynb   # Interactive walkthrough of the pipeline
├── pyproject.toml            # Project metadata, pytest & ruff configuration
├── requirements.txt          # Runtime + development dependencies
└── .pre-commit-config.yaml   # Pre-commit hooks (ruff lint + format)
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

- **`filter_active_users`** – inactive rows are excluded; active row values are unchanged; empty input yields empty output.
- **`enrich_with_full_name`** – `full_name` column is added and concatenated correctly.
- **`aggregate_user_count_by_country`** – correct columns and per-country counts.

---

## Running the Demo Notebook

```bash
pip install jupyter
jupyter notebook notebooks/demo_pipeline.ipynb
```

The notebook walks through each pipeline layer interactively using a local
`SparkSession`.

---

## Running the Declarative Pipeline

To run the full pipeline via the Spark Pipelines CLI (requires a Spark Connect
server):

```bash
spark-pipelines run pipeline/demo_pipeline.py
```

---

## Code Quality & Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) with
[ruff](https://docs.astral.sh/ruff/) for linting and formatting.

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
