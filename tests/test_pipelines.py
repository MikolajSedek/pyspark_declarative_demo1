"""Smoke tests for pipeline registration functions.

These tests verify that the pipeline module can be imported and that the
registration functions are callable.  They also exercise the private
``_compute_*`` helpers by registering the required DataFrames as temporary
Spark views, which is the same mechanism the declarative runner uses to
resolve ``spark.table(...)`` calls.

Coverage
--------
* Module-level import (catches ``pyspark.pipelines`` absence early).
* ``register_user_pipeline`` – callable, returns ``None``.
* ``register_customer_pipeline`` – callable, returns ``None``.
* ``_compute_silver_active_users`` – correct columns, only active rows.
* ``_compute_gold_user_count_by_country`` – correct output columns.
* ``_compute_silver_valid_customers`` – correct columns, enrichments applied.
* ``_compute_gold_revenue_by_country`` – correct output columns.
* ``_compute_gold_revenue_by_segment`` – correct output columns.
* ``_compute_dim_customers_scd1`` – one row per customer_id.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.python.pipelines import (
    _compute_dim_customers_scd1,
    _compute_gold_revenue_by_country,
    _compute_gold_revenue_by_segment,
    _compute_gold_user_count_by_country,
    _compute_silver_active_users,
    _compute_silver_valid_customers,
    register_customer_pipeline,
    register_user_pipeline,
)

# ---------------------------------------------------------------------------
# Schemas reused for temp-view fixtures
# ---------------------------------------------------------------------------

_USER_SCHEMA = StructType(
    [
        StructField("id", StringType(), nullable=False),
        StructField("first_name", StringType(), nullable=True),
        StructField("last_name", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
    ]
)

_CUSTOMER_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), nullable=True),
        StructField("first_name", StringType(), nullable=True),
        StructField("last_name", StringType(), nullable=True),
        StructField("email", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("city", StringType(), nullable=True),
        StructField("customer_segment", StringType(), nullable=True),
        StructField("revenue_ytd", DoubleType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
        StructField("updated_at", TimestampType(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Temp-view fixtures (register DataFrames so spark.table() resolves)
# ---------------------------------------------------------------------------


@pytest.fixture()
def bronze_users_view(spark: SparkSession) -> None:
    """Register a minimal ``bronze_users`` temp view."""
    data = [
        ("1", "Alice", "Smith", "US", True),
        ("2", "Bob", "Jones", "US", False),
        ("3", "Carlo", "Rossi", "IT", True),
    ]
    spark.createDataFrame(data, schema=_USER_SCHEMA).createOrReplaceTempView(
        "bronze_users"
    )
    yield
    spark.catalog.dropTempView("bronze_users")


@pytest.fixture()
def silver_users_view(spark: SparkSession) -> None:
    """Register a minimal ``silver_active_users`` temp view."""
    data = [
        ("1", "Alice", "Smith", "US", True),
        ("3", "Carlo", "Rossi", "IT", True),
    ]
    spark.createDataFrame(data, schema=_USER_SCHEMA).createOrReplaceTempView(
        "silver_active_users"
    )
    yield
    spark.catalog.dropTempView("silver_active_users")


@pytest.fixture()
def bronze_customers_view(spark: SparkSession) -> None:
    """Register a minimal ``bronze_customers`` temp view."""
    from datetime import datetime

    ts = datetime(2024, 1, 1)
    data = [
        ("C001", "Alice", "Smith", "a@ex.com", "US", "NYC", "Retail", 75_000.0, True, ts),  # noqa: E501
        ("C002", "Bob", "Jones", "b@ex.com", "UK", "London", "SMB", 5_000.0, True, ts),
    ]
    spark.createDataFrame(data, schema=_CUSTOMER_SCHEMA).createOrReplaceTempView(
        "bronze_customers"
    )
    yield
    spark.catalog.dropTempView("bronze_customers")


@pytest.fixture()
def silver_customers_view(spark: SparkSession) -> None:
    """Register a minimal ``silver_valid_customers`` temp view (with enrichments)."""
    from datetime import datetime

    ts = datetime(2024, 1, 1)
    data = [
        ("C001", "Alice", "Smith", "a@ex.com", "US", "NYC", "Retail", 75_000.0, True, ts),  # noqa: E501
        ("C002", "Bob", "Jones", "b@ex.com", "UK", "London", "SMB", 5_000.0, True, ts),
    ]
    spark.createDataFrame(data, schema=_CUSTOMER_SCHEMA).createOrReplaceTempView(
        "silver_valid_customers"
    )
    yield
    spark.catalog.dropTempView("silver_valid_customers")


# ---------------------------------------------------------------------------
# Registration smoke tests
# ---------------------------------------------------------------------------


def test_register_user_pipeline_is_callable() -> None:
    """``register_user_pipeline`` must be a callable Python function."""
    assert callable(register_user_pipeline)


def test_register_customer_pipeline_is_callable() -> None:
    """``register_customer_pipeline`` must be a callable Python function."""
    assert callable(register_customer_pipeline)


def test_register_user_pipeline_returns_none(spark: SparkSession) -> None:
    """``register_user_pipeline`` must return ``None`` (it only registers datasets)."""
    result = register_user_pipeline(spark)
    assert result is None


def test_register_customer_pipeline_returns_none(spark: SparkSession) -> None:
    """``register_customer_pipeline`` must return ``None`` (registers datasets)."""
    result = register_customer_pipeline(spark)
    assert result is None


# ---------------------------------------------------------------------------
# User CSV pipeline – _compute_* helpers
# ---------------------------------------------------------------------------


def test_compute_silver_active_users_columns(
    spark: SparkSession, bronze_users_view: None
) -> None:
    """Silver users DataFrame must contain a ``full_name`` column."""
    result = _compute_silver_active_users(spark)
    assert "full_name" in result.columns


def test_compute_silver_active_users_only_active(
    spark: SparkSession, bronze_users_view: None
) -> None:
    """Silver users must contain only rows where ``is_active`` is True."""
    result = _compute_silver_active_users(spark)
    inactive = result.filter(~result["is_active"]).count()
    assert inactive == 0


def test_compute_gold_user_count_columns(
    spark: SparkSession, silver_users_view: None
) -> None:
    """Gold user-count DataFrame must have ``country`` and ``user_count`` columns."""
    result = _compute_gold_user_count_by_country(spark)
    assert set(result.columns) == {"country", "user_count"}


# ---------------------------------------------------------------------------
# Customer Parquet + SCD pipeline – _compute_* helpers
# ---------------------------------------------------------------------------


def test_compute_silver_valid_customers_columns(
    spark: SparkSession, bronze_customers_view: None
) -> None:
    """Silver customers must have ``full_name`` and ``revenue_tier`` columns."""
    result = _compute_silver_valid_customers(spark)
    assert "full_name" in result.columns
    assert "revenue_tier" in result.columns


def test_compute_gold_revenue_by_country_columns(
    spark: SparkSession, silver_customers_view: None
) -> None:
    """Gold revenue-by-country must have the expected four columns."""
    result = _compute_gold_revenue_by_country(spark)
    assert set(result.columns) == {
        "country",
        "customer_count",
        "total_revenue",
        "avg_revenue",
    }


def test_compute_gold_revenue_by_segment_columns(
    spark: SparkSession, silver_customers_view: None
) -> None:
    """Gold revenue-by-segment must have the expected four columns."""
    from src.python.transformations import enrich_with_revenue_tier

    enriched = enrich_with_revenue_tier(
        spark.table("silver_valid_customers")
    )
    enriched.createOrReplaceTempView("silver_valid_customers")
    result = _compute_gold_revenue_by_segment(spark)
    spark.catalog.dropTempView("silver_valid_customers")
    assert set(result.columns) == {
        "country",
        "revenue_tier",
        "customer_count",
        "total_revenue",
    }


def test_compute_dim_customers_scd1_one_row_per_customer(
    spark: SparkSession, silver_customers_view: None
) -> None:
    """SCD1 dimension must have exactly one row per ``customer_id``."""
    result = _compute_dim_customers_scd1(spark)
    total = result.count()
    distinct = result.select("customer_id").distinct().count()
    assert total == distinct
