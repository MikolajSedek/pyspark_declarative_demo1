"""Declarative pipeline that loads Parquet files and builds SCD dimension tables.

This module defines a **Bronze → Silver → Gold + Dimension** pipeline using
the ``pyspark.pipelines`` API (Apache Spark 4.0).

Pipeline overview
-----------------
.. code-block:: text

    Parquet files
         │
    bronze_customers   (@table, raw Parquet ingestion)
         │
    silver_valid_customers  (@materialized_view, filtered + enriched)
         │
    ┌────┴────┐
    │         │
    gold_revenue_by_country   (@materialized_view, country-level aggregation)
    gold_revenue_by_segment   (@materialized_view, segment-level aggregation)
         │
    dim_customers_scd1   (@table, SCD Type 1 – latest record per customer)
    dim_customers_scd2   (@table, SCD Type 2 – full history with effective dates)

SCD Type 1 (``dim_customers_scd1``)
    Overwrites each customer's row with the most recent attribute values.
    No history is preserved.

SCD Type 2 (``dim_customers_scd2``)
    Maintains a full change history.  Each row carries ``effective_from``,
    ``effective_to`` (``NULL`` for the current row), and ``is_current``.

Usage
-----
The public entry point is :func:`register_pipeline`.  Pass the active
``SparkSession`` once; all pipeline datasets are registered via the
``@table`` / ``@materialized_view`` decorators inside the function scope,
so no ``getActiveSession()`` call is scattered through the file.

Example::

    from pyspark.sql import SparkSession
    from pipeline.parquet_scd_pipeline import register_pipeline

    spark = SparkSession.builder.getOrCreate()
    register_pipeline(spark)

Reference:
    https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html
    https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide
"""

from pyspark.errors import AnalysisException
from pyspark.pipelines import materialized_view, table
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline.scd_transformations import (
    aggregate_revenue_by_country,
    aggregate_revenue_by_segment,
    apply_scd_type1,
    apply_scd_type2,
    enrich_with_full_name,
    enrich_with_revenue_tier,
    filter_active_customers,
    filter_valid_customers,
)

# ---------------------------------------------------------------------------
# Layer computation helpers
# Each helper takes a SparkSession and returns a plain DataFrame.
# These are independently testable without the pipeline decorator.
# ---------------------------------------------------------------------------


def _compute_bronze_customers(spark: SparkSession) -> DataFrame:
    """Read raw customer Parquet files into the Bronze layer."""
    return spark.read.parquet("data/customers/")


def _compute_silver_valid_customers(spark: SparkSession) -> DataFrame:
    """Filter, validate, and enrich Bronze customer records."""
    bronze_df = spark.table("bronze_customers")
    valid_df = filter_valid_customers(bronze_df)
    active_df = filter_active_customers(valid_df)
    named_df = enrich_with_full_name(active_df)
    return enrich_with_revenue_tier(named_df)


def _compute_gold_revenue_by_country(spark: SparkSession) -> DataFrame:
    """Aggregate Silver customers by country."""
    return aggregate_revenue_by_country(spark.table("silver_valid_customers"))


def _compute_gold_revenue_by_segment(spark: SparkSession) -> DataFrame:
    """Aggregate Silver customers by country and revenue tier."""
    return aggregate_revenue_by_segment(spark.table("silver_valid_customers"))


def _compute_dim_customers_scd1(spark: SparkSession) -> DataFrame:
    """Build the SCD Type 1 customer dimension."""
    return apply_scd_type1(spark.table("silver_valid_customers"))


def _compute_dim_customers_scd2(spark: SparkSession) -> DataFrame:
    """Build the SCD Type 2 customer dimension.

    Loads the existing SCD2 history (empty DataFrame on the first run) and
    merges it against the current Silver snapshot.
    """
    silver_df = spark.table("silver_valid_customers")

    try:
        existing_scd2 = spark.table("dim_customers_scd2")
    except AnalysisException:
        existing_scd2 = spark.createDataFrame(
            [],
            schema=(
                silver_df.schema
                .add("effective_from", "timestamp")
                .add("effective_to", "timestamp")
                .add("is_current", "boolean")
            ),
        )

    if "updated_at" in silver_df.columns:
        incoming_df = silver_df
    else:
        incoming_df = silver_df.withColumn("updated_at", F.current_timestamp())

    return apply_scd_type2(existing_scd2, incoming_df)


# ---------------------------------------------------------------------------
# Pipeline registration
# The SparkSession is provided once; all @table / @materialized_view
# definitions close over it so getActiveSession() is not scattered.
# ---------------------------------------------------------------------------


def register_pipeline(spark: SparkSession) -> None:
    """Register all pipeline datasets with the Spark Pipelines runner.

    Call this function once at pipeline startup, passing the active
    ``SparkSession``.  All Bronze, Silver, Gold, and Dimension datasets are
    registered via ``@table`` / ``@materialized_view`` decorators and are
    subsequently discoverable by the pipeline runner.

    Args:
        spark: The active ``SparkSession`` for this pipeline run.
    """

    # -----------------------------------------------------------------------
    # Bronze layer – raw Parquet ingestion
    # -----------------------------------------------------------------------

    @table(
        comment=(
            "Raw customer records ingested from Parquet source files. "
            "No transformations are applied at this layer."
        ),
    )
    def bronze_customers() -> DataFrame:
        return _compute_bronze_customers(spark)

    # -----------------------------------------------------------------------
    # Silver layer – validated, cleansed, and enriched
    # -----------------------------------------------------------------------

    @materialized_view(
        comment=(
            "Active, valid customer records enriched with a full_name column "
            "and a revenue_tier classification."
        ),
    )
    def silver_valid_customers() -> DataFrame:
        return _compute_silver_valid_customers(spark)

    # -----------------------------------------------------------------------
    # Gold layer – aggregated business metrics
    # -----------------------------------------------------------------------

    @materialized_view(
        comment="Aggregated customer count and revenue metrics grouped by country.",
    )
    def gold_revenue_by_country() -> DataFrame:
        return _compute_gold_revenue_by_country(spark)

    @materialized_view(
        comment=(
            "Aggregated customer count and revenue broken down by country "
            "and revenue tier (Bronze / Silver / Gold / Platinum)."
        ),
    )
    def gold_revenue_by_segment() -> DataFrame:
        return _compute_gold_revenue_by_segment(spark)

    # -----------------------------------------------------------------------
    # Dimension layer – SCD tables
    # -----------------------------------------------------------------------

    @table(
        comment=(
            "SCD Type 1 customer dimension. "
            "Each customer_id has exactly one row representing the latest state. "
            "Previous attribute values are overwritten and not retained."
        ),
    )
    def dim_customers_scd1() -> DataFrame:
        return _compute_dim_customers_scd1(spark)

    @table(
        comment=(
            "SCD Type 2 customer dimension. "
            "All historical attribute changes are preserved. "
            "Each row carries effective_from, effective_to (NULL if current), "
            "and is_current to identify the active record for a customer."
        ),
    )
    def dim_customers_scd2() -> DataFrame:
        return _compute_dim_customers_scd2(spark)
