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

Reference:
    https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html
    https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide
"""

from pyspark.pipelines import materialized_view, table
from pyspark.sql import SparkSession
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
# Bronze layer – raw Parquet ingestion
# ---------------------------------------------------------------------------


@table(
    comment=(
        "Raw customer records ingested from Parquet source files. "
        "No transformations are applied at this layer."
    ),
)
def bronze_customers() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Ingest raw customer data from Parquet files into the Bronze layer."""
    spark = SparkSession.getActiveSession()
    return spark.read.parquet("data/customers/")


# ---------------------------------------------------------------------------
# Silver layer – validated, cleansed, and enriched
# ---------------------------------------------------------------------------


@materialized_view(
    comment=(
        "Active, valid customer records enriched with a full_name column "
        "and a revenue_tier classification."
    ),
)
def silver_valid_customers() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Filter invalid/inactive customers and add derived columns."""
    spark = SparkSession.getActiveSession()
    bronze_df = spark.table("bronze_customers")
    valid_df = filter_valid_customers(bronze_df)
    active_df = filter_active_customers(valid_df)
    named_df = enrich_with_full_name(active_df)
    return enrich_with_revenue_tier(named_df)


# ---------------------------------------------------------------------------
# Gold layer – aggregated business metrics
# ---------------------------------------------------------------------------


@materialized_view(
    comment="Aggregated customer count and revenue metrics grouped by country.",
)
def gold_revenue_by_country() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Aggregate active customers by country for executive reporting."""
    spark = SparkSession.getActiveSession()
    silver_df = spark.table("silver_valid_customers")
    return aggregate_revenue_by_country(silver_df)


@materialized_view(
    comment=(
        "Aggregated customer count and revenue broken down by country "
        "and revenue tier (Bronze / Silver / Gold / Platinum)."
    ),
)
def gold_revenue_by_segment() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Aggregate active customers by country and revenue tier."""
    spark = SparkSession.getActiveSession()
    silver_df = spark.table("silver_valid_customers")
    return aggregate_revenue_by_segment(silver_df)


# ---------------------------------------------------------------------------
# Dimension layer – SCD tables
# ---------------------------------------------------------------------------


@table(
    comment=(
        "SCD Type 1 customer dimension. "
        "Each customer_id has exactly one row representing the latest state. "
        "Previous attribute values are overwritten and not retained."
    ),
)
def dim_customers_scd1() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Build the SCD Type 1 customer dimension from the Silver layer."""
    spark = SparkSession.getActiveSession()
    silver_df = spark.table("silver_valid_customers")
    return apply_scd_type1(silver_df)


@table(
    comment=(
        "SCD Type 2 customer dimension. "
        "All historical attribute changes are preserved. "
        "Each row carries effective_from, effective_to (NULL if current), "
        "and is_current to identify the active record for a customer."
    ),
)
def dim_customers_scd2() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Build the SCD Type 2 customer dimension from the Silver layer.

    On first execution the existing SCD2 table is empty; subsequent runs
    merge the Silver snapshot against the existing history.
    """
    spark = SparkSession.getActiveSession()
    silver_df = spark.table("silver_valid_customers")

    # Attempt to read the existing SCD2 table; fall back to an empty frame
    # with the required schema on the very first pipeline run.
    try:
        existing_scd2 = spark.table("dim_customers_scd2")
    except Exception:  # noqa: BLE001
        existing_scd2 = spark.createDataFrame(
            [],
            schema=silver_df.schema.add("effective_from", "timestamp").add(
                "effective_to", "timestamp"
            ).add("is_current", "boolean"),
        )

    # Attach a synthetic updated_at if the Silver view does not already
    # include one (e.g. when first bootstrapping).
    if "updated_at" in silver_df.columns:
        incoming_df = silver_df
    else:
        incoming_df = silver_df.withColumn("updated_at", F.current_timestamp())

    return apply_scd_type2(existing_scd2, incoming_df)
