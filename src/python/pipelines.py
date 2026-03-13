"""Pipeline definitions for the PySpark Declarative Pipelines demo.

This module registers all datasets for both pipelines using the
``pyspark.pipelines`` API (introduced in Apache Spark 4.0).

Pipeline 1 – User CSV pipeline
    Reads user records from a CSV file and builds a Bronze → Silver → Gold
    medallion architecture.

    .. code-block:: text

        CSV file
             │
        bronze_users            (@table, raw CSV ingestion)
             │
        silver_active_users     (@materialized_view, filtered + full_name)
             │
        gold_user_count_by_country  (@materialized_view, country aggregation)

Pipeline 2 – Customer Parquet + SCD pipeline
    Reads customer records from Parquet files and builds a
    Bronze → Silver → Gold + Dimension architecture with Slowly Changing
    Dimension tables.

    .. code-block:: text

        Parquet files
             │
        bronze_customers            (@table, raw Parquet ingestion)
             │
        silver_valid_customers      (@materialized_view, filtered + enriched)
             │
        ┌────┴────┐
        │         │
        gold_revenue_by_country     (@materialized_view, country-level metrics)
        gold_revenue_by_segment     (@materialized_view, segment-level metrics)
             │
        dim_customers_scd1          (@table, SCD Type 1)
        dim_customers_scd2          (@table, SCD Type 2 with history)

Usage
-----
Call :func:`register_user_pipeline` or :func:`register_customer_pipeline`
once at startup, passing the active ``SparkSession``::

    from pyspark.sql import SparkSession
    from src.python.pipelines import register_user_pipeline, register_customer_pipeline

    spark = SparkSession.builder.getOrCreate()
    register_user_pipeline(spark)
    register_customer_pipeline(spark)

Reference:
    https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html
"""

from pyspark.errors import AnalysisException
from pyspark.pipelines import materialized_view, table
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.python.io import read_customers_parquet, read_users_csv
from src.python.transformations import (
    aggregate_revenue_by_country,
    aggregate_revenue_by_segment,
    aggregate_user_count_by_country,
    apply_scd_type1,
    apply_scd_type2,
    enrich_with_full_name,
    enrich_with_revenue_tier,
    filter_active_customers,
    filter_active_users,
    filter_valid_customers,
)

# ===========================================================================
# Pipeline 1 – User CSV pipeline
# Bronze → Silver → Gold medallion architecture
# ===========================================================================


def _compute_bronze_users(spark: SparkSession) -> DataFrame:
    """Read raw user CSV files into the Bronze layer."""
    return read_users_csv(spark)


def _compute_silver_active_users(spark: SparkSession) -> DataFrame:
    """Filter inactive users and add a ``full_name`` derived column."""
    bronze_df = spark.table("bronze_users")
    active_df = filter_active_users(bronze_df)
    return enrich_with_full_name(active_df)


def _compute_gold_user_count_by_country(spark: SparkSession) -> DataFrame:
    """Aggregate active users by country for reporting."""
    silver_df = spark.table("silver_active_users")
    return aggregate_user_count_by_country(silver_df)


def register_user_pipeline(spark: SparkSession) -> None:
    """Register all User CSV pipeline datasets with the Spark Pipelines runner.

    Registers Bronze, Silver, and Gold datasets for the CSV user pipeline.
    All datasets are discoverable by the pipeline runner after this call.

    Args:
        spark: The active ``SparkSession`` for this pipeline run.
    """

    # -----------------------------------------------------------------------
    # Bronze layer – raw CSV ingestion
    # -----------------------------------------------------------------------

    @table(comment="Raw user records ingested from the CSV source.")
    def bronze_users() -> DataFrame:
        return _compute_bronze_users(spark)

    # -----------------------------------------------------------------------
    # Silver layer – cleansed & enriched
    # -----------------------------------------------------------------------

    @materialized_view(comment="Active users enriched with a full_name column.")
    def silver_active_users() -> DataFrame:
        return _compute_silver_active_users(spark)

    # -----------------------------------------------------------------------
    # Gold layer – aggregated / business-ready
    # -----------------------------------------------------------------------

    @materialized_view(comment="Aggregated count of active users per country.")
    def gold_user_count_by_country() -> DataFrame:
        return _compute_gold_user_count_by_country(spark)


# ===========================================================================
# Pipeline 2 – Customer Parquet + SCD pipeline
# Bronze → Silver → Gold + Dimension architecture
# ===========================================================================


def _compute_bronze_customers(spark: SparkSession) -> DataFrame:
    """Read raw customer Parquet files into the Bronze layer."""
    return read_customers_parquet(spark)


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
                silver_df.schema.add("effective_from", "timestamp")
                .add("effective_to", "timestamp")
                .add("is_current", "boolean")
            ),
        )

    if "updated_at" in silver_df.columns:
        incoming_df = silver_df
    else:
        incoming_df = silver_df.withColumn("updated_at", F.current_timestamp())

    return apply_scd_type2(existing_scd2, incoming_df)


def register_customer_pipeline(spark: SparkSession) -> None:
    """Register all Customer Parquet + SCD pipeline datasets with the runner.

    Registers Bronze, Silver, Gold, and Dimension datasets for the Parquet
    customer pipeline.  All datasets are discoverable by the pipeline runner
    after this call.

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
