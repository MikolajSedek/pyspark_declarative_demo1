"""Demo pipeline defined with Spark Declarative Pipelines.

This module uses the ``pyspark.pipelines`` API (introduced in Apache Spark 4.0)
to declare a multi-layer (Bronze → Silver → Gold) data pipeline.

Each decorated function is registered with the pipeline runner and executed
in dependency order when the pipeline runs via ``spark-pipelines run``.

Reference:
    https://spark.apache.org/docs/latest/declarative-pipelines-programming-guide.html
"""

from pyspark.pipelines import materialized_view, table
from pyspark.sql import SparkSession

from pipeline.transformations import (
    aggregate_user_count_by_country,
    enrich_with_full_name,
    filter_active_users,
)

# ---------------------------------------------------------------------------
# Bronze layer – raw ingestion
# ---------------------------------------------------------------------------


@table(
    comment="Raw user records ingested from the CSV source.",
)
def bronze_users() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Ingest raw user data from a CSV file into the Bronze layer."""
    spark = SparkSession.getActiveSession()
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv("data/users.csv")
    )


# ---------------------------------------------------------------------------
# Silver layer – cleansed & enriched
# ---------------------------------------------------------------------------


@materialized_view(
    comment="Active users enriched with a full_name column.",
)
def silver_active_users() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Filter inactive users and add a full_name derived column."""
    spark = SparkSession.getActiveSession()
    bronze_df = spark.table("bronze_users")
    active_df = filter_active_users(bronze_df)
    return enrich_with_full_name(active_df)


# ---------------------------------------------------------------------------
# Gold layer – aggregated / business-ready
# ---------------------------------------------------------------------------


@materialized_view(
    comment="Aggregated count of active users per country.",
)
def gold_user_count_by_country() -> "DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Aggregate active users by country for reporting."""
    spark = SparkSession.getActiveSession()
    silver_df = spark.table("silver_active_users")
    return aggregate_user_count_by_country(silver_df)
