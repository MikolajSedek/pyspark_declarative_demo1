"""Pure transformation functions for the demo pipeline.

These functions encapsulate the core business logic and are designed to be
testable independently of the pipeline runner.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def filter_active_users(df: DataFrame) -> DataFrame:
    """Keep only rows where the ``is_active`` flag is ``True``.

    Args:
        df: Input DataFrame with at least an ``is_active`` boolean column.

    Returns:
        DataFrame containing only active user rows.
    """
    return df.filter(F.col("is_active") == True)  # noqa: E712


def enrich_with_full_name(df: DataFrame) -> DataFrame:
    """Add a ``full_name`` column combining ``first_name`` and ``last_name``.

    Args:
        df: Input DataFrame with ``first_name`` and ``last_name`` string columns.

    Returns:
        DataFrame with an additional ``full_name`` column.
    """
    return df.withColumn(
        "full_name", F.concat(F.col("first_name"), F.lit(" "), F.col("last_name"))
    )


def aggregate_user_count_by_country(df: DataFrame) -> DataFrame:
    """Count users per country.

    Args:
        df: Input DataFrame with a ``country`` string column.

    Returns:
        DataFrame with ``country`` and ``user_count`` columns.
    """
    return df.groupBy("country").agg(F.count("*").alias("user_count"))
