"""I/O functions for the PySpark Declarative Pipelines demo.

This module contains all data-source reading and writing helpers used by the
pipeline definitions.  Keeping I/O operations here makes the pipeline layer
thin and the transformation layer fully independent of any external storage.

Explicit schemas are declared for every source so that:

* Schema inference (which requires a full file scan and produces
  non-deterministic types on sparse/dirty data) is avoided.
* The pipeline fails fast with a clear error when the source schema drifts,
  rather than silently propagating wrong types downstream.

Reference:
    Palantir PySpark Style Guide:
    https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Source schemas
# ---------------------------------------------------------------------------

#: Expected schema for the user CSV source.
USER_CSV_SCHEMA = StructType(
    [
        StructField("id", StringType(), nullable=True),
        StructField("first_name", StringType(), nullable=True),
        StructField("last_name", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
    ]
)

#: Expected schema for the customer Parquet source.
CUSTOMER_PARQUET_SCHEMA = StructType(
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


def read_users_csv(spark: SparkSession, path: str = "data/users.csv") -> DataFrame:
    """Read raw user records from a CSV file using an explicit schema.

    The CSV file must contain a header row.  The schema is enforced at read
    time (no inference), so any mismatch between the source and
    :data:`USER_CSV_SCHEMA` raises an ``AnalysisException`` immediately.

    Args:
        spark: Active ``SparkSession``.
        path:  Path to the CSV file.  Defaults to ``data/users.csv``.

    Returns:
        DataFrame containing all rows from the CSV file.
    """
    return spark.read.schema(USER_CSV_SCHEMA).option("header", "true").csv(path)


def read_customers_parquet(
    spark: SparkSession,
    path: str = "data/customers/",
) -> DataFrame:
    """Read raw customer records from Parquet files using an explicit schema.

    The schema is enforced at read time; any mismatch with
    :data:`CUSTOMER_PARQUET_SCHEMA` raises an ``AnalysisException`` immediately.

    Args:
        spark: Active ``SparkSession``.
        path:  Path to the directory containing Parquet part-files.
               Defaults to ``data/customers/``.

    Returns:
        DataFrame containing all rows from the Parquet source.
    """
    return spark.read.schema(CUSTOMER_PARQUET_SCHEMA).parquet(path)
