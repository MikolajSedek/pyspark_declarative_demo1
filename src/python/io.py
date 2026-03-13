"""I/O functions for the PySpark Declarative Pipelines demo.

This module contains all data-source reading and writing helpers used by the
pipeline definitions.  Keeping I/O operations here makes the pipeline layer
thin and the transformation layer fully independent of any external storage.

Reference:
    Palantir PySpark Style Guide:
    https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide
"""

from pyspark.sql import DataFrame, SparkSession


def read_users_csv(spark: SparkSession, path: str = "data/users.csv") -> DataFrame:
    """Read raw user records from a CSV file.

    The CSV file must contain a header row; schema is inferred automatically.

    Args:
        spark: Active ``SparkSession``.
        path:  Path to the CSV file.  Defaults to ``data/users.csv``.

    Returns:
        DataFrame containing all rows from the CSV file.
    """
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )


def read_customers_parquet(
    spark: SparkSession,
    path: str = "data/customers/",
) -> DataFrame:
    """Read raw customer records from Parquet files.

    Args:
        spark: Active ``SparkSession``.
        path:  Path to the directory containing Parquet part-files.
               Defaults to ``data/customers/``.

    Returns:
        DataFrame containing all rows from the Parquet source.
    """
    return spark.read.parquet(path)
