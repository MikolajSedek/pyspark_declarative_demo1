"""Tests for src/python/io.py – schema enforcement and read correctness."""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.python.io import (
    CUSTOMER_PARQUET_SCHEMA,
    USER_CSV_SCHEMA,
    read_customers_parquet,
    read_users_csv,
)

# ---------------------------------------------------------------------------
# read_users_csv
# ---------------------------------------------------------------------------


def test_read_users_csv_schema(spark: SparkSession, tmp_path: Path) -> None:
    """read_users_csv must produce exactly USER_CSV_SCHEMA."""
    csv_file = tmp_path / "users.csv"
    csv_file.write_text(
        "id,first_name,last_name,country,is_active\n"
        "U001,Alice,Smith,US,true\n"
        "U002,Bob,Jones,UK,false\n",
        encoding="utf-8",
    )
    df = read_users_csv(spark, str(csv_file))
    assert df.schema == USER_CSV_SCHEMA


def test_read_users_csv_row_count(spark: SparkSession, tmp_path: Path) -> None:
    """read_users_csv must return one row per data line."""
    csv_file = tmp_path / "users.csv"
    csv_file.write_text(
        "id,first_name,last_name,country,is_active\n"
        "U001,Alice,Smith,US,true\n"
        "U002,Bob,Jones,UK,false\n",
        encoding="utf-8",
    )
    df = read_users_csv(spark, str(csv_file))
    assert df.count() == 2


def test_read_users_csv_empty(spark: SparkSession, tmp_path: Path) -> None:
    """read_users_csv returns empty DataFrame (not error) on header-only input."""
    csv_file = tmp_path / "users_empty.csv"
    csv_file.write_text("id,first_name,last_name,country,is_active\n", encoding="utf-8")
    df = read_users_csv(spark, str(csv_file))
    assert df.count() == 0
    assert df.schema == USER_CSV_SCHEMA


# ---------------------------------------------------------------------------
# read_customers_parquet
# ---------------------------------------------------------------------------

_MINIMAL_CUSTOMER_SCHEMA = StructType(
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


def test_read_customers_parquet_schema(spark: SparkSession, tmp_path: Path) -> None:
    """read_customers_parquet must produce exactly CUSTOMER_PARQUET_SCHEMA."""
    from datetime import datetime

    parquet_dir = tmp_path / "customers"
    data = [
        (
            "C001",
            "Alice",
            "Smith",
            "a@ex.com",
            "US",
            "NYC",
            "Retail",
            75_000.0,
            True,
            datetime(2024, 1, 1),
        )
    ]  # noqa: E501
    spark.createDataFrame(data, schema=_MINIMAL_CUSTOMER_SCHEMA).write.parquet(
        str(parquet_dir)
    )
    df = read_customers_parquet(spark, str(parquet_dir))
    assert df.schema == CUSTOMER_PARQUET_SCHEMA


def test_read_customers_parquet_row_count(spark: SparkSession, tmp_path: Path) -> None:
    """read_customers_parquet must return the correct number of rows."""
    from datetime import datetime

    parquet_dir = tmp_path / "customers2"
    ts = datetime(2024, 1, 1)
    data = [
        (
            "C001",
            "Alice",
            "Smith",
            "a@ex.com",
            "US",
            "NYC",
            "Retail",
            75_000.0,
            True,
            ts,
        ),  # noqa: E501
        ("C002", "Bob", "Jones", "b@ex.com", "UK", "London", "SMB", 5_000.0, True, ts),
    ]
    spark.createDataFrame(data, schema=_MINIMAL_CUSTOMER_SCHEMA).write.parquet(
        str(parquet_dir)
    )
    df = read_customers_parquet(spark, str(parquet_dir))
    assert df.count() == 2
