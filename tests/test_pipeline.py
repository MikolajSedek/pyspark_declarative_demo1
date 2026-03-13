"""Unit tests for pipeline transformation functions.

These tests exercise the pure-Python transformation logic in
``src.python.transformations`` using a local SparkSession provided by the
``spark`` fixture defined in ``conftest.py``.

Following pytest best-practices:
- Pure test functions (no test classes).
- ``pytest.fixture`` for reusable DataFrame inputs.
- ``pytest.mark.parametrize`` for data-driven assertions.
"""

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from src.python.transformations import (
    aggregate_user_count_by_country,
    enrich_with_full_name,
    filter_active_users,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

USER_SCHEMA = StructType(
    [
        StructField("id", StringType(), nullable=False),
        StructField("first_name", StringType(), nullable=True),
        StructField("last_name", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_users_df(spark: SparkSession) -> DataFrame:
    """DataFrame with a mix of active and inactive users across countries."""
    data = [
        ("1", "Alice", "Smith", "US", True),
        ("2", "Bob", "Jones", "US", False),
        ("3", "Carlo", "Rossi", "IT", True),
        ("4", "Diana", "Prince", "IT", True),
        ("5", "Ethan", "Brown", "UK", False),
    ]
    return spark.createDataFrame(data, schema=USER_SCHEMA)


@pytest.fixture()
def active_users_df(spark: SparkSession) -> DataFrame:
    """DataFrame containing only active users with varied names and countries."""
    data = [
        ("1", "Alice", "Smith", "US", True),
        ("3", "Carlo", "Rossi", "IT", True),
        ("4", "Diana", "Prince", "IT", True),
    ]
    return spark.createDataFrame(data, schema=USER_SCHEMA)


@pytest.fixture()
def empty_df(spark: SparkSession) -> DataFrame:
    """Empty DataFrame matching the user schema."""
    return spark.createDataFrame([], schema=USER_SCHEMA)


# ---------------------------------------------------------------------------
# filter_active_users
# ---------------------------------------------------------------------------


def test_filter_active_users_excludes_inactive(mixed_users_df: DataFrame) -> None:
    """Only active rows must be present in the output."""
    result = filter_active_users(mixed_users_df)
    assert result.count() == 3


def test_filter_active_users_no_inactive_flag_in_result(
    mixed_users_df: DataFrame,
) -> None:
    """No row in the result may have ``is_active == False``."""
    result = filter_active_users(mixed_users_df)
    inactive_count = result.filter(~result["is_active"]).count()
    assert inactive_count == 0


def test_filter_active_users_empty_input_returns_empty(empty_df: DataFrame) -> None:
    """An empty input DataFrame must produce an empty output DataFrame."""
    assert filter_active_users(empty_df).count() == 0


@pytest.mark.parametrize(
    ("row_id", "expected_country"),
    [
        ("1", "US"),
        ("3", "IT"),
        ("4", "IT"),
    ],
)
def test_filter_active_users_preserves_row_data(
    active_users_df: DataFrame, row_id: str, expected_country: str
) -> None:
    """Active rows must retain their original field values after filtering."""
    result = filter_active_users(active_users_df)
    row = result.filter(result["id"] == row_id).collect()[0]
    assert row["country"] == expected_country
    assert row["is_active"] is True


# ---------------------------------------------------------------------------
# enrich_with_full_name
# ---------------------------------------------------------------------------


def test_enrich_with_full_name_adds_column(active_users_df: DataFrame) -> None:
    """The output must contain a ``full_name`` column."""
    result = enrich_with_full_name(active_users_df)
    assert "full_name" in result.columns


@pytest.mark.parametrize(
    ("first_name", "last_name", "expected_full_name"),
    [
        ("Alice", "Smith", "Alice Smith"),
        ("Carlo", "Rossi", "Carlo Rossi"),
        ("Diana", "Prince", "Diana Prince"),
    ],
)
def test_enrich_with_full_name_concatenates_correctly(
    spark: SparkSession,
    first_name: str,
    last_name: str,
    expected_full_name: str,
) -> None:
    """``full_name`` must equal ``first_name`` + ' ' + ``last_name``."""
    data = [("x", first_name, last_name, "US", True)]
    df = spark.createDataFrame(data, schema=USER_SCHEMA)
    result = enrich_with_full_name(df)
    assert result.collect()[0]["full_name"] == expected_full_name


@pytest.mark.parametrize(
    ("first_name", "last_name", "expected_full_name"),
    [
        (None, "Smith", "Smith"),
        ("Alice", None, "Alice"),
        (None, None, ""),
    ],
)
def test_enrich_with_full_name_null_safe(
    spark: SparkSession,
    first_name: str | None,
    last_name: str | None,
    expected_full_name: str,
) -> None:
    """``full_name`` must not be null when one or both name parts are null.

    A null ``first_name`` or ``last_name`` must be silently omitted rather than
    propagating null to the output column.
    """
    data = [("x", first_name, last_name, "US", True)]
    df = spark.createDataFrame(data, schema=USER_SCHEMA)
    result = enrich_with_full_name(df)
    assert result.collect()[0]["full_name"] == expected_full_name


# ---------------------------------------------------------------------------
# aggregate_user_count_by_country
# ---------------------------------------------------------------------------


def test_aggregate_output_columns(active_users_df: DataFrame) -> None:
    """The result must contain exactly ``country`` and ``user_count`` columns."""
    result = aggregate_user_count_by_country(active_users_df)
    assert set(result.columns) == {"country", "user_count"}


@pytest.mark.parametrize(
    ("country", "expected_count"),
    [
        ("IT", 2),
        ("US", 1),
    ],
)
def test_aggregate_counts_per_country(
    active_users_df: DataFrame, country: str, expected_count: int
) -> None:
    """Each country must report the correct number of users."""
    result = aggregate_user_count_by_country(active_users_df)
    counts = {row["country"]: row["user_count"] for row in result.collect()}
    assert counts[country] == expected_count
