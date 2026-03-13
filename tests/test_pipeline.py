"""Unit tests for pipeline transformation functions.

These tests exercise the pure-Python transformation logic in
``pipeline.transformations`` using a local SparkSession provided by the
``spark`` fixture defined in ``conftest.py``.

Following pytest best-practices:
- One assertion per test where feasible.
- Descriptive test names that read as specifications.
- Minimal, self-contained fixture data created inline.
"""

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from pipeline.transformations import (
    aggregate_user_count_by_country,
    enrich_with_full_name,
    filter_active_users,
)

# ---------------------------------------------------------------------------
# Shared test data schema
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
# filter_active_users
# ---------------------------------------------------------------------------


class TestFilterActiveUsers:
    """Tests for the ``filter_active_users`` transformation."""

    def test_keeps_only_active_rows(self, spark: SparkSession) -> None:
        """Inactive rows must be excluded from the output."""
        data = [
            Row(
                id="1", first_name="Alice", last_name="A", country="US", is_active=True
            ),
            Row(id="2", first_name="Bob", last_name="B", country="UK", is_active=False),
        ]
        df = spark.createDataFrame(data, schema=USER_SCHEMA)

        result = filter_active_users(df)

        assert result.count() == 1

    def test_active_row_values_are_unchanged(self, spark: SparkSession) -> None:
        """Active rows must not have their data modified."""
        data = [
            Row(
                id="1", first_name="Alice", last_name="A", country="US", is_active=True
            ),
        ]
        df = spark.createDataFrame(data, schema=USER_SCHEMA)

        result = filter_active_users(df)

        row = result.collect()[0]
        assert row["id"] == "1"
        assert row["is_active"] is True

    def test_empty_dataframe_returns_empty(self, spark: SparkSession) -> None:
        """An empty input must yield an empty output."""
        df = spark.createDataFrame([], schema=USER_SCHEMA)

        result = filter_active_users(df)

        assert result.count() == 0


# ---------------------------------------------------------------------------
# enrich_with_full_name
# ---------------------------------------------------------------------------


class TestEnrichWithFullName:
    """Tests for the ``enrich_with_full_name`` transformation."""

    def test_full_name_column_is_added(self, spark: SparkSession) -> None:
        """The output DataFrame must contain a ``full_name`` column."""
        data = [
            Row(
                id="1",
                first_name="Alice",
                last_name="Smith",
                country="US",
                is_active=True,
            ),
        ]
        df = spark.createDataFrame(data, schema=USER_SCHEMA)

        result = enrich_with_full_name(df)

        assert "full_name" in result.columns

    def test_full_name_is_concatenated_correctly(self, spark: SparkSession) -> None:
        """``full_name`` must be ``first_name`` + space + ``last_name``."""
        data = [
            Row(
                id="1",
                first_name="Alice",
                last_name="Smith",
                country="US",
                is_active=True,
            ),
        ]
        df = spark.createDataFrame(data, schema=USER_SCHEMA)

        result = enrich_with_full_name(df)

        assert result.collect()[0]["full_name"] == "Alice Smith"


# ---------------------------------------------------------------------------
# aggregate_user_count_by_country
# ---------------------------------------------------------------------------


class TestAggregateUserCountByCountry:
    """Tests for the ``aggregate_user_count_by_country`` transformation."""

    def test_output_has_country_and_user_count_columns(
        self, spark: SparkSession
    ) -> None:
        """The result must contain exactly ``country`` and ``user_count``."""
        data = [
            Row(
                id="1", first_name="Alice", last_name="A", country="US", is_active=True
            ),
        ]
        df = spark.createDataFrame(data, schema=USER_SCHEMA)

        result = aggregate_user_count_by_country(df)

        assert set(result.columns) == {"country", "user_count"}

    def test_counts_are_correct_per_country(self, spark: SparkSession) -> None:
        """Each country must show the correct user count."""
        data = [
            Row(
                id="1", first_name="Alice", last_name="A", country="US", is_active=True
            ),
            Row(id="2", first_name="Bob", last_name="B", country="US", is_active=True),
            Row(
                id="3", first_name="Carlo", last_name="C", country="IT", is_active=True
            ),
        ]
        df = spark.createDataFrame(data, schema=USER_SCHEMA)

        result = aggregate_user_count_by_country(df)
        counts = {row["country"]: row["user_count"] for row in result.collect()}

        assert counts["US"] == 2
        assert counts["IT"] == 1
