"""Unit tests for the SCD pipeline transformation functions.

These tests exercise every pure-Python transformation function defined in
``src.python.transformations`` and ``src.python.utils`` using a local
SparkSession provided by the ``spark`` fixture in ``conftest.py``.

Test organisation
-----------------
* **Fixtures** – reusable DataFrames created once per test function.
* **Pure assertions** – each ``test_*`` function has a single responsibility.
* ``pytest.mark.parametrize`` is used wherever the same logic should be
  verified across multiple inputs.

Coverage
--------
* ``filter_valid_customers``
* ``filter_active_customers``
* ``enrich_with_full_name``
* ``enrich_with_revenue_tier``
* ``deduplicate_by_latest``
* ``aggregate_revenue_by_country``
* ``aggregate_revenue_by_segment``
* ``apply_scd_type1``
* ``apply_scd_type2``
"""

from datetime import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.python.transformations import (
    aggregate_revenue_by_country,
    aggregate_revenue_by_segment,
    apply_scd_type1,
    apply_scd_type2,
    enrich_with_full_name,
    enrich_with_revenue_tier,
    filter_active_customers,
    filter_valid_customers,
)
from src.python.utils import SCD2_TRACKED_COLUMNS, deduplicate_by_latest

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

CUSTOMER_SCHEMA = StructType(
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

SCD2_SCHEMA = StructType(
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
        StructField("effective_from", TimestampType(), nullable=True),
        StructField("effective_to", TimestampType(), nullable=True),
        StructField("is_current", BooleanType(), nullable=True),
        StructField("hash_version", IntegerType(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Helper timestamps
# ---------------------------------------------------------------------------

TS_JAN = datetime(2024, 1, 1)
TS_FEB = datetime(2024, 2, 1)
TS_MAR = datetime(2024, 3, 1)

# ---------------------------------------------------------------------------
# Shared row builder
# ---------------------------------------------------------------------------


def _cust(
    cid: str | None,
    first: str,
    last: str,
    country: str,
    city: str,
    segment: str,
    revenue: float,
    active: bool,
    ts: datetime | None,
) -> tuple:
    """Return a CUSTOMER_SCHEMA-ordered tuple with a derived email."""
    email = f"{first[0].lower()}@ex.com" if first and len(first) > 0 else "x@ex.com"
    return (cid, first, last, email, country, city, segment, revenue, active, ts)


# ---------------------------------------------------------------------------
# Pre-built customer rows (module-level constants reused across fixtures)
# ---------------------------------------------------------------------------

_ALICE_JAN = _cust(
    "C001", "Alice", "Smith", "US", "NYC", "Retail", 75_000.0, True, TS_JAN
)
_ALICE_FEB = _cust(
    "C001", "Alice", "Smith", "US", "NYC", "Retail", 80_000.0, True, TS_FEB
)
_ALICE_MAR = _cust(
    "C001", "Alice", "Smith", "US", "NYC", "Retail", 90_000.0, True, TS_MAR
)
_ALICE_MAR_99 = _cust(
    "C001", "Alice", "Smith", "US", "NYC", "Retail", 99_000.0, True, TS_MAR
)
_BOB_JAN = _cust("C002", "Bob", "Jones", "US", "LA", "SMB", 20_000.0, False, TS_JAN)
_BOB_UK_JAN = _cust(
    "C002", "Bob", "Jones", "UK", "London", "SMB", 20_000.0, True, TS_JAN
)
_BOB_UK_MAR = _cust(
    "C002", "Bob", "Jones", "UK", "London", "SMB", 20_000.0, True, TS_MAR
)
_BOB_UK_MAR_21 = _cust(
    "C002", "Bob", "Jones", "UK", "London", "SMB", 21_000.0, True, TS_MAR
)
_CARLO_JAN = _cust(
    "C003", "Carlo", "Rossi", "IT", "Rome", "Enterprise", 150_000.0, True, TS_JAN
)
_CARLO_NULL_TS = _cust(
    "C003", "Carlo", "Rossi", "IT", "Rome", "Enterprise", 150_000.0, True, None
)
_DIANA_JAN = _cust(
    "C004", "Diana", "Prince", "IT", "Milan", "Retail", 8_000.0, True, TS_JAN
)
_ETHAN_JAN = _cust(
    "C005", "Ethan", "Brown", "UK", "London", "SMB", 5_000.0, False, TS_JAN
)
_ZOE_MAR = _cust(
    "C999", "Zoe", "Zhao", "CN", "Beijing", "Enterprise", 200_000.0, True, TS_MAR
)
_NULL_ID_JAN = _cust(None, "Bob", "Jones", "US", "LA", "SMB", 20_000.0, True, TS_JAN)

# ---------------------------------------------------------------------------
# Fixtures – customer DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def customers_with_nulls_df(spark: SparkSession) -> DataFrame:
    """DataFrame with rows that have null customer_id or null updated_at."""
    data = [_ALICE_JAN, _NULL_ID_JAN, _CARLO_NULL_TS]
    return spark.createDataFrame(data, schema=CUSTOMER_SCHEMA)


@pytest.fixture()
def mixed_customers_df(spark: SparkSession) -> DataFrame:
    """DataFrame with active and inactive customers across multiple countries."""
    data = [_ALICE_JAN, _BOB_JAN, _CARLO_JAN, _DIANA_JAN, _ETHAN_JAN]
    return spark.createDataFrame(data, schema=CUSTOMER_SCHEMA)


@pytest.fixture()
def active_customers_df(spark: SparkSession) -> DataFrame:
    """DataFrame with only active customers with varied revenue tiers."""
    data = [_ALICE_JAN, _CARLO_JAN, _DIANA_JAN]
    return spark.createDataFrame(data, schema=CUSTOMER_SCHEMA)


@pytest.fixture()
def empty_customers_df(spark: SparkSession) -> DataFrame:
    """Empty DataFrame matching the customer schema."""
    return spark.createDataFrame([], schema=CUSTOMER_SCHEMA)


@pytest.fixture()
def duplicate_customers_df(spark: SparkSession) -> DataFrame:
    """DataFrame with two snapshots of customer C001 at different timestamps."""
    data = [_ALICE_JAN, _ALICE_FEB, _BOB_UK_JAN]
    return spark.createDataFrame(data, schema=CUSTOMER_SCHEMA)


# ---------------------------------------------------------------------------
# Fixtures – SCD2 state
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_scd2_df(spark: SparkSession) -> DataFrame:
    """Empty SCD2 DataFrame (first-run scenario)."""
    return spark.createDataFrame([], schema=SCD2_SCHEMA)


@pytest.fixture()
def existing_scd2_df(spark: SparkSession) -> DataFrame:
    """SCD2 table with one current and one historical row for customer C001."""
    data = [
        # C001 – historical row (revenue 75k, closed in Feb)
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
            TS_JAN,
            TS_JAN,
            TS_FEB,
            False,
            1,
        ),
        # C001 – current row (revenue 80k, opened in Feb)
        (
            "C001",
            "Alice",
            "Smith",
            "a@ex.com",
            "US",
            "NYC",
            "Retail",
            80_000.0,
            True,
            TS_FEB,
            TS_FEB,
            None,
            True,
            1,
        ),
        # C002 – only current row
        (
            "C002",
            "Bob",
            "Jones",
            "b@ex.com",
            "UK",
            "London",
            "SMB",
            20_000.0,
            True,
            TS_JAN,
            TS_JAN,
            None,
            True,
            1,
        ),
    ]
    return spark.createDataFrame(data, schema=SCD2_SCHEMA)


# ---------------------------------------------------------------------------
# filter_valid_customers
# ---------------------------------------------------------------------------


def test_filter_valid_customers_removes_null_id(
    customers_with_nulls_df: DataFrame,
) -> None:
    """Rows with null customer_id must be excluded."""
    result = filter_valid_customers(customers_with_nulls_df)
    assert result.filter(result["customer_id"].isNull()).count() == 0


def test_filter_valid_customers_removes_null_updated_at(
    customers_with_nulls_df: DataFrame,
) -> None:
    """Rows with null updated_at must be excluded."""
    result = filter_valid_customers(customers_with_nulls_df)
    assert result.filter(result["updated_at"].isNull()).count() == 0


def test_filter_valid_customers_retains_valid_rows(
    customers_with_nulls_df: DataFrame,
) -> None:
    """Valid rows (both key columns non-null) must be retained."""
    result = filter_valid_customers(customers_with_nulls_df)
    assert result.count() == 1


def test_filter_valid_customers_empty_input(empty_customers_df: DataFrame) -> None:
    """An empty input DataFrame must produce an empty output."""
    assert filter_valid_customers(empty_customers_df).count() == 0


# ---------------------------------------------------------------------------
# filter_active_customers
# ---------------------------------------------------------------------------


def test_filter_active_customers_excludes_inactive(
    mixed_customers_df: DataFrame,
) -> None:
    """No inactive row must appear in the output."""
    result = filter_active_customers(mixed_customers_df)
    assert result.filter(~result["is_active"]).count() == 0


def test_filter_active_customers_count(mixed_customers_df: DataFrame) -> None:
    """Three of the five mixed customers are active."""
    assert filter_active_customers(mixed_customers_df).count() == 3


def test_filter_active_customers_empty_input(empty_customers_df: DataFrame) -> None:
    """An empty input produces an empty output."""
    assert filter_active_customers(empty_customers_df).count() == 0


# ---------------------------------------------------------------------------
# enrich_with_full_name
# ---------------------------------------------------------------------------


def test_enrich_with_full_name_adds_column(active_customers_df: DataFrame) -> None:
    """The output DataFrame must contain a ``full_name`` column."""
    result = enrich_with_full_name(active_customers_df)
    assert "full_name" in result.columns


@pytest.mark.parametrize(
    ("first_name", "last_name", "expected"),
    [
        ("Alice", "Smith", "Alice Smith"),
        ("Carlo", "Rossi", "Carlo Rossi"),
        ("Diana", "Prince", "Diana Prince"),
    ],
)
def test_enrich_with_full_name_value(
    spark: SparkSession,
    first_name: str,
    last_name: str,
    expected: str,
) -> None:
    """``full_name`` must equal ``first_name`` + ' ' + ``last_name``."""
    row = _cust("C999", first_name, last_name, "US", "NYC", "Retail", 0.0, True, TS_JAN)
    df = spark.createDataFrame([row], schema=CUSTOMER_SCHEMA)
    result = enrich_with_full_name(df)
    assert result.collect()[0]["full_name"] == expected


# ---------------------------------------------------------------------------
# enrich_with_revenue_tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("revenue", "expected_tier"),
    [
        (150_000.0, "Platinum"),
        (75_000.0, "Gold"),
        (25_000.0, "Silver"),
        (5_000.0, "Bronze"),
        (100_000.0, "Platinum"),  # boundary – exactly 100 000
        (50_000.0, "Gold"),  # boundary – exactly 50 000
        (10_000.0, "Silver"),  # boundary – exactly 10 000
        (9_999.99, "Bronze"),  # just below Silver threshold
    ],
)
def test_enrich_with_revenue_tier_classification(
    spark: SparkSession,
    revenue: float,
    expected_tier: str,
) -> None:
    """Revenue tier must match the threshold rules for each boundary case."""
    row = _cust("C001", "A", "B", "US", "NYC", "Retail", revenue, True, TS_JAN)
    df = spark.createDataFrame([row], schema=CUSTOMER_SCHEMA)
    result = enrich_with_revenue_tier(df)
    assert result.collect()[0]["revenue_tier"] == expected_tier


def test_enrich_with_revenue_tier_adds_column(
    active_customers_df: DataFrame,
) -> None:
    """The output DataFrame must contain a ``revenue_tier`` column."""
    result = enrich_with_revenue_tier(active_customers_df)
    assert "revenue_tier" in result.columns


# ---------------------------------------------------------------------------
# deduplicate_by_latest
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_one_row_per_key(
    duplicate_customers_df: DataFrame,
) -> None:
    """Each customer_id must appear exactly once in the output."""
    result = deduplicate_by_latest(duplicate_customers_df, "customer_id", "updated_at")
    assert result.count() == 2


def test_deduplicate_keeps_latest_row(
    duplicate_customers_df: DataFrame,
) -> None:
    """The retained row for C001 must be the Feb snapshot (revenue_ytd=80 000)."""
    result = deduplicate_by_latest(duplicate_customers_df, "customer_id", "updated_at")
    c001 = result.filter(result["customer_id"] == "C001").collect()[0]
    assert c001["revenue_ytd"] == 80_000.0


def test_deduplicate_empty_input(empty_customers_df: DataFrame) -> None:
    """An empty input DataFrame must produce an empty output."""
    result = deduplicate_by_latest(empty_customers_df, "customer_id", "updated_at")
    assert result.count() == 0


# ---------------------------------------------------------------------------
# aggregate_revenue_by_country
# ---------------------------------------------------------------------------


def test_aggregate_revenue_by_country_columns(
    active_customers_df: DataFrame,
) -> None:
    """Output must contain country, customer_count, total_revenue, avg_revenue."""
    result = aggregate_revenue_by_country(active_customers_df)
    assert set(result.columns) == {
        "country",
        "customer_count",
        "total_revenue",
        "avg_revenue",
    }


@pytest.mark.parametrize(
    ("country", "expected_count", "expected_total"),
    [
        ("IT", 2, 158_000.0),
        ("US", 1, 75_000.0),
    ],
)
def test_aggregate_revenue_by_country_values(
    active_customers_df: DataFrame,
    country: str,
    expected_count: int,
    expected_total: float,
) -> None:
    """Country aggregation must produce correct counts and totals."""
    result = aggregate_revenue_by_country(active_customers_df)
    rows = {row["country"]: row for row in result.collect()}
    assert rows[country]["customer_count"] == expected_count
    assert rows[country]["total_revenue"] == expected_total


# ---------------------------------------------------------------------------
# aggregate_revenue_by_segment
# ---------------------------------------------------------------------------


def test_aggregate_revenue_by_segment_columns(
    active_customers_df: DataFrame,
) -> None:
    """Output must contain country, revenue_tier, customer_count, total_revenue."""
    enriched = enrich_with_revenue_tier(active_customers_df)
    result = aggregate_revenue_by_segment(enriched)
    assert set(result.columns) == {
        "country",
        "revenue_tier",
        "customer_count",
        "total_revenue",
    }


def test_aggregate_revenue_by_segment_count(
    active_customers_df: DataFrame,
) -> None:
    """Number of rows must equal the distinct (country, revenue_tier) pairs."""
    enriched = enrich_with_revenue_tier(active_customers_df)
    result = aggregate_revenue_by_segment(enriched)
    # IT-Platinum (Carlo 150k), IT-Bronze (Diana 8k), US-Gold (Alice 75k) → 3
    assert result.count() == 3


# ---------------------------------------------------------------------------
# apply_scd_type1
# ---------------------------------------------------------------------------


def test_scd1_one_row_per_customer(duplicate_customers_df: DataFrame) -> None:
    """SCD Type 1 must produce exactly one row per customer_id."""
    result = apply_scd_type1(duplicate_customers_df)
    assert result.count() == 2


def test_scd1_keeps_latest_values(duplicate_customers_df: DataFrame) -> None:
    """The retained C001 row must reflect the most recent revenue_ytd (80 000)."""
    result = apply_scd_type1(duplicate_customers_df)
    c001 = result.filter(result["customer_id"] == "C001").collect()[0]
    assert c001["revenue_ytd"] == 80_000.0


def test_scd1_empty_input(empty_customers_df: DataFrame) -> None:
    """SCD Type 1 on an empty input must return an empty DataFrame."""
    assert apply_scd_type1(empty_customers_df).count() == 0


# ---------------------------------------------------------------------------
# apply_scd_type2
# ---------------------------------------------------------------------------


def test_scd2_first_run_inserts_all_as_current(
    active_customers_df: DataFrame,
    empty_scd2_df: DataFrame,
) -> None:
    """On the first run (empty existing table) all rows must be current."""
    result = apply_scd_type2(empty_scd2_df, active_customers_df)
    total = result.count()
    current = result.filter(result["is_current"] == True).count()  # noqa: E712
    assert total == current == 3


def test_scd2_first_run_effective_to_is_null(
    active_customers_df: DataFrame,
    empty_scd2_df: DataFrame,
) -> None:
    """All rows inserted on the first run must have effective_to = NULL."""
    result = apply_scd_type2(empty_scd2_df, active_customers_df)
    assert result.filter(result["effective_to"].isNotNull()).count() == 0


def test_scd2_unchanged_row_not_duplicated(
    spark: SparkSession,
    existing_scd2_df: DataFrame,
) -> None:
    """An unchanged incoming row must not create a new SCD2 version."""
    # C001 changed (revenue 90k), C002 unchanged (still 20k)
    incoming = spark.createDataFrame([_ALICE_MAR, _BOB_UK_MAR], schema=CUSTOMER_SCHEMA)
    result = apply_scd_type2(existing_scd2_df, incoming)

    c002_current = result.filter(
        (result["customer_id"] == "C002") & (result["is_current"] == True)  # noqa: E712
    ).count()
    assert c002_current == 1


def test_scd2_changed_row_closes_old_version(
    spark: SparkSession,
    existing_scd2_df: DataFrame,
) -> None:
    """A changed incoming row must close the existing active row."""
    incoming = spark.createDataFrame([_ALICE_MAR_99], schema=CUSTOMER_SCHEMA)
    result = apply_scd_type2(existing_scd2_df, incoming)

    # The previously current C001 row (revenue=80 000) must now be closed.
    old_active_count = result.filter(
        (result["customer_id"] == "C001")
        & (result["revenue_ytd"] == 80_000.0)
        & (result["is_current"] == True)  # noqa: E712
    ).count()
    assert old_active_count == 0


def test_scd2_changed_row_inserts_new_version(
    spark: SparkSession,
    existing_scd2_df: DataFrame,
) -> None:
    """A changed incoming row must create a new current row with updated values."""
    incoming = spark.createDataFrame([_ALICE_MAR_99], schema=CUSTOMER_SCHEMA)
    result = apply_scd_type2(existing_scd2_df, incoming)

    new_row = result.filter(
        (result["customer_id"] == "C001")
        & (result["revenue_ytd"] == 99_000.0)
        & (result["is_current"] == True)  # noqa: E712
    )
    assert new_row.count() == 1


def test_scd2_history_is_preserved(
    spark: SparkSession,
    existing_scd2_df: DataFrame,
) -> None:
    """Existing historical rows must not be removed by a new merge."""
    incoming = spark.createDataFrame([_ALICE_MAR_99], schema=CUSTOMER_SCHEMA)
    result = apply_scd_type2(existing_scd2_df, incoming)

    # The oldest C001 row (revenue=75 000, is_current=False) must still exist.
    oldest = result.filter(
        (result["customer_id"] == "C001")
        & (result["revenue_ytd"] == 75_000.0)
        & (result["is_current"] == False)  # noqa: E712
    )
    assert oldest.count() == 1


def test_scd2_new_customer_inserted(
    spark: SparkSession,
    existing_scd2_df: DataFrame,
) -> None:
    """A brand-new customer not in the existing table must be inserted as current."""
    incoming = spark.createDataFrame([_ZOE_MAR], schema=CUSTOMER_SCHEMA)
    result = apply_scd_type2(existing_scd2_df, incoming)

    new_customer = result.filter(
        (result["customer_id"] == "C999") & (result["is_current"] == True)  # noqa: E712
    )
    assert new_customer.count() == 1


def test_scd2_only_one_current_row_per_customer(
    spark: SparkSession,
    existing_scd2_df: DataFrame,
) -> None:
    """After a merge there must be at most one current row per customer_id."""
    incoming = spark.createDataFrame(
        [_ALICE_MAR_99, _BOB_UK_MAR_21], schema=CUSTOMER_SCHEMA
    )
    result = apply_scd_type2(existing_scd2_df, incoming)

    current_df = result.filter(result["is_current"] == True)  # noqa: E712
    duplicates = (
        current_df.groupBy("customer_id")
        .agg(F.count("*").alias("cnt"))
        .filter(F.col("cnt") > 1)
    )
    assert duplicates.count() == 0


# ---------------------------------------------------------------------------
# SCD2_TRACKED_COLUMNS – constant contract
# ---------------------------------------------------------------------------


def test_scd2_tracked_columns_is_immutable() -> None:
    """``SCD2_TRACKED_COLUMNS`` must be a tuple so callers cannot mutate it."""
    assert isinstance(
        SCD2_TRACKED_COLUMNS, tuple
    ), "SCD2_TRACKED_COLUMNS must be a tuple to prevent accidental mutation"


def test_scd2_tracked_columns_contains_expected_fields() -> None:
    """``SCD2_TRACKED_COLUMNS`` must include all business-critical attributes."""
    required = {
        "first_name",
        "last_name",
        "email",
        "country",
        "revenue_ytd",
        "is_active",
    }
    assert required.issubset(set(SCD2_TRACKED_COLUMNS))


# ---------------------------------------------------------------------------
# enrich_with_full_name – null safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first_name", "last_name", "expected"),
    [
        (None, "Smith", "Smith"),
        ("Carlo", None, "Carlo"),
        (None, None, ""),
    ],
)
def test_enrich_with_full_name_null_safe(
    spark: SparkSession,
    first_name: str | None,
    last_name: str | None,
    expected: str,
) -> None:
    """A null name component must not propagate null into the full_name column.

    A null ``first_name`` or ``last_name`` is silently omitted.  When both are
    null the result is an empty string, never null.
    """
    row = _cust("C001", first_name, last_name, "US", "NYC", "Retail", 0.0, True, TS_JAN)
    df = spark.createDataFrame([row], schema=CUSTOMER_SCHEMA)
    result = enrich_with_full_name(df)
    assert result.collect()[0]["full_name"] == expected
