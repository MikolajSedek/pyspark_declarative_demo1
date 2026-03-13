"""Pure transformation functions for the PySpark Declarative Pipelines demo.

This module implements all filtering, enrichment, aggregation, and Slowly
Changing Dimension (SCD) logic for both pipelines:

* **User CSV pipeline** – ``filter_active_users``, ``enrich_with_full_name``,
  ``aggregate_user_count_by_country``.
* **Customer Parquet + SCD pipeline** – ``filter_valid_customers``,
  ``filter_active_customers``, ``enrich_with_revenue_tier``,
  ``aggregate_revenue_by_country``, ``aggregate_revenue_by_segment``,
  ``apply_scd_type1``, ``apply_scd_type2``.

Each function is a pure transformation: it takes one or more DataFrames and
returns a new DataFrame without side effects, making them independently
testable.

SCD Type 1 – Overwrite
    Keep only the latest version of each customer; no history is preserved.

SCD Type 2 – Track history
    Maintain a full history of changes per customer with ``effective_from``,
    ``effective_to``, and ``is_current`` columns.

Reference:
    Palantir PySpark Style Guide:
    https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide
"""

from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from src.python.utils import (
    SCD2_TRACKED_COLUMNS,
    build_closed_rows,
    build_inserted_rows,
    deduplicate_by_latest,
    detect_changed_or_new,
    get_current_active,
    get_historical_rows,
    get_unchanged_current_rows,
    join_incoming_with_current,
)

# ---------------------------------------------------------------------------
# Revenue tier thresholds – used in enrich_with_revenue_tier
# ---------------------------------------------------------------------------

PLATINUM_THRESHOLD: int = 100_000
GOLD_THRESHOLD: int = 50_000
SILVER_THRESHOLD: int = 10_000

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_active(df: DataFrame) -> DataFrame:
    """Return only rows where ``is_active`` is ``True``."""
    return df.filter(F.col("is_active"))

# ---------------------------------------------------------------------------
# Filtering – User CSV pipeline
# ---------------------------------------------------------------------------


def filter_active_users(df: DataFrame) -> DataFrame:
    """Keep only rows where the ``is_active`` flag is ``True``.

    Args:
        df: Input DataFrame with at least an ``is_active`` boolean column.

    Returns:
        DataFrame containing only active user rows.
    """
    return _filter_active(df)


# ---------------------------------------------------------------------------
# Filtering – Customer Parquet + SCD pipeline
# ---------------------------------------------------------------------------


def filter_valid_customers(df: DataFrame) -> DataFrame:
    """Remove records that have a null ``customer_id`` or ``updated_at``.

    Args:
        df: Input DataFrame with at least ``customer_id`` and ``updated_at``
            columns.

    Returns:
        DataFrame with only records where both key columns are non-null.
    """
    return df.filter(F.col("customer_id").isNotNull() & F.col("updated_at").isNotNull())


def filter_active_customers(df: DataFrame) -> DataFrame:
    """Keep only records where ``is_active`` is ``True``.

    Args:
        df: Input DataFrame with at least an ``is_active`` boolean column.

    Returns:
        DataFrame containing only active customer rows.
    """
    return _filter_active(df)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_with_full_name(df: DataFrame) -> DataFrame:
    """Add a ``full_name`` column by concatenating ``first_name`` and ``last_name``.

    Uses ``concat_ws`` so that a null in either name component does not
    propagate null to the output – the non-null part is preserved and both-null
    inputs yield an empty string rather than null.

    Args:
        df: Input DataFrame with ``first_name`` and ``last_name`` string columns.

    Returns:
        DataFrame with an additional ``full_name`` column.
    """
    return df.withColumn(
        "full_name",
        F.concat_ws(" ", F.col("first_name"), F.col("last_name")),
    )


def enrich_with_revenue_tier(df: DataFrame) -> DataFrame:
    """Classify each customer into a ``revenue_tier`` based on ``revenue_ytd``.

    Tier thresholds:

    * **Platinum** – revenue_ytd ≥ 100 000
    * **Gold**     – revenue_ytd ≥  50 000
    * **Silver**   – revenue_ytd ≥  10 000
    * **Bronze**   – revenue_ytd <  10 000

    Args:
        df: Input DataFrame with a numeric ``revenue_ytd`` column.

    Returns:
        DataFrame with an additional ``revenue_tier`` string column.
    """
    return df.withColumn(
        "revenue_tier",
        F.when(F.col("revenue_ytd") >= PLATINUM_THRESHOLD, F.lit("Platinum"))
        .when(F.col("revenue_ytd") >= GOLD_THRESHOLD, F.lit("Gold"))
        .when(F.col("revenue_ytd") >= SILVER_THRESHOLD, F.lit("Silver"))
        .otherwise(F.lit("Bronze")),
    )


# ---------------------------------------------------------------------------
# Aggregation – User CSV pipeline
# ---------------------------------------------------------------------------


def aggregate_user_count_by_country(df: DataFrame) -> DataFrame:
    """Count users per country.

    Args:
        df: Input DataFrame with a ``country`` string column.

    Returns:
        DataFrame with ``country`` and ``user_count`` columns.
    """
    return df.groupBy("country").agg(F.count("*").alias("user_count"))


# ---------------------------------------------------------------------------
# Aggregation – Customer Parquet + SCD pipeline
# ---------------------------------------------------------------------------


def aggregate_revenue_by_country(df: DataFrame) -> DataFrame:
    """Compute customer count and revenue metrics grouped by ``country``.

    Args:
        df: Input DataFrame with ``country`` (string) and ``revenue_ytd``
            (numeric) columns.

    Returns:
        DataFrame with columns: ``country``, ``customer_count``,
        ``total_revenue``, ``avg_revenue``.
    """
    return df.groupBy(F.col("country")).agg(
        F.count("*").alias("customer_count"),
        F.round(F.sum(F.col("revenue_ytd")), 2).alias("total_revenue"),
        F.round(F.avg(F.col("revenue_ytd")), 2).alias("avg_revenue"),
    )


def aggregate_revenue_by_segment(df: DataFrame) -> DataFrame:
    """Compute customer count and total revenue grouped by country and segment.

    Args:
        df: Input DataFrame with ``country``, ``revenue_tier``, and
            ``revenue_ytd`` columns.

    Returns:
        DataFrame with columns: ``country``, ``revenue_tier``,
        ``customer_count``, ``total_revenue``.
    """
    return df.groupBy(F.col("country"), F.col("revenue_tier")).agg(
        F.count("*").alias("customer_count"),
        F.sum(F.col("revenue_ytd")).alias("total_revenue"),
    )


# ---------------------------------------------------------------------------
# SCD Type 1 – overwrite / no history
# ---------------------------------------------------------------------------


def apply_scd_type1(df: DataFrame) -> DataFrame:
    """Apply SCD Type 1: return the latest record per ``customer_id``.

    Older records for the same customer are discarded; no history is kept.
    The newest record is determined by the ``updated_at`` timestamp.

    Args:
        df: Input DataFrame containing customer records with ``customer_id``
            and ``updated_at`` columns.

    Returns:
        DataFrame with exactly one row per ``customer_id``, representing the
        most recent state.
    """
    return deduplicate_by_latest(df, key_col="customer_id", order_col="updated_at")


# ---------------------------------------------------------------------------
# SCD Type 2 – track history with effective dates
# ---------------------------------------------------------------------------


def apply_scd_type2(
    existing_scd2: DataFrame,
    incoming: DataFrame,
    tracked_cols: Sequence[str] | None = None,
) -> DataFrame:
    """Apply SCD Type 2 merge logic to produce an updated history table.

    **Algorithm**

    1. Deduplicate ``incoming`` by ``customer_id``, keeping the latest row.
    2. Left-join the snapshot against the currently active SCD2 rows.
    3. Detect **new** and **changed** customers via attribute hashing.
    4. **Close** existing active rows for changed customers.
    5. **Insert** new rows for new and changed customers.
    6. **Retain** all historical (non-current) rows unchanged.
    7. **Carry forward** unchanged current rows.
    8. Union all result sets into the final SCD2 table.

    Args:
        existing_scd2: The current SCD2 dimension table.  May be empty on the
            first run.  Must contain ``customer_id``, ``effective_from``,
            ``effective_to``, ``is_current``, and all columns in
            ``tracked_cols``.
        incoming: Snapshot of source customer records.  Must contain
            ``customer_id``, ``updated_at``, and all columns in
            ``tracked_cols``.
        tracked_cols: Columns whose changes trigger a new SCD2 row.  Defaults
            to :data:`~src.python.utils.SCD2_TRACKED_COLUMNS`.

    Returns:
        New SCD2 DataFrame with all historical and current rows.
    """
    if tracked_cols is None:
        tracked_cols = SCD2_TRACKED_COLUMNS

    # Migration shim: backfill hash_version for tables persisted before this
    # column was introduced.  Without this, unionByName raises AnalysisException
    # when inserting rows (which always have hash_version) into an older table.
    if "hash_version" not in existing_scd2.columns:
        existing_scd2 = existing_scd2.withColumn(
            "hash_version", F.lit(None).cast(IntegerType())
        )

    latest_incoming = deduplicate_by_latest(
        incoming, key_col="customer_id", order_col="updated_at"
    )
    current_active = get_current_active(existing_scd2)
    joined = join_incoming_with_current(latest_incoming, current_active)
    changed_or_new = detect_changed_or_new(joined, tracked_cols)
    unchanged_current_rows = get_unchanged_current_rows(current_active, changed_or_new)

    historical_rows = get_historical_rows(existing_scd2)
    closed_rows = build_closed_rows(changed_or_new, existing_scd2)
    inserted_rows = build_inserted_rows(changed_or_new, latest_incoming)

    return (
        historical_rows.unionByName(unchanged_current_rows)
        .unionByName(closed_rows)
        .unionByName(inserted_rows)
    )
