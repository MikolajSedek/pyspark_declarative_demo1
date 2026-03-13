"""Pure transformation functions for the Parquet SCD pipeline.

These functions implement filtering, enrichment, aggregation, and Slowly
Changing Dimension (SCD) logic for customer data loaded from Parquet files.

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

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_valid_customers(df: DataFrame) -> DataFrame:
    """Remove records that have a null ``customer_id`` or ``updated_at``.

    Args:
        df: Input DataFrame with at least ``customer_id`` and ``updated_at``
            columns.

    Returns:
        DataFrame with only records where both key columns are non-null.
    """
    return df.filter(
        F.col("customer_id").isNotNull() & F.col("updated_at").isNotNull()
    )


def filter_active_customers(df: DataFrame) -> DataFrame:
    """Keep only records where ``is_active`` is ``True``.

    Args:
        df: Input DataFrame with at least an ``is_active`` boolean column.

    Returns:
        DataFrame containing only active customer rows.
    """
    return df.filter(F.col("is_active") == True)  # noqa: E712


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_with_full_name(df: DataFrame) -> DataFrame:
    """Add a ``full_name`` column by concatenating ``first_name`` and ``last_name``.

    Args:
        df: Input DataFrame with ``first_name`` and ``last_name`` string columns.

    Returns:
        DataFrame with an additional ``full_name`` column.
    """
    return df.withColumn(
        "full_name",
        F.concat(F.col("first_name"), F.lit(" "), F.col("last_name")),
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
        F.when(F.col("revenue_ytd") >= 100_000, F.lit("Platinum"))
        .when(F.col("revenue_ytd") >= 50_000, F.lit("Gold"))
        .when(F.col("revenue_ytd") >= 10_000, F.lit("Silver"))
        .otherwise(F.lit("Bronze")),
    )


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def deduplicate_by_latest(
    df: DataFrame,
    key_col: str,
    order_col: str,
) -> DataFrame:
    """Return one row per ``key_col`` value, keeping the row with the largest
    ``order_col`` value (typically a timestamp).

    Args:
        df: Input DataFrame.
        key_col: Column name used to partition rows (the natural key).
        order_col: Column name used to order rows within each partition;
            the row with the highest value is retained.

    Returns:
        Deduplicated DataFrame with one row per ``key_col``.
    """
    window = Window.partitionBy(F.col(key_col)).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


# ---------------------------------------------------------------------------
# Aggregation
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
        F.sum(F.col("revenue_ytd")).alias("total_revenue"),
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

# Columns whose changes trigger a new SCD2 history row.
_SCD2_TRACKED_COLUMNS: list[str] = [
    "first_name",
    "last_name",
    "email",
    "country",
    "city",
    "customer_segment",
    "revenue_ytd",
    "is_active",
]


def _build_change_hash(df: DataFrame, alias: str, tracked_cols: list[str]) -> "Column":  # noqa: F821
    """Return a SHA-256 hash Column of the concatenated ``tracked_cols`` values.

    The hash is computed over the concatenation of all tracked column values
    (cast to string and separated by ``"||"``).  This hash is used to detect
    whether a customer's attributes have changed between the existing SCD2
    table and the incoming snapshot.

    Args:
        df: DataFrame with columns named ``alias.<col>`` after a join.
        alias: The DataFrame alias used in the join (e.g. ``"new"`` or
            ``"old"``).
        tracked_cols: List of column names whose values contribute to the hash.

    Returns:
        Scalar Column expression representing the SHA-256 hash.
    """
    return F.sha2(
        F.concat_ws(
            "||",
            *[F.col(f"{alias}.{c}").cast("string") for c in tracked_cols],
        ),
        256,
    )


def _get_current_active(existing_scd2: DataFrame) -> DataFrame:
    """Return only the currently active rows from an existing SCD2 table.

    Args:
        existing_scd2: Full SCD2 history table.

    Returns:
        DataFrame containing only rows where ``is_current == True``.
    """
    return existing_scd2.filter(F.col("is_current") == True)  # noqa: E712


def _get_historical_rows(existing_scd2: DataFrame) -> DataFrame:
    """Return only the historical (non-current) rows from an SCD2 table.

    Args:
        existing_scd2: Full SCD2 history table.

    Returns:
        DataFrame containing only rows where ``is_current == False``.
    """
    return existing_scd2.filter(F.col("is_current") == False)  # noqa: E712


def _join_incoming_with_current(
    latest_incoming: DataFrame,
    current_active: DataFrame,
) -> DataFrame:
    """Left-join the deduplicated snapshot against the active SCD2 rows.

    Args:
        latest_incoming: One row per ``customer_id`` from the new snapshot.
        current_active: Currently active rows from the existing SCD2 table.

    Returns:
        Joined DataFrame aliased as ``"new"`` (incoming) and ``"old"`` (active).
    """
    return latest_incoming.alias("new").join(
        current_active.alias("old"),
        on=F.col("new.customer_id") == F.col("old.customer_id"),
        how="left",
    )


def _detect_changed_or_new(
    joined: DataFrame,
    tracked_cols: list[str],
) -> DataFrame:
    """Filter the join result to rows that are new or have changed attributes.

    A row is considered **changed** when the SHA-256 hash of ``tracked_cols``
    differs between the incoming and the currently active record.  A row is
    **new** when there is no matching record in the existing SCD2 table.

    Args:
        joined: Output of :func:`_join_incoming_with_current`.
        tracked_cols: Attribute columns to include in the change-detection hash.

    Returns:
        Subset of ``joined`` containing only new or changed customer rows.
    """
    new_hash = _build_change_hash(joined, "new", tracked_cols)
    old_hash = _build_change_hash(joined, "old", tracked_cols)
    return joined.filter(
        F.col("old.customer_id").isNull() | (new_hash != old_hash)
    )


def _build_closed_rows(
    changed_or_new: DataFrame,
    existing_scd2: DataFrame,
) -> DataFrame:
    """Produce closed-out rows for customers whose attributes have changed.

    Copies all SCD2 columns from the *old* (existing active) side of the join,
    overwriting ``effective_to`` with the incoming ``updated_at`` timestamp and
    setting ``is_current`` to ``False``.

    Args:
        changed_or_new: Rows detected as changed or new by
            :func:`_detect_changed_or_new`.
        existing_scd2: Full SCD2 history table (used to derive the passthrough
            column list).

    Returns:
        DataFrame of closed rows ready to union into the final SCD2 table.
    """
    passthrough = [
        c for c in existing_scd2.columns if c not in ("effective_to", "is_current")
    ]
    return (
        changed_or_new.filter(F.col("old.customer_id").isNotNull())
        .select(
            *[F.col(f"old.{c}").alias(c) for c in passthrough],
            F.col("new.updated_at").alias("effective_to"),
            F.lit(False).alias("is_current"),
        )
    )


def _build_inserted_rows(
    changed_or_new: DataFrame,
    latest_incoming: DataFrame,
) -> DataFrame:
    """Produce new SCD2 rows for brand-new and changed customers.

    Each inserted row gets ``effective_from = updated_at``,
    ``effective_to = NULL``, and ``is_current = True``.

    Args:
        changed_or_new: Rows detected as changed or new by
            :func:`_detect_changed_or_new`.
        latest_incoming: Deduplicated incoming snapshot (used to derive the
            column list for the ``"new"`` alias).

    Returns:
        DataFrame of newly opened SCD2 rows.
    """
    new_cols = [
        F.col(f"new.{c}").alias(c)
        for c in latest_incoming.columns
        if c != "customer_id"
    ]
    return changed_or_new.select(
        F.col("new.customer_id").alias("customer_id"),
        *new_cols,
        F.col("new.updated_at").alias("effective_from"),
        F.lit(None).cast("timestamp").alias("effective_to"),
        F.lit(True).alias("is_current"),
    )


def _get_unchanged_current_rows(
    current_active: DataFrame,
    changed_or_new: DataFrame,
) -> DataFrame:
    """Return active rows for customers whose attributes have not changed.

    Uses a ``left_anti`` join to exclude customers that appear in the
    changed-or-new set.

    Args:
        current_active: Currently active SCD2 rows.
        changed_or_new: Rows detected as changed or new by
            :func:`_detect_changed_or_new`.

    Returns:
        DataFrame of unchanged active SCD2 rows to carry forward.
    """
    changed_ids = changed_or_new.select(
        F.col("new.customer_id").alias("customer_id")
    )
    return current_active.join(changed_ids, on="customer_id", how="left_anti")


def apply_scd_type2(
    existing_scd2: DataFrame,
    incoming: DataFrame,
    tracked_cols: list[str] | None = None,
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
            to ``_SCD2_TRACKED_COLUMNS``.

    Returns:
        New SCD2 DataFrame with all historical and current rows.
    """
    if tracked_cols is None:
        tracked_cols = _SCD2_TRACKED_COLUMNS

    latest_incoming = deduplicate_by_latest(
        incoming, key_col="customer_id", order_col="updated_at"
    )
    current_active = _get_current_active(existing_scd2)
    joined = _join_incoming_with_current(latest_incoming, current_active)
    changed_or_new = _detect_changed_or_new(joined, tracked_cols)
    unchanged_current_rows = _get_unchanged_current_rows(current_active, changed_or_new)

    historical_rows = _get_historical_rows(existing_scd2)
    closed_rows = _build_closed_rows(changed_or_new, existing_scd2)
    inserted_rows = _build_inserted_rows(changed_or_new, latest_incoming)

    return (
        historical_rows
        .unionByName(unchanged_current_rows)
        .unionByName(closed_rows)
        .unionByName(inserted_rows)
    )
