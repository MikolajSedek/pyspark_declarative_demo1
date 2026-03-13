"""Utility and helper functions for the PySpark Declarative Pipelines demo.

This module provides generic, reusable helpers – deduplication logic and the
internal SCD Type 2 building-blocks – that are shared across pipelines.

All functions are pure transformations: they take DataFrames (and optional
parameters) and return new DataFrames without side effects, making them
independently testable.

Reference:
    Palantir PySpark Style Guide:
    https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide
"""

from collections.abc import Sequence

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Deduplication
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
# SCD Type 2 helpers
# ---------------------------------------------------------------------------

# Columns whose changes trigger a new SCD2 history row.
# Declared as a tuple so callers cannot accidentally mutate the default.
SCD2_TRACKED_COLUMNS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "email",
    "country",
    "city",
    "customer_segment",
    "revenue_ytd",
    "is_active",
)


def build_change_hash(alias: str, tracked_cols: Sequence[str]) -> Column:
    """Return a SHA-256 hash Column of the concatenated ``tracked_cols`` values.

    The hash is computed over the concatenation of all tracked column values
    (cast to string and separated by ``"||"``).  This hash is used to detect
    whether a customer's attributes have changed between the existing SCD2
    table and the incoming snapshot.

    Args:
        alias: The DataFrame alias used in the join (e.g. ``"new"`` or
            ``"old"``).
        tracked_cols: Sequence of column names whose values contribute to the
            hash.

    Returns:
        Column expression representing the SHA-256 hash.
    """
    return F.sha2(
        F.concat_ws(
            "||",
            *[F.col(f"{alias}.{c}").cast("string") for c in tracked_cols],
        ),
        256,
    )


def get_current_active(existing_scd2: DataFrame) -> DataFrame:
    """Return only the currently active rows from an existing SCD2 table.

    Args:
        existing_scd2: Full SCD2 history table.

    Returns:
        DataFrame containing only rows where ``is_current == True``.
    """
    return existing_scd2.filter(F.col("is_current"))


def get_historical_rows(existing_scd2: DataFrame) -> DataFrame:
    """Return only the historical (non-current) rows from an SCD2 table.

    Args:
        existing_scd2: Full SCD2 history table.

    Returns:
        DataFrame containing only rows where ``is_current == False``.
    """
    return existing_scd2.filter(~F.col("is_current"))


def join_incoming_with_current(
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


def detect_changed_or_new(
    joined: DataFrame,
    tracked_cols: Sequence[str],
) -> DataFrame:
    """Filter the join result to rows that are new or have changed attributes.

    A row is considered **changed** when the SHA-256 hash of ``tracked_cols``
    differs between the incoming and the currently active record.  A row is
    **new** when there is no matching record in the existing SCD2 table.

    Args:
        joined: Output of :func:`join_incoming_with_current`.
        tracked_cols: Attribute columns to include in the change-detection hash.

    Returns:
        Subset of ``joined`` containing only new or changed customer rows.
    """
    new_hash = build_change_hash("new", tracked_cols)
    old_hash = build_change_hash("old", tracked_cols)
    return joined.filter(F.col("old.customer_id").isNull() | (new_hash != old_hash))


def build_closed_rows(
    changed_or_new: DataFrame,
    existing_scd2: DataFrame,
) -> DataFrame:
    """Produce closed-out rows for customers whose attributes have changed.

    Copies all SCD2 columns from the *old* (existing active) side of the join,
    overwriting ``effective_to`` with the incoming ``updated_at`` timestamp and
    setting ``is_current`` to ``False``.

    Args:
        changed_or_new: Rows detected as changed or new by
            :func:`detect_changed_or_new`.
        existing_scd2: Full SCD2 history table (used to derive the passthrough
            column list).

    Returns:
        DataFrame of closed rows ready to union into the final SCD2 table.
    """
    passthrough = [
        c for c in existing_scd2.columns if c not in ("effective_to", "is_current")
    ]
    return changed_or_new.filter(F.col("old.customer_id").isNotNull()).select(
        *[F.col(f"old.{c}").alias(c) for c in passthrough],
        F.col("new.updated_at").alias("effective_to"),
        F.lit(False).alias("is_current"),
    )


def build_inserted_rows(
    changed_or_new: DataFrame,
    latest_incoming: DataFrame,
) -> DataFrame:
    """Produce new SCD2 rows for brand-new and changed customers.

    Each inserted row gets ``effective_from = updated_at``,
    ``effective_to = NULL``, and ``is_current = True``.

    Args:
        changed_or_new: Rows detected as changed or new by
            :func:`detect_changed_or_new`.
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


def get_unchanged_current_rows(
    current_active: DataFrame,
    changed_or_new: DataFrame,
) -> DataFrame:
    """Return active rows for customers whose attributes have not changed.

    Uses a ``left_anti`` join to exclude customers that appear in the
    changed-or-new set.

    Args:
        current_active: Currently active SCD2 rows.
        changed_or_new: Rows detected as changed or new by
            :func:`detect_changed_or_new`.

    Returns:
        DataFrame of unchanged active SCD2 rows to carry forward.
    """
    changed_ids = changed_or_new.select(F.col("new.customer_id").alias("customer_id"))
    return current_active.join(changed_ids, on="customer_id", how="left_anti")
