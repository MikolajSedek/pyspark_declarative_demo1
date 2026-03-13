"""Pytest fixtures shared across all test modules.

The ``spark`` fixture provides a local SparkSession suitable for unit testing.
It is session-scoped so Spark starts only once per test run, keeping tests fast.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Create (or reuse) a local SparkSession for the test suite.

    The session is stopped automatically at the end of the test run.
    """
    session = (
        SparkSession.builder.master("local[1]")
        .appName("pyspark-declarative-demo-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
