import pytest
from pyspark.sql import SparkSession

from pyspark_job import clean_data


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-clean-data")
        .getOrCreate()
    )

    yield spark

    spark.stop()


def test_valid_records_are_kept(spark):
    data = [
        ("Alice", 100.0),
        ("Bob", 200.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    assert result.count() == 2


def test_records_with_non_positive_amount_are_removed(spark):
    data = [
        ("Alice", 100.0),
        ("Bob", 0.0),
        ("Charlie", -50.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    names = [row["name"] for row in result.collect()]

    assert names == ["Alice"]


def test_records_with_null_names_are_removed(spark):
    data = [
        ("Alice", 100.0),
        (None, 200.0),
        ("Charlie", 300.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    names = [row["name"] for row in result.collect()]

    assert names == ["Alice", "Charlie"]


def test_amount_with_tax_is_calculated_correctly(spark):
    data = [
        ("Alice", 100.0),
        ("Bob", 50.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    rows = result.collect()

    assert rows[0]["amount_with_tax"] == pytest.approx(120.0)
    assert rows[1]["amount_with_tax"] == pytest.approx(60.0)