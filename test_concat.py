from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("TestConcatWS").getOrCreate()

data = [
    ("a", None, "b"),
    ("a", "b", None),
    ("a", "", "b"),
]

df = spark.createDataFrame(data, ["col1", "col2", "col3"])

df_hashed = df.withColumn(
    "concat", F.concat_ws("||", F.col("col1"), F.col("col2"), F.col("col3"))
).withColumn(
    "hash", F.sha2(F.concat_ws("||", F.col("col1"), F.col("col2"), F.col("col3")), 256)
)

df_hashed.show(truncate=False)
