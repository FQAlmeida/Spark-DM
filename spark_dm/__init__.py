from typing import List
from pyspark import RDD, SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.fpm import FPGrowth
import polars as pl

spark: SparkSession = SparkSession.builder.getOrCreate()  # type: ignore
spark_context: SparkContext = spark.sparkContext

conf = (
    SparkConf()
    .set("spark.driver.memory", "25g")
    .set("spark.executor.memory", "25g")
    .set("spark.driver.maxResultSize", "25g")
)
# spark_context = SparkContext(conf=conf)

df = pl.read_csv(
    "data/Assignment-1_Data.csv",
    separator=";",
    dtypes={"BillNo": pl.Utf8, "Itemname": pl.Utf8},
)

transactions = df.groupby("BillNo").agg(pl.col("Itemname").unique())

spark_transactions: RDD[List[str]] = spark_context.parallelize(
    transactions["Itemname"].to_list(), 600
)

model = FPGrowth.train(spark_transactions, 0.005, 4000)

for rule in model.freqItemsets():  # type: ignore
    print(rule)
