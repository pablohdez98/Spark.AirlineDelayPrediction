from pyspark.sql import SparkSession
from preprocessing import *
from ml import *

spark = SparkSession.builder.appName('Airline Delay Prediction').getOrCreate()

df = load_data(spark)
df = prepare_df(df, 'ArrDelay')
train, test = split_data(df, 0.9)
linear_regression(train, test)
