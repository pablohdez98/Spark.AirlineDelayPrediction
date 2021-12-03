from pyspark.sql import SparkSession
from preprocessing import *

spark = SparkSession.builder.appName('Practise').getOrCreate()

df = load_data(spark)
