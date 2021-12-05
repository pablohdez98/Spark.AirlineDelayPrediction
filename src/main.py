from pyspark.sql import SparkSession
from preprocessing import *
from ml import *

spark = SparkSession.builder.appName('Airline Delay Prediction').getOrCreate()

df = load_data(spark)
train, test = split_data(df, 0.99)

lrModel = linear_regression_model(train)
predictions = evaluate_regression_model(lrModel, test)
predictions.sample(False, 0.1, seed=0).show(5)

dtModel = decision_tree_model(train)
predictions = evaluate_regression_model(dtModel, test)
predictions.sample(False, 0.1, seed=0).show(5)
