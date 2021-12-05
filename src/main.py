from pyspark.sql import SparkSession
from preprocessing import *
from ml import *
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.appName('Airline Delay Prediction').getOrCreate()

df = load_data(spark)
train, test = split_data(df, 0.99)
'''lrModel = linear_regression_model(train)
evaluate_model(lrModel, test)'''
dtModel = decision_tree_model(train)
evalute_decisiontree_model(dtModel, test)
