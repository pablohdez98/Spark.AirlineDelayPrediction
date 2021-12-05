from pyspark.sql import SparkSession
from preprocessing import *
from ml import *

file = -1
ml = -1

while file not in range(1987, 2009, 1):
    try:
        file = int(input("Enter year of file (1987-2008): "))
    except:
        print('Not a valid year')

while ml not in range(1, 3, 1):
    try:
        ml = int(input("Enter the number of the algorithm desired:"
                       "\n(1) Linear Regression"
                       "\n(2) Decision Tree"
                       "\n-> "))
    except:
        print('Not a valid algorithm')

spark = SparkSession.builder.appName('Airline Delay Prediction').getOrCreate()

df = load_data(spark, file)
train, test = split_data(df, 0.99)

if ml == 1:
    lrModel = linear_regression_model(train)
    predictions = evaluate_regression_model(lrModel, test)
    predictions.sample(False, 0.1, seed=0).show(5)

elif ml == 2:
    dtModel = decision_tree_model(train)
    predictions = evaluate_regression_model(dtModel, test)
    predictions.sample(False, 0.1, seed=0).show(5)
