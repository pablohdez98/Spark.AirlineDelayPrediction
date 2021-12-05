from pyspark.sql import SparkSession
from preprocessing import *
from ml import *

file_year = ml_algorithm = tunning_type = -1

# User inputs
while file_year not in range(1987, 2009, 1):
    try:
        file_year = int(input("Enter year of file (1987-2008): "))
    except:
        print('Not a valid year')

while ml_algorithm not in range(1, 3, 1):
    try:
        ml_algorithm = int(input("Enter the number of the algorithm desired:"
                                 "\n(1) Linear Regression"
                                 "\n(2) Decision Tree"
                                 "\n-> "))
    except:
        print('Not a valid algorithm')

if ml_algorithm == 1:
    while tunning_type not in range(1, 4, 1):
        try:
            tunning_type = int(input("Enter the number of how you want to run it:"
                                     "\n(1) Default parameters"
                                     "\n(2) Tunning by splitting the train into train and dev"
                                     "\n(3) Tunning by applying cross-validation with k=5"
                                     "\n-> "))
        except:
            print('Not a valid number')

# Create Spark Session
spark = SparkSession.builder.appName('Airline Delay Prediction').getOrCreate()

df = load_data(spark, file_year)
train, test = split_data(df, 0.99)  # Split dataset in 2: train and test

model = None
if ml_algorithm == 1:  # Linear Regression Model
    if tunning_type == 1:  # No tunning
        model = linear_regression_model(train)
    elif tunning_type == 2:  # Train Validation Split
        model = ml_tunning(train, 'TRAIN VALIDATION SPLIT')
    elif tunning_type == 3:  # Cross-validation
        model = ml_tunning(train, 'CROSS-VALIDATION')
elif ml_algorithm == 2:  # Decision Tree Model
    model = decision_tree_model(train)

if model is not None:
    predictions = evaluate_regression_model(model, test)
    predictions.sample(False, 0.1, seed=0).show(5)  # Display 5 random predictions
