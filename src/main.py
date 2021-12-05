from pyspark.sql import SparkSession
from user_input import questions
from preprocessing import *
from ml import *

# Ask user for some configuration
file_year, ml_algorithm, tunning_type, analize = questions()

# Create Spark Session
spark = SparkSession.builder.appName('Airline Delay Prediction').getOrCreate()

df = load_csv(spark, file_year)
if analize:
    analysis(df)
df = process_data(df)

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
