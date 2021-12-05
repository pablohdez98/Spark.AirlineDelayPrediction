import sys

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit


def split_data(df, trainRatio=0.8):
    return df.randomSplit([trainRatio, 1 - trainRatio], seed=12345)


def linear_regression_model(train):
    print("CREATING THE LINEAR REGRESSION MODEL (DEFAULT PARAMETERS)...")
    lr = LinearRegression()
    return lr.fit(train)


def decision_tree_model(train):
    print("CREATING THE DECISION TREE MODEL...")
    dt = DecisionTreeRegressor(featuresCol="features")
    return dt.fit(train)


def ml_tunning(train, tunning_type):
    print("CREATING THE LINEAR REGRESSION MODEL (" + tunning_type + ")...")
    lr = LinearRegression(maxIter=10)
    # Establising different parameters for the Linear Regression Model
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.3, 0.1, 0.01]) \
        .addGrid(lr.fitIntercept, [False, True]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    model = None
    if tunning_type == 'TRAIN VALIDATION SPLIT':
        model = TrainValidationSplit(estimator=lr,
                                     estimatorParamMaps=paramGrid,
                                     evaluator=RegressionEvaluator(),
                                     collectSubModels=True,
                                     trainRatio=0.99)
    elif tunning_type == 'CROSS-VALIDATION':
        model = CrossValidator(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               numFolds=5)

    return model.fit(train)


def evaluate_regression_model(model, test):
    print("EVALUATING THE MODEL...")
    if model is None:
        print('Error, model could not be created')
        sys.exit(1)
    predictions = model.transform(test)

    # Evaluate predictions with different metrics
    rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    mse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse").evaluate(predictions)

    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    print("R2 on test data = %g" % r2)
    print("Mean Absolute Error (MAE) on test data = %g" % mae)
    print("Mean Squared Error (MSE) on test data = %g" % mse)

    return predictions
