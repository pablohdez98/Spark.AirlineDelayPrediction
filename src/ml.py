from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor


def split_data(df, trainRatio=0.8):
    return df.randomSplit([trainRatio, 1 - trainRatio], seed=12345)


def linear_regression_model(train):
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    return lr.fit(train)


def decision_tree_model(train):
    dt = DecisionTreeRegressor(featuresCol="features")
    return dt.fit(train)


def evaluate_regression_model(model, test):
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
