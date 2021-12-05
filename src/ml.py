from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor


def split_data(df, trainRatio=0.8):
    return df.randomSplit([trainRatio, 1 - trainRatio], seed=12345)


def linear_regression_model(train):
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    model = lr.fit(train)
    print("Coefficients: %s" % str(model.coefficients))
    print("Intercept: %s" % str(model.intercept))

    trainingSummary = model.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    return model


def evaluate_model(model, test):
    pred_results = model.evaluate(test)
    pred_results.predictions.sort('label', ascending=False).show()
    print("MAE: %f" % pred_results.meanAbsoluteError)
    print("MSE: %f" % pred_results.meanSquaredError)


def decision_tree_model(trainingData):

    # Create a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="features")
    # Train model.
    model = dt.fit(trainingData)
    print(model)
    return model

def evalute_decisiontree_model(model, testData) :
    # Make predictions.
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
