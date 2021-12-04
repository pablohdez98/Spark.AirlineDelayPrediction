from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


def prepare_df(df, label):
    columns = df.columns
    columns.remove(label)

    predictors = VectorAssembler(inputCols=columns, outputCol='features')
    df = predictors.transform(df)
    df = df.withColumnRenamed(label, 'label')
    df = df.drop(*columns)
    return df


def split_data(df, trainRatio=0.8):
    return df.randomSplit([trainRatio, 1 - trainRatio], seed=12345)


def linear_regression(train, test):
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='predictors', labelCol='target')
    lrModel = lr.fit(train)
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))


def linear_regression_model(train):
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='features', labelCol='label')
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
