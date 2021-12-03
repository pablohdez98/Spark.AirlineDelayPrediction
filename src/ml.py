from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


def prepare_df(df, target):
    columns = df.columns
    columns.remove(target)

    predictors = VectorAssembler(inputCols=columns, outputCol='predictors')
    df = predictors.transform(df)
    df = df.withColumnRenamed(target, 'target')
    df = df.drop(*columns)
    return df


def split_data(df, trainRatio=0.8):
    return df.randomSplit([trainRatio, 1 - trainRatio], seed=12345)


def linear_regression(train, test):
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='predictors', labelCol='target')
    lrModel = lr.fit(train)
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))

    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    pred_results = lrModel.evaluate(test)
    pred_results.predictions.sort('target', ascending=False).show()
    print(pred_results.meanAbsoluteError)
    print(pred_results.meanSquaredError)
