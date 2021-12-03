from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer


def load_data(spark):
    df = spark.read.csv('../data/2008.csv', header=True, inferSchema=True)
    # Remove forbidden variables
    df = df.drop(*('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay',
                   'NASDelay', 'SecurityDelay', 'LateAircraftDelay'))
    # Remove useless variables
    df = df.drop(*('CancellationCode', 'TailNum'))

    print(df.count())
    # Remove canceled flights
    df = df.replace('NA', None)
    df = df.na.drop(how='any', subset=['ArrDelay'])
    print(df.count())

    # Remove rows with NA values
    df = df.na.drop(how='any')
    print(df.count())

    # Parse columns from string to integer or double
    df = df.withColumn('DepTime', df['DepTime'].cast('integer'))
    df = df.withColumn('CRSElapsedTime', df['CRSElapsedTime'].cast('integer'))
    df = df.withColumn('ArrDelay', df['ArrDelay'].cast('double'))
    df = df.withColumn('DepDelay', df['DepDelay'].cast('integer'))
    df = df.withColumn('TaxiOut', df['TaxiOut'].cast('integer'))

    # OneHotEncoder
    indexer = StringIndexer(inputCols=['UniqueCarrier', 'Origin', 'Dest'],
                            outputCols=['UniqueCarrier_index', 'Origin_index', 'Dest_index']).fit(df)
    df = indexer.transform(df)
    indexer = OneHotEncoder(inputCols=['UniqueCarrier_index', 'Origin_index', 'Dest_index'],
                            outputCols=['UniqueCarrier_ohe', 'Origin_index_ohe', 'Dest_index_ohe']).fit(df)
    df = indexer.transform(df)
    df = df.drop(*('Dest', 'Origin', 'UniqueCarrier', 'Dest_index', 'Origin_index', 'UniqueCarrier_index'))
    df = df.withColumnRenamed('UniqueCarrier_ohe', 'UniqueCarrier') \
        .withColumnRenamed('Origin_index_ohe', 'Origin') \
        .withColumnRenamed('Dest_index_ohe', 'Dest')
    df.show()

    return df