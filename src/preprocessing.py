from pyspark.ml.feature import OneHotEncoder


def load_data(spark):
    df = spark.read.csv("../data/2008.csv", header=True, inferSchema=True)
    # Remove forbidden variables
    df = df.drop(*('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay',
                   'NASDelay', 'SecurityDelay', 'LateAircraftDelay'))
    # Remove useless variables
    df = df.drop('CancellationCode')

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

    encoder = OneHotEncoder(inputCols=["UniqueCarrier", "TailNum", "Origin", "Dest"],
                            outputCols=["UniqueCarrier", "TailNum", "Origin", "Dest"])
    model = encoder.fit(df)
    df = model.transform(df)

    return df
