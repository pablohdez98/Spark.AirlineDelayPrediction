from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.functions import mean as _mean
from pyspark.sql.functions import isnan, when, count, col


def load_data(spark):
    df = spark.read.csv('../data/2008.csv', header=True, inferSchema=True)
    # Remove forbidden variables
    df = df.drop(*('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay',
                   'NASDelay', 'SecurityDelay', 'LateAircraftDelay'))
    # Remove useless variables
    df = df.drop(*('Year', 'CancellationCode', 'DepTime', 'FlightNum', 'TailNum', 'Distance', 'Cancelled'))

    # Remove canceled flights
    df = df.replace('NA', None)
    df = df.na.drop(how='any', subset=['ArrDelay'])

    # Check if columns have more than 50% empty values
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    df_nan = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    print(df.columns)
    for c in df_nan.columns:
        if (df.count() * 0.5) < df_nan.first()[c]:
            df = df.drop(c)

    print(df.columns)
    # Remove rows with NA values
    df = df.na.drop(how='any')

    # Parse columns from string to integer or double
    df = df.withColumn('CRSElapsedTime', df['CRSElapsedTime'].cast('integer'))
    df = df.withColumn('ArrDelay', df['ArrDelay'].cast('integer'))
    df = df.withColumn('DepDelay', df['DepDelay'].cast('integer'))
    df = df.withColumn('TaxiOut', df['TaxiOut'].cast('integer')) if 'TaxiOut' in df.columns else df

    # Mean Taxi Out
    if 'TaxiOut' in df.columns:
        df_TaxiOutMean = df.groupby('Origin').agg(_mean('TaxiOut'))
        df_TaxiOutMean = df_TaxiOutMean.withColumnRenamed('Origin', 'Origin2')
        df = df.join(df_TaxiOutMean, df['Origin'] == df_TaxiOutMean['Origin2'], 'left')
        df = df.withColumn('TaxiOut', df['TaxiOut'] - df['avg(TaxiOut)'])
        df = df.drop(*('avg(TaxiOut)', 'Origin2'))

    # OneHotEncoder
    indexer = StringIndexer(inputCols=['UniqueCarrier', 'Origin', 'Dest'],
                            outputCols=['UniqueCarrier_index', 'Origin_index', 'Dest_index']).fit(df)
    df = indexer.transform(df)
    indexer = OneHotEncoder(inputCols=['UniqueCarrier_index', 'Origin_index', 'Dest_index', 'Month', 'DayOfWeek'],
                            outputCols=['UniqueCarrier_ohe', 'Origin_index_ohe', 'Dest_index_ohe', 'Month_index_ohe',
                                        'DayOfWeek_index_ohe']).fit(df)
    df = indexer.transform(df)
    df = df.drop(
        *('Dest', 'Origin', 'UniqueCarrier', 'Dest_index', 'Origin_index', 'UniqueCarrier_index', 'Month', 'DayOfWeek'))
    df = df.withColumnRenamed('UniqueCarrier_ohe', 'UniqueCarrier') \
        .withColumnRenamed('Origin_index_ohe', 'Origin') \
        .withColumnRenamed('Dest_index_ohe', 'Dest') \
        .withColumnRenamed('Month_index_ohe', 'Month') \
        .withColumnRenamed('DayOfWeek_index_ohe', 'DayOfWeek')

    return df
