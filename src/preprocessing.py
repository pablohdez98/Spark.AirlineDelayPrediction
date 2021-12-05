import sys

from pyspark.ml.feature import RFormula
from pyspark.sql.functions import mean as _mean
from pyspark.sql.functions import isnan, when, count, col


def load_data(spark, file):
    # Check if file exists
    try:
        df = spark.read.csv('../data/'+str(file)+'.csv', header=True, inferSchema=True)
    except:
        print("Exception, file does not exist")
        sys.exit(1)

    # Check if file is empty
    if df.count() == 0:
        print('Exception, file is empty')
        sys.exit(1)

    # Remove forbidden variables
    df = df.drop(*('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay',
                   'NASDelay', 'SecurityDelay', 'LateAircraftDelay'))
    # Remove useless variables
    df = df.drop(*('Year', 'CancellationCode', 'DepTime', 'FlightNum', 'TailNum', 'Distance', 'Cancelled'))

    # Check if the columns remaining are in the file
    c1 = df.schema.simpleString().find('Month')
    c2 = df.schema.simpleString().find('DayofMonth')
    c3 = df.schema.simpleString().find('DayOfWeek')
    c4 = df.schema.simpleString().find('CRSDepTime')
    c5 = df.schema.simpleString().find('CRSArrTime')
    c6 = df.schema.simpleString().find('CRSElapsedTime')
    c7 = df.schema.simpleString().find('DepDelay')
    c8 = df.schema.simpleString().find('TaxiOut')
    c9 = df.schema.simpleString().find('UniqueCarrier')
    c10 = df.schema.simpleString().find('Origin')
    c11 = df.schema.simpleString().find('Dest')
    c12 = df.schema.simpleString().find('ArrDelay')

    if c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 == -1:
        print("Exception, there are at least one important column missing")
        sys.exit(1)

    # Remove canceled flights
    df = df.replace('NA', None)
    df = df.na.drop(how='any', subset=['ArrDelay'])

    # Check if columns have more than 50% empty values
    df_nan = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    for c in df_nan.columns:
        if (df.count() * 0.5) < df_nan.first()[c]:
            df = df.drop(c)

    # Remove rows with NA values
    df = df.na.drop(how='any')

    # Parse all columns to their type
    df = df.withColumn('Month', df['Month'].cast('string'))
    df = df.withColumn('DayofMonth', df['DayofMonth'].cast('string'))
    df = df.withColumn('DayOfWeek', df['DayOfWeek'].cast('string'))
    df = df.withColumn('CRSDepTime', df['CRSDepTime'].cast('integer'))
    df = df.withColumn('CRSArrTime', df['CRSArrTime'].cast('integer'))
    df = df.withColumn('CRSElapsedTime', df['CRSElapsedTime'].cast('integer'))
    df = df.withColumn('DepDelay', df['DepDelay'].cast('integer'))
    df = df.withColumn('TaxiOut', df['TaxiOut'].cast('integer')) if 'TaxiOut' in df.columns else df
    df = df.withColumn('UniqueCarrier', df['UniqueCarrier'].cast('string'))
    df = df.withColumn('Origin', df['Origin'].cast('string'))
    df = df.withColumn('Dest', df['Dest'].cast('string'))
    df = df.withColumn('ArrDelay', df['ArrDelay'].cast('integer'))

    df.describe().show()

    # Remove rows with CRSElapsedTime (flight duration) less than 15 min and more than 15 hours
    df = df.filter((df['CRSElapsedTime'] >= 15) & (df['CRSElapsedTime'] <= 900))

    # Mean Taxi Out
    if 'TaxiOut' in df.columns:
        df_TaxiOutMean = df.groupby('Origin').agg(_mean('TaxiOut'))
        df_TaxiOutMean = df_TaxiOutMean.withColumnRenamed('Origin', 'Origin2')
        df = df.join(df_TaxiOutMean, df['Origin'] == df_TaxiOutMean['Origin2'], 'left')
        df = df.withColumn('TaxiOut', df['TaxiOut'] - df['avg(TaxiOut)'])
        df = df.drop(*('avg(TaxiOut)', 'Origin2'))

    # Reorder df, label -> numeric features -> nominal features
    df = df.select('ArrDelay', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'DepDelay', 'TaxiOut', 'UniqueCarrier',
                   'Origin', 'Dest', 'Month', 'DayofMonth', 'DayOfWeek')

    formula = RFormula(
        formula="ArrDelay ~ .",
        featuresCol="features",
        labelCol="label").fit(df)
    df = formula.transform(df)
    df = df.select("features", "label")

    return df
