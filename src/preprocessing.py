import sys

from pyspark.ml.feature import RFormula, VectorAssembler, PCA
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import mean as _mean
from pyspark.sql.functions import isnan, when, count, col
import seaborn as sns
import matplotlib.pyplot as plt


def load_csv(spark, file):
    # Check if file exists
    try:
        df = spark.read.csv('../data/' + str(file) + '.csv', header=True, inferSchema=True)
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

    # All columns that a csv should have
    all_columns = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier',
                   'FlightNum', 'TailNum', 'CRSElapsedTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance',
                   'TaxiOut', 'Cancelled', 'CancellationCode']

    # Check if the columns remaining are in the file
    for c in all_columns:
        if df.schema.simpleString().find(c) == -1:  # if -1, a column is missing and execution is stopped
            print("Exception, there is at least one column missing")
            sys.exit(1)

    # Parse all columns to their type
    df = df.withColumn('Year', df['Month'].cast('integer'))
    df = df.withColumn('Month', df['Month'].cast('string'))
    df = df.withColumn('DayofMonth', df['DayofMonth'].cast('string'))
    df = df.withColumn('DayOfWeek', df['DayOfWeek'].cast('string'))
    df = df.withColumn('DepTime', df['DepTime'].cast('integer'))
    df = df.withColumn('CRSDepTime', df['CRSDepTime'].cast('integer'))
    df = df.withColumn('CRSArrTime', df['CRSArrTime'].cast('integer'))
    df = df.withColumn('UniqueCarrier', df['UniqueCarrier'].cast('string'))
    df = df.withColumn('FlightNum', df['FlightNum'].cast('integer'))
    df = df.withColumn('TailNum', df['TailNum'].cast('string'))
    df = df.withColumn('CRSElapsedTime', df['CRSElapsedTime'].cast('integer'))
    df = df.withColumn('ArrDelay', df['ArrDelay'].cast('integer'))
    df = df.withColumn('DepDelay', df['DepDelay'].cast('integer'))
    df = df.withColumn('Origin', df['Origin'].cast('string'))
    df = df.withColumn('Dest', df['Dest'].cast('string'))
    df = df.withColumn('Distance', df['Distance'].cast('integer'))
    df = df.withColumn('TaxiOut', df['TaxiOut'].cast('integer'))
    df = df.withColumn('Cancelled', df['Cancelled'].cast('string'))
    df = df.withColumn('CancellationCode', df['CancellationCode'].cast('string'))

    # Remove canceled flights
    df = df.replace('NA', None)
    df = df.na.drop(how='any', subset=['ArrDelay'])

    return df


def analysis(df):
    print("PERFORMING ANALYSIS...")
    df = df.withColumn('Month', df['Month'].cast('integer'))
    df = df.withColumn('DayofMonth', df['DayofMonth'].cast('integer'))
    df = df.withColumn('DayOfWeek', df['DayOfWeek'].cast('integer'))

    df = df.select(['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'FlightNum',
                    'CRSElapsedTime', 'ArrDelay', 'DepDelay', 'Distance', 'TaxiOut'])

    assembler = VectorAssembler(inputCols=df.columns, outputCol='features')
    df_vector = assembler.transform(df).select('features')

    # Correlation
    print("Calculating correlation...")
    matrix = Correlation.corr(df_vector, 'features').collect()[0][0]
    print(matrix)

    # Check linearity between DepDelay and ArrDelay with a plot
    print("Plot of data between DepDelay and ArrDelay(label)...")
    s_df = df.select(['DepDelay', 'ArrDelay'])\
        .sample(withReplacement=False, fraction=0.5, seed=42)  # Reduce dataset, hard to plot it
    s_df_pandas = s_df.toPandas()  # Change to Pandas in order to plot it
    sns.lmplot(y='ArrDelay', x='DepDelay', data=s_df_pandas)
    plt.show()

    print("Calculating PCA...")
    pca = PCA(k=5, inputCol="features", outputCol="PCA")
    model = pca.fit(df_vector)

    result = model.transform(df_vector).select("PCA")
    result.show(truncate=False)


def process_data(df):
    # Remove useless variables
    df = df.drop(
        *('Year', 'CancellationCode', 'DepTime', 'FlightNum', 'TailNum', 'Distance', 'Cancelled', 'CRSArrTime'))

    # Check if columns have more than 50% empty values
    df_nan = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    for c in df_nan.columns:
        if (df.count() * 0.5) < df_nan.first()[c]:
            df = df.drop(c)

    # Remove rows with NA values
    df = df.na.drop(how='any')

    # Remove rows with CRSElapsedTime (flight duration) less than 15 min and more than 15 hours
    df = df.filter((df['CRSElapsedTime'] >= 15) & (df['CRSElapsedTime'] <= 900))

    # COLUMN TRANSFORMATIONS
    # Mean Taxi Out
    if 'TaxiOut' in df.columns:
        df_TaxiOutMean = df.groupby('Origin').agg(_mean('TaxiOut'))
        df_TaxiOutMean = df_TaxiOutMean.withColumnRenamed('Origin', 'Origin2')
        df = df.join(df_TaxiOutMean, df['Origin'] == df_TaxiOutMean['Origin2'], 'left')
        df = df.withColumn('TaxiOut', df['TaxiOut'] - df['avg(TaxiOut)'])
        df = df.drop(*('avg(TaxiOut)', 'Origin2'))

    # Reorder df, label -> numeric features -> nominal features
    df = df.select('ArrDelay', 'CRSDepTime', 'CRSElapsedTime', 'DepDelay', 'TaxiOut', 'UniqueCarrier',
                   'Origin', 'Dest', 'Month', 'DayofMonth', 'DayOfWeek')

    # Transform dataframe to another one with 2 columns: label (output) and features (all predictors combined in a
    # vector). For categorical columns, OneHotCoding is applied
    formula = RFormula(
        formula="ArrDelay ~ .",
        featuresCol="features",
        labelCol="label").fit(df)
    df = formula.transform(df)
    df = df.select("features", "label")

    return df
