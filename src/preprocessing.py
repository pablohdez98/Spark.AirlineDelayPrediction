def load_data(spark):
    df = spark.read.csv("../data/2008.csv", header=True)
    df.show()
    print(df.columns)
    df = df.drop(*('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay',
                   'NASDelay', 'SecurityDelay', 'LateAircraftDelay'))
    print(df.columns)

