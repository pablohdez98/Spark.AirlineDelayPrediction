from pyspark.sql import SparkSession
from preprocessing import *

spark = SparkSession.builder.appName('Practise').getOrCreate()

load_data(spark)

'''
df.show()
print(df.count())
df.groupby('Year').avg().show()
df.groupby('Month').avg().show()
df.groupby('DayofMonth').avg().show()
df.groupby('DayOfWeek').avg().show()
'''

'''
for file in os.listdir():
    print(file)
    df_pyspark = spark.read.csv('../data/'+file, header=True)
    df_pyspark.show()
'''