
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('RemoveDuplicates').getOrCreate()

# Read the CSV file
df = spark.read.csv('./src/data/world_population.csv', header=True, inferSchema=True)

# Remove duplicates
df = df.dropDuplicates()

# Print total data count and show the DataFrame
print('Total data:', df.count())
df.show()

# Save the DataFrame as a CSV file
file_name = 'world_population_nodup.csv'
content = df.toPandas()
with open(file_name, 'w') as f:
    content.to_csv(f, index=False)


