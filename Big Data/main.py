from click.core import F
from pyspark.sql import SparkSession
from pyspark.sql.connect.functions import avg, col, mean
from pyspark.sql.functions import when
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.sql.functions import col, sum
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression


spark = SparkSession.builder \
    .appName("CassandraSpark") \
    .config("spark.cassandra.connection.host", "localhost") \
    .config("spark.cassandra.connection.port", "9042") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.cassandra.output.consistency.level", "LOCAL_ONE") \
    .getOrCreate()


df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="traffic", keyspace="bigdata") \
    .load()
df2 = df.limit(100)

df3 = df2.withColumn("severity", col("severity").cast("double"))

imputer = Imputer(
          inputCols=['temperature', 'wind_chill','humidity', 'pressure', 'visibility','wind_speed','precipitation'],
          outputCols=["{}_new".format(c) for c in ['temperature', 'wind_chill','humidity', 'pressure', 'visibility','wind_speed','precipitation']]
  ).setStrategy("mean")
df3=imputer.fit(df3).transform(df3)
df3=df3.na.fill('Unknown',["airport_code","astronomical_twilight","city","Civil_Twilight","Nautical_Twilight","sunrise_sunset","timezone","wind_direction","zipcode"])


cols_to_drop = ['temperature', 'wind_chill','humidity', 'pressure', 'visibility','wind_speed','precipitation']
df3 = df3.drop(*cols_to_drop)

assembler = VectorAssembler(
    inputCols=["temperature_new", "wind_chill_new", "humidity_new", "pressure_new", "visibility_new", "wind_speed_new", "precipitation_new"],
    outputCol="features"
)
df3 = assembler.transform(df3)

df3 = df3.withColumn("label", df3["severity"])

train, test = df3.randomSplit([0.7, 0.3], seed=2018)
train_count=train.count()
test_count=test.count()


lr = LogisticRegression(labelCol="severity", featuresCol="features")
model = lr.fit(train)

predictions = model.transform(test)


res = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='severity')


ROC_AUC = res.evaluate(predictions)
print(predictions.select('severity', 'rawPrediction', 'prediction', 'probability').show(10))
print(predictions.show())
print("Training Dataset Count: " + str(train_count))
print("Test Dataset Count: " + str(test_count))
print("ROC_AUC="+ str(ROC_AUC))
spark.stop()

