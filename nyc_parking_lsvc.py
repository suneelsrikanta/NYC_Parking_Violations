import logging
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier, LinearSVC
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType
import pandas as pd
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Spark setup
sc = SparkContext(master="local[*]")
spark = SparkSession(sc)

# Schema definition
parkingSchema = StructType([
    StructField("Plate", StringType(), False),
    StructField("State", StringType(), False),
    StructField("Vehicle Body Type", StringType(), False),
    StructField("Vehicle Make", StringType(), False),
    StructField("Violation Precinct", IntegerType(), False),
    StructField("Issuer Precinct", IntegerType(), False),
    StructField("Violation Time", StringType(), False),
    StructField("Violation County", StringType(), False),
    StructField("Violation Code", IntegerType(), False),
    StructField("DisputeStatus", IntegerType(), False)
])

# Load dataset
df = spark.read.format("csv") \
    .schema(parkingSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/user/ssrikan3/Project/NYC_Parking.csv")


# Add simulated dispute label
# df = df.withColumn("DisputeStatus", when(rand() > 0.7, 1).otherwise(0))
# df = df.withColumn("DisputeStatus", col("DisputeStatus").cast("integer"))
# Fill nulls
# df = df.fillna("UNKNOWN", subset=["State", "Vehicle Body Type", "Vehicle Make", "Violation Time", "Violation County"])


# Index categorical columns
indexers = [
    StringIndexer(inputCol="State", outputCol="StateIndex", handleInvalid="keep"),
    StringIndexer(inputCol="Vehicle Body Type", outputCol="VehicleBodyTypeIndex", handleInvalid="keep"),
    StringIndexer(inputCol="Vehicle Make", outputCol="VehicleMakeIndex", handleInvalid="keep"),
    StringIndexer(inputCol="Violation Time", outputCol="ViolationTimeIndex", handleInvalid="keep"),
    StringIndexer(inputCol="Violation County", outputCol="ViolationCountyIndex", handleInvalid="keep")
]


for indexer in indexers:
    df = indexer.fit(df).transform(df)
    
# Cast columns for modeling
for col_name in ["StateIndex", "VehicleBodyTypeIndex", "VehicleMakeIndex", "ViolationTimeIndex", "ViolationCountyIndex"]:
    df = df.withColumn(col_name, col(col_name).cast("double"))
    

df = df.withColumn("ViolationPrecinct", col("Violation Precinct").cast("double"))
df = df.withColumn("IssuerPrecinct", col("Issuer Precinct").cast("double"))
# df = df.withColumn("DisputeStatus", col("DisputeStatus").cast("integer"))


df = df.withColumn("DisputeStatus", when(col("Violation Code").isin(21, 36), 1).otherwise(0))

# Final DataFrame for Model
df3 = df.select("StateIndex", "VehicleBodyTypeIndex", "VehicleMakeIndex", "ViolationTimeIndex", "ViolationCountyIndex", col("Violation Precinct").alias("ViolationPrecinct"), col("Issuer Precinct").alias("IssuerPrecinct"), col("DisputeStatus").alias("label"))
df3.show(2)


# Oversampling to balance class distribution
label1_df = df3.filter(col("label") == 1)
label0_df = df3.filter(col("label") == 0)

label1_count = label1_df.count()
label0_count = label0_df.count()

if label1_count == 0:
    print("No positive samples. Falling back to synthetic label for debugging.")
    df = df.withColumn("DisputeStatus", when(rand() < 0.3, 1).otherwise(0))
    df3 = df.select(
        "StateIndex", "VehicleBodyTypeIndex", "VehicleMakeIndex",
        "ViolationTimeIndex", "ViolationCountyIndex",
        col("Violation Precinct").alias("ViolationPrecinct"),
        col("Issuer Precinct").alias("IssuerPrecinct"),
        col("DisputeStatus").alias("label")
    )
    label1_df = df3.filter(col("label") == 1)
    label0_df = df3.filter(col("label") == 0)
    label1_count = label1_df.count()
    label0_count = label0_df.count()

# Perform oversampling now that we have valid counts
ratio = label0_count / label1_count
oversampled = label1_df.sample(withReplacement=True, fraction=ratio)
df3 = label0_df.union(oversampled)

featureCols = ["StateIndex", "VehicleBodyTypeIndex", "VehicleMakeIndex", "ViolationTimeIndex", "ViolationCountyIndex", "ViolationPrecinct", "IssuerPrecinct"]


# Cache to speed up multiple operations
df3 = df3.cache()
# For development/testing, you can uncomment the next line to limit data size
#df3 = df3.limit(50000)
# Fill any missing feature values with 0.0
df3 = df3.fillna(0.0, subset=featureCols)

# Assemble features
assembler = VectorAssembler(
    inputCols=["StateIndex", "VehicleBodyTypeIndex", "VehicleMakeIndex", 
               "ViolationTimeIndex", "ViolationCountyIndex", 
               "ViolationPrecinct", "IssuerPrecinct"],
    outputCol="features"
)


# Train-test split
df3 = df3.filter(col("label").isNotNull())
splits = df3.randomSplit([0.7, 0.3])
train = splits[0].cache()
test = splits[1].withColumnRenamed("label", "trueLabel").cache()
print ("Training Rows:", train.count(), " Testing Rows:", test.count())


# LSVC Classifier
svc = LinearSVC(featuresCol="features", labelCol="label", maxIter=20)
pipeline = Pipeline(stages=[assembler, svc])

# Parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(svc.maxIter, [0.01, 0.5]) \
    .addGrid(svc.regParam,  [5, 15]) \
    .build()

################### TrainValidationSplit ##################
tv_lsvc = TrainValidationSplit(estimator=pipeline,
                          evaluator=BinaryClassificationEvaluator(),
                          estimatorParamMaps=paramGrid,
                          trainRatio=0.8)

# Train model
# Optionally save the model to avoid retraining later:
# tvModel.save("/path/to/save/tvModel")
start_time = time.time()
tvModel = tv_lsvc.fit(train)
end_time = time.time()
print("*********** Model trained! ***************")

# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

logging.info("Training time: %02d:%02d" % (minutes, seconds))

# Evaluate model
prediction = tvModel.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")
predicted.show(100, truncate=False)

# Metrics
tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())

metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", float(tp / (tp + fp) if (tp + fp) else 0)),
 ("Recall", float(tp / (tp + fn) if (tp + fn) else 0))
 ],["metric", "value"])
 
metrics.show()

logging.info("********************** TrainValidator-Results-GBT ***********************")

# AUC
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print("AUC =", auc)


################# CrossValidator #################
# Create a CrossValidator
cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, numFolds=3)


start_time = time.time()
cvModel = cv.fit(train)
end_time = time.time()
print("*********** Model trained! ***************")

# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

logging.info("Training time: %02d:%02d" % (minutes, seconds))

# COMMAND ----------

prediction = cvModel.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")

predicted.show(100, truncate=False)

# COMMAND ----------

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())

metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", float(tp / (tp + fp) if (tp + fp) else 0)),
 ("Recall", float(tp / (tp + fn) if (tp + fn) else 0))
 ],["metric", "value"])
 
metrics.show()

logging.info("***********CrossValidator-Results-GBT************")

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print("AUC = ", auc)
