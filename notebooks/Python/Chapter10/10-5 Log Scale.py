# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # Log Scale
# MAGIC 
# MAGIC In the lab, you will improve your model performance by transforming the label to log scale, predicting on log scale, then exponentiating to evaluate the result. 

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from pyspark.sql.functions import col, log
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression

logTrainDF = trainDF.withColumn("log_price", log(col("price")))
logTestDF = testDF.withColumn("log_price", log(col("price")))

rFormula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip") 

lr = LinearRegression(labelCol="log_price", predictionCol="log_pred")
pipeline = Pipeline(stages = [rFormula, lr])
pipelineModel = pipeline.fit(logTrainDF)
predDF = pipelineModel.transform(logTestDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exponentiate
# MAGIC 
# MAGIC In order to interpret our RMSE, we need to convert our predictions back from logarithmic scale.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, exp

expDF = predDF.withColumn("prediction", exp(col("log_pred")))

regressionEvaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
rmse = regressionEvaluator.setMetricName("rmse").evaluate(expDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(expDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
