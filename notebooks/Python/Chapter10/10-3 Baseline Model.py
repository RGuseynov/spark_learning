# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # Baseline Model
# MAGIC 
# MAGIC Let's compute the average `price` on the training dataset, and use that as our prediction column for our test dataset, then evaluate the result.

# COMMAND ----------

from pyspark.sql.functions import avg, lit

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

avgPrice = float(trainDF.select(avg("price")).first()[0])
predDF = testDF.withColumn("avgPrediction", lit(avgPrice))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionMeanEvaluator = RegressionEvaluator(predictionCol="avgPrediction", labelCol="price", metricName="rmse")

print(f"The RMSE for predicting the average price is: {regressionMeanEvaluator.evaluate(predDF):.2f}")
