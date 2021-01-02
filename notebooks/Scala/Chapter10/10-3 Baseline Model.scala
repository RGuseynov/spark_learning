// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Baseline Model
// MAGIC 
// MAGIC Let's compute the average `price` on the training dataset, and use that as our prediction column for our test dataset, then evaluate the result.

// COMMAND ----------

import org.apache.spark.sql.functions.{avg, lit}

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
val airbnbDF = spark.read.parquet(filePath)
val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

val avgPrice = trainDF.select(avg("price")).first().getDouble(0)

val predDF = testDF.withColumn("avgPrediction", lit(avgPrice))

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val regressionMeanEvaluator = new RegressionEvaluator()
  .setPredictionCol("avgPrediction")
  .setLabelCol("price")
  .setMetricName("rmse")

val rmse = regressionMeanEvaluator.evaluate(predDF)
println (f"The RMSE for predicting the average price is: $rmse%1.2f")
