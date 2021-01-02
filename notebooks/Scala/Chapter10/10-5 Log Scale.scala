// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Log Scale
// MAGIC 
// MAGIC In the lab, you will improve your model performance by transforming the label to log scale, predicting on log scale, then exponentiating to evaluate the result. 

// COMMAND ----------

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
val airbnbDF = spark.read.parquet(filePath)
val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

// COMMAND ----------

import org.apache.spark.sql.functions.{col, log}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.regression.LinearRegression

val logTrainDF = trainDF.withColumn("log_price", log(col("price")))
val logTestDF = testDF.withColumn("log_price", log(col("price")))

val rFormula = new RFormula()
  .setFormula("log_price ~ . - price")
  .setFeaturesCol("features")
  .setLabelCol("log_price")
  .setHandleInvalid("skip")

val lr = new LinearRegression()
             .setLabelCol("log_price")
             .setPredictionCol("log_pred")

val pipeline = new Pipeline().setStages(Array(rFormula, lr))
val pipelineModel = pipeline.fit(logTrainDF)
val predDF = pipelineModel.transform(logTestDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Exponentiate
// MAGIC 
// MAGIC In order to interpret our RMSE, we need to convert our predictions back from logarithmic scale.

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions.{col, exp}

val expDF = predDF.withColumn("prediction", exp(col("log_pred")))

val regressionEvaluator = new RegressionEvaluator()
  .setPredictionCol("prediction")
  .setLabelCol("price")

val rmse = regressionEvaluator.setMetricName("rmse").evaluate(expDF)
val r2 = regressionEvaluator.setMetricName("r2").evaluate(expDF)
println(s"RMSE is $rmse")
println(s"R2 is $r2")
println("*-"*80)
