// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # XGBoost
// MAGIC  
// MAGIC If you are not using the DBR 7.x ML Runtime, you will need to install `ml.dmlc:xgboost4j-spark_2.12:1.0.0` from Maven, well as `xgboost` from PyPI.
// MAGIC 
// MAGIC **NOTE:** There is currently only a distributed version of XGBoost for Scala, not Python. We will switch to Scala for that section.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Data Preparation
// MAGIC 
// MAGIC Let's go ahead and index all of our categorical features, and set our label to be `log(price)`.

// COMMAND ----------

import org.apache.spark.sql.functions.log
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
val airbnbDF = spark.read.parquet(filePath)
val Array(trainDF, testDF) = airbnbDF.withColumn("label", log($"price")).randomSplit(Array(.8, .2), seed=42)

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price" && field != "label"}.map(_._1)
val assemblerInputs = indexOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

val pipeline = new Pipeline()
  .setStages(Array(stringIndexer, vecAssembler))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Scala
// MAGIC 
// MAGIC Distributed XGBoost with Spark only has a Scala API, so we are going to create views of our DataFrames to use in Scala, as well as save our (untrained) pipeline to load in to Scala.

// COMMAND ----------

trainDF.createOrReplaceTempView("trainDF")
testDF.createOrReplaceTempView("testDF")

val fileName = "/tmp/xgboost_feature_pipeline"
pipeline.write.overwrite().save(fileName)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Load Data/Pipeline in Scala
// MAGIC 
// MAGIC This section is only available in Scala because there is no distributed Python API for XGBoost in Spark yet.
// MAGIC 
// MAGIC Let's load in our data/pipeline that we defined in Python. 

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.ml.Pipeline
// MAGIC 
// MAGIC val fileName = "tmp/xgboost_feature_pipeline"
// MAGIC val pipeline = Pipeline.load(fileName)
// MAGIC 
// MAGIC val trainDF = spark.table("trainDF")
// MAGIC val testDF = spark.table("testDF")

// COMMAND ----------

// MAGIC %md
// MAGIC ## XGBoost
// MAGIC 
// MAGIC Now we are ready to train our XGBoost model!

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import ml.dmlc.xgboost4j.scala.spark._
// MAGIC import org.apache.spark.sql.functions._
// MAGIC 
// MAGIC val paramMap = List("num_round" -> 100, "eta" -> 0.1, "max_leaf_nodes" -> 50, "seed" -> 42, "missing" -> 0).toMap
// MAGIC 
// MAGIC val xgboostEstimator = new XGBoostRegressor(paramMap)
// MAGIC 
// MAGIC val xgboostPipeline = new Pipeline().setStages(pipeline.getStages ++ Array(xgboostEstimator))
// MAGIC 
// MAGIC val xgboostPipelineModel = xgboostPipeline.fit(trainDF)
// MAGIC val xgboostLogPredictedDF = xgboostPipelineModel.transform(testDF)
// MAGIC 
// MAGIC val expXgboostDF = xgboostLogPredictedDF.withColumn("prediction", exp(col("prediction")))
// MAGIC expXgboostDF.createOrReplaceTempView("expXgboostDF")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate
// MAGIC 
// MAGIC Now we can evaluate how well our XGBoost model performed.

// COMMAND ----------

val expXgboostDF = spark.table("expXgboostDF")

display(expXgboostDF.select("price", "prediction"))

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val regressionEvaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = regressionEvaluator.evaluate(expXgboostDF)
val r2 = regressionEvaluator.setMetricName("r2").evaluate(expXgboostDF)
println(s"RMSE is $rmse")
println(s"R2 is $r2")
println("*-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Export to Python
// MAGIC 
// MAGIC We can also export our XGBoost model to use in Python for fast inference on small datasets.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val nativeModelPath = "xgboost_native_model"
// MAGIC val xgboostModel = xgboostPipelineModel.stages.last.asInstanceOf[XGBoostRegressionModel]
// MAGIC xgboostModel.nativeBooster.saveModel(nativeModelPath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Predictions in Python
// MAGIC 
// MAGIC Let's pass in an example record to our Python XGBoost model and see how fast we can get predictions!!
// MAGIC 
// MAGIC Don't forget to exponentiate!

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC import xgboost as xgb
// MAGIC bst = xgb.Booster({'nthread': 4})
// MAGIC bst.load_model("xgboost_native_model")
// MAGIC 
// MAGIC # Per https://stackoverflow.com/questions/55579610/xgboost-attributeerror-dataframe-object-has-no-attribute-feature-names, DMatrix did the trick
// MAGIC 
// MAGIC data = np.array([[0.0, 2.0, 0.0, 14.0, 1.0, 0.0, 0.0, 1.0, 37.72001, -122.39249, 2.0, 1.0, 1.0, 1.0, 2.0, 128.0, 97.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
// MAGIC log_pred = bst.predict(xgb.DMatrix(data))
// MAGIC print(f"The predicted price for this rental is ${np.exp(log_pred)[0]:.2f}")