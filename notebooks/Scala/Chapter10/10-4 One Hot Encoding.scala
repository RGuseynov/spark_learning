// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # One-Hot Encoding
// MAGIC 
// MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.

// COMMAND ----------

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
val airbnbDF = spark.read.parquet(filePath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Train/Test Split
// MAGIC 
// MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results apples to apples (unless you changed the cluster config!)

// COMMAND ----------

val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Option 1: StringIndexer, OneHotEncoder, and VectorAssembler
// MAGIC 
// MAGIC Here, we are going to One Hot Encode (OHE) our categorical variables. The first approach we are going to use will combine StringIndexer, OneHotEncoder, and VectorAssembler.
// MAGIC 
// MAGIC First we need to use `StringIndexer` to map a string column of labels to an ML column of label indices [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StringIndexer)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.StringIndexer).
// MAGIC 
// MAGIC Then, we can apply the `OneHotEncoder` to the output of the StringIndexer [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.OneHotEncoder).

// COMMAND ----------

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")
val oheOutputCols = categoricalCols.map(_ + "OHE")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val oheEncoder = new OneHotEncoder()
  .setInputCols(indexOutputCols)
  .setOutputCols(oheOutputCols)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can combine our OHE categorical features with our numeric features.

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => 
  dataType == "DoubleType" && field != "price"}.map(_._1)
val assemblerInputs = oheOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Option 2: RFormula
// MAGIC Instead of manually specifying which columns are categorical to the StringIndexer and OneHotEncoder, RFormula can do that automatically for you [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RFormula)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.RFormula).
// MAGIC 
// MAGIC With RFormula, if you have any columns of type String, it treats it as a categorical feature and string indexes & one hot encodes it for us. Otherwise, it leaves as it is. Then it combines all of one-hot encoded features and numeric features into a single vector, called `features`.

// COMMAND ----------

import org.apache.spark.ml.feature.RFormula

val rFormula = new RFormula()
  .setFormula("price ~ .")
  .setFeaturesCol("features")
  .setLabelCol("price")
  .setHandleInvalid("skip")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Linear Regression
// MAGIC 
// MAGIC Now that we have all of our features, let's build a linear regression model.

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
  .setLabelCol("price")
  .setFeaturesCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Pipeline
// MAGIC 
// MAGIC Let's put all these stages in a Pipeline. A `Pipeline` is a way of organizing all of our transformers and estimators [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Pipeline).
// MAGIC 
// MAGIC Verify you get the same results with Option 1 (StringIndexer, OneHotEncoderEstimator, and VectorAssembler) and Option 2 (RFormula)

// COMMAND ----------

// Option 1: StringIndexer + OHE + VectorAssembler
import org.apache.spark.ml.Pipeline

val stages = Array(stringIndexer, oheEncoder, vecAssembler,  lr)

val pipeline = new Pipeline()
  .setStages(stages)

val pipelineModel = pipeline.fit(trainDF)
val predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

// COMMAND ----------

// Option 2: RFormula
import org.apache.spark.ml.Pipeline

val pipeline = new Pipeline().setStages(Array(rFormula, lr))

val pipelineModel = pipeline.fit(trainDF)
val predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate Model: RMSE

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val regressionEvaluator = new RegressionEvaluator()
  .setPredictionCol("prediction")
  .setLabelCol("price")
  .setMetricName("rmse")

val rmse = regressionEvaluator.evaluate(predDF)
println(f"RMSE is $rmse%1.2f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## R2
// MAGIC 
// MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing? 

// COMMAND ----------

val r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
println(f"R2 is $r2%1.2f")

// COMMAND ----------

val pipelinePath = "/tmp/sf-airbnb/lr-pipeline-model"
pipelineModel.write.overwrite().save(pipelinePath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Loading models
// MAGIC 
// MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
// MAGIC 
// MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic PipelineModel back in.

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

val savedPipelineModel = PipelineModel.load(pipelinePath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Distributed Setting
// MAGIC 
// MAGIC If you are interested in learning how linear regression is implemented in the distributed setting and bottlenecks, check out these lecture slides:
// MAGIC * [distributed-linear-regression-1](https://files.training.databricks.com/static/docs/distributed-linear-regression-1.pdf)
// MAGIC * [distributed-linear-regression-2](https://files.training.databricks.com/static/docs/distributed-linear-regression-2.pdf)