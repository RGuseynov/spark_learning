// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Hyperparameter Tuning
// MAGIC 
// MAGIC Let's perform hyperparameter tuning on a random forest to find the best hyperparameters!

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
val airbnbDF = spark.read.parquet(filePath)
val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price"}.map(_._1)
val assemblerInputs = indexOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Random Forest

// COMMAND ----------

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.Pipeline

val rf = new RandomForestRegressor()
  .setLabelCol("price")
  .setMaxBins(40)
  .setSeed(42)

val pipeline = new Pipeline()
  .setStages(Array(stringIndexer, vecAssembler, rf))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Grid Search
// MAGIC 
// MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
// MAGIC 
// MAGIC Let's use Spark's `ParamGridBuilder` to find the optimal hyperparameters in a more systematic approach [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.ParamGridBuilder).
// MAGIC 
// MAGIC Let's define a grid of hyperparameters to test:
// MAGIC   - maxDepth: max depth of the decision tree (Use the values `2, 4, 6`)
// MAGIC   - numTrees: number of decision trees (Use the values `10, 100`)

// COMMAND ----------

import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(2, 4, 6))
  .addGrid(rf.numTrees, Array(10, 100))
  .build()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Cross Validation
// MAGIC 
// MAGIC We are also going to use 3-fold cross validation to identify the optimal maxDepth.
// MAGIC 
// MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
// MAGIC 
// MAGIC With 3-fold cross-validation, we train on 2/3 of the data, and evaluate with the remaining (held-out) 1/3. We repeat this process 3 times, so each fold gets the chance to act as the validation set. We then average the results of the three rounds.

// COMMAND ----------

// MAGIC %md
// MAGIC We pass in the `estimator` (pipeline), `evaluator`, and `estimatorParamMaps` to `CrossValidator` so that it knows:
// MAGIC - Which model to use
// MAGIC - How to evaluate the model
// MAGIC - What hyperparameters to set for the model
// MAGIC 
// MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.CrossValidator).

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator

val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val cv = new CrossValidator()
 .setEstimator(pipeline)
 .setEvaluator(evaluator)
 .setEstimatorParamMaps(paramGrid)
 .setNumFolds(3)
 .setSeed(42)

// COMMAND ----------

// MAGIC %md
// MAGIC **Question**: How many models are we training right now?

// COMMAND ----------

val cvModel = cv.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Parallelism Parameter
// MAGIC 
// MAGIC Hmmm... that took a long time to run. That's because the models were being trained sequentially rather than in parallel!
// MAGIC 
// MAGIC Spark 2.3 introduced a [parallelism](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator.parallelism) parameter. From the docs: `the number of threads to use when running parallel algorithms (>= 1)`.
// MAGIC 
// MAGIC Let's set this value to 4 and see if we can train any faster.

// COMMAND ----------

val cvModel = cv.setParallelism(4).fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC **Question**: Hmmm... that still took a long time to run. Should we put the pipeline in the cross validator, or the cross validator in the pipeline?
// MAGIC 
// MAGIC It depends if there are estimators or transformers in the pipeline. If you have things like StringIndexer (an estimator) in the pipeline, then you have to refit it every time if you put the entire pipeline in the cross validator.

// COMMAND ----------

val cv = new CrossValidator()
  .setEstimator(rf)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
  .setParallelism(4)
  .setSeed(42)

val pipeline = new Pipeline()
                   .setStages(Array(stringIndexer, vecAssembler, cv))

val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC Let's take a look at the model with the best hyperparameter configuration

// COMMAND ----------

cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC Let's see how it does on the test dataset.

// COMMAND ----------

val predDF = pipelineModel.transform(testDF)

val regressionEvaluator = new RegressionEvaluator()
  .setPredictionCol("prediction")
  .setLabelCol("price")
  .setMetricName("rmse")

val rmse = regressionEvaluator.evaluate(predDF)
val r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
println(s"RMSE is $rmse")
println(s"R2 is $r2")
println("*-"*80)
