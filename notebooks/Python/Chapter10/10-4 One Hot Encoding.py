# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # One-Hot Encoding
# MAGIC 
# MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results apples to apples (unless you changed the cluster config!)

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: StringIndexer, OneHotEncoder, and VectorAssembler
# MAGIC 
# MAGIC Here, we are going to One Hot Encode (OHE) our categorical variables. The first approach we are going to use will combine StringIndexer, OneHotEncoder, and VectorAssembler.
# MAGIC 
# MAGIC First we need to use `StringIndexer` to map a string column of labels to an ML column of label indices [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StringIndexer)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.StringIndexer).
# MAGIC 
# MAGIC Then, we can apply the `OneHotEncoder` to the output of the StringIndexer [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.OneHotEncoder).

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categoricalCols = [field for (field, dataType) in trainDF.dtypes 
                   if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]
oheOutputCols = [x + "OHE" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, 
                              outputCols=indexOutputCols, 
                              handleInvalid="skip")
oheEncoder = OneHotEncoder(inputCols=indexOutputCols, 
                           outputCols=oheOutputCols)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can combine our OHE categorical features with our numeric features.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numericCols = [field for (field, dataType) in trainDF.dtypes 
               if ((dataType == "double") & (field != "price"))]
assemblerInputs = oheOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, 
                               outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 2: RFormula
# MAGIC Instead of manually specifying which columns are categorical to the StringIndexer and OneHotEncoder, RFormula can do that automatically for you [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RFormula)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.RFormula).
# MAGIC 
# MAGIC With RFormula, if you have any columns of type String, it treats it as a categorical feature and string indexes & one hot encodes it for us. Otherwise, it leaves as it is. Then it combines all of one-hot encoded features and numeric features into a single vector, called `features`.

# COMMAND ----------

from pyspark.ml.feature import RFormula

rFormula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC 
# MAGIC Now that we have all of our features, let's build a linear regression model.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline
# MAGIC 
# MAGIC Let's put all these stages in a Pipeline. A `Pipeline` is a way of organizing all of our transformers and estimators [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Pipeline).
# MAGIC 
# MAGIC Verify you get the same results with Option 1 (StringIndexer, OneHotEncoderEstimator, and VectorAssembler) and Option 2 (RFormula)

# COMMAND ----------

# Option 1: StringIndexer + OHE + VectorAssembler
from pyspark.ml import Pipeline

stages = [stringIndexer, oheEncoder, vecAssembler, lr]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

# COMMAND ----------

# Option 2: RFormula
from pyspark.ml import Pipeline

pipeline = Pipeline(stages = [rFormula, lr])

pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Model: RMSE

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = round(regressionEvaluator.evaluate(predDF), 2)
print(f"RMSE is {rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## R2
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing? 

# COMMAND ----------

r2 = round(regressionEvaluator.setMetricName("r2").evaluate(predDF), 2)
print(f"R2 is {r2}")

# COMMAND ----------

pipelinePath = "/tmp/sf-airbnb/lr-pipeline-model"
pipelineModel.write().overwrite().save(pipelinePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading models
# MAGIC 
# MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
# MAGIC 
# MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic PipelineModel back in.

# COMMAND ----------

from pyspark.ml import PipelineModel

savedPipelineModel = PipelineModel.load(pipelinePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Setting
# MAGIC 
# MAGIC If you are interested in learning how linear regression is implemented in the distributed setting and bottlenecks, check out these lecture slides:
# MAGIC * [distributed-linear-regression-1](https://files.training.databricks.com/static/docs/distributed-linear-regression-1.pdf)
# MAGIC * [distributed-linear-regression-2](https://files.training.databricks.com/static/docs/distributed-linear-regression-2.pdf)