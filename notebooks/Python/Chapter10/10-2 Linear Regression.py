# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # Regression: Predicting Rental Price
# MAGIC 
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC When we are building ML models, we don't want to look at our test data (why is that?). 
# MAGIC 
# MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set. We will use the `randomSplit` method [Python](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset).
# MAGIC 
# MAGIC **Question**: Why is it necessary to set a seed?

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)
print(f"There are {trainDF.cache().count()} rows in the training set, and {testDF.cache().count()} in the test set")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Question**: What happens if you change your cluster configuration?
# MAGIC 
# MAGIC To test this out, try spinning up a cluster with just one worker, and another with two workers. NOTE: This data is quite small (one partition), and you will need to test it out with a larger dataset (e.g. 2+ partitions) to see the difference, such as: `databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean-100p.parquet.` However, in this code below, we will simply repartition our data to simulate how it could have been partitioned differently on a different cluster configuration, and see if we get the same number of data points in our training set. 

# COMMAND ----------

(trainRepartitionDF, testRepartitionDF) = (airbnbDF
                                           .repartition(24)
                                           .randomSplit([.8, .2], seed=42))

print(trainRepartitionDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC When you do an 80/20 train/test split, it is an "approximate" 80/20 split. It is not an exact 80/20 split, and when we the partitioning of our data changes, we show that we get not only a different # of data points in train/test, but also different data points.
# MAGIC 
# MAGIC Our recommendation is to split your data once, then write it out to its own train/test folder so you don't have these reproducibility issues.

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to build a very simple linear regression model predicting `price` just given the number of `bedrooms`.
# MAGIC 
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

display(trainDF.select("price", "bedrooms").summary())

# COMMAND ----------

# MAGIC %md
# MAGIC There do appear some outliers in our dataset for the price ($10,000 a night??). Just keep this in mind when we are building our models :).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC 
# MAGIC Linear Regression expects a column of Vector type as input.
# MAGIC 
# MAGIC We can easily get the values from the `bedrooms` column into a single vector using `VectorAssembler` [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler). VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vecTrainDF = vecAssembler.transform(trainDF)

vecTrainDF.select("bedrooms", "features", "price").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC 
# MAGIC Now that we have prepared our data, we can use the `LinearRegression` estimator to build our first model [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.LinearRegression). Estimators accept a DataFrame as input and return a model, and have a `.fit()` method. 

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="price")
lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the model

# COMMAND ----------

m = round(lrModel.coefficients[0], 2)
b = round(lrModel.intercept, 2)

print(f"The formula for the linear regression line is price = {m}*bedrooms + {b}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply to Test Set

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

predDF.select("bedrooms", "features", "price", "prediction").show(10)
