# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # Caching Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use _cache()_

# COMMAND ----------

# MAGIC %md
# MAGIC Create a large data set with couple of columns

# COMMAND ----------

from pyspark.sql.functions import col
df = spark.range(1 * 10000000).toDF("id").withColumn("square", col("id") * col("id"))
df.cache().count()

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Check the Spark UI storage tab to see where the data is stored.

# COMMAND ----------

df.unpersist() # If you do not unpersist, df2 below will not be cached because it has the same query plan as df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use _persist(StorageLevel.Level)_

# COMMAND ----------

from pyspark import StorageLevel

df2 = spark.range(1 * 10000000).toDF("id").withColumn("square", col("id") * col("id"))
df2.persist(StorageLevel.DISK_ONLY).count()

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Check the Spark UI storage tab to see where the data is stored

# COMMAND ----------

df2.unpersist()

# COMMAND ----------

df.createOrReplaceTempView("dfTable")
spark.sql("CACHE TABLE dfTable")

# COMMAND ----------

# MAGIC %md
# MAGIC Check the Spark UI storage tab to see where the data is stored.

# COMMAND ----------

spark.sql("SELECT count(*) FROM dfTable").show()
