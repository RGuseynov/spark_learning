// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC ## Example 2-1 Line Count

// COMMAND ----------

spark.version

// COMMAND ----------

val strings = spark.read.text("/databricks-datasets/learning-spark-v2/SPARK_README.md")
strings.show(10, false)

// COMMAND ----------

strings.count()

// COMMAND ----------

val filtered = strings.filter($"value".contains("Spark"))
filtered.count()
