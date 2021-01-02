# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC ## Example 2-1 Line Count

# COMMAND ----------

spark.version

# COMMAND ----------

strings = spark.read.text("/databricks-datasets/learning-spark-v2/SPARK_README.md")
strings.show(10, truncate=False)

# COMMAND ----------

strings.count()

# COMMAND ----------

filtered = strings.filter(strings.value.contains("Spark"))
filtered.count()
