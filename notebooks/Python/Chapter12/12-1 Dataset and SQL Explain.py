# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC ### Use `ds.explain("format")`
# MAGIC 
# MAGIC Format include `simple`, `extended`, `codegen`, `cost`, `formatted`

# COMMAND ----------

strings = spark.read.text("/databricks-datasets/learning-spark-v2/SPARK_README.md")
filtered = strings.filter(strings.value.contains("Spark"))
filtered.count()

# COMMAND ----------

filtered.explain(mode="simple")

# COMMAND ----------

filtered.explain(mode="extended")

# COMMAND ----------

filtered.explain(mode="cost")

# COMMAND ----------

filtered.explain(mode="codegen")

# COMMAND ----------

filtered.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use SQL EXPLAIN format

# COMMAND ----------

strings.createOrReplaceTempView("tmp_spark_readme")

# COMMAND ----------

# MAGIC %sql
# MAGIC EXPLAIN FORMATTED SELECT * FROM tmp_spark_readme WHERE value % 'Spark'