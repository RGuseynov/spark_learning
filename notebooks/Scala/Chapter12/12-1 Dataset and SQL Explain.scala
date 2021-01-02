// Databricks notebook source

val ds = spark.createDataset(Seq(20, 3, 3, 2, 4, 8, 1, 1, 3))

// COMMAND ----------

ds.show()

// COMMAND ----------

ds.groupByKey(l => l).count.show

// COMMAND ----------

// MAGIC %md
// MAGIC ### Use `ds.explain("format")`
// MAGIC 
// MAGIC Format include `simple`, `extended`, `codegen`, `cost`, `formatted`

// COMMAND ----------

val strings = spark.read.text("/databricks-datasets/learning-spark-v2/SPARK_README.md")
val filtered = strings.filter($"value".contains("Spark"))
filtered.count()

// COMMAND ----------

filtered.explain("simple")

// COMMAND ----------

filtered.explain("extended")

// COMMAND ----------

filtered.explain("cost")

// COMMAND ----------

filtered.explain("codegen")

// COMMAND ----------

filtered.explain("formatted")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Use SQL EXPLAIN format

// COMMAND ----------

strings.createOrReplaceTempView("tmp_spark_readme")

// COMMAND ----------

// MAGIC %sql
// MAGIC EXPLAIN FORMATTED SELECT * FROM tmp_spark_readme WHERE value % 'Spark'