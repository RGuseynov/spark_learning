// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC ## Example 2-1 M&M Count

// COMMAND ----------

import org.apache.spark.sql.functions._

val mnmFile = "/databricks-datasets/learning-spark-v2/mnm_dataset.csv"

// COMMAND ----------

// MAGIC %md
// MAGIC ### Read from the CSV and infer the schema

// COMMAND ----------

val mnmDF = spark
  .read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load(mnmFile)

display(mnmDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Aggregate count of all colors and groupBy state and color, orderBy descending order

// COMMAND ----------

val countMnMDF = mnmDF
  .select("State", "Color", "Count")
  .groupBy("State", "Color")
  .agg(count("Count")
  .alias("Total"))
  .orderBy(desc("Total"))

countMnMDF.show(60)
println(s"Total Rows = ${countMnMDF.count()}")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Find the aggregate count for California by filtering on State

// COMMAND ----------

val caCountMnMDF = mnmDF
  .select("State", "Color", "Count")
  .where(col("State") === "CA")
  .groupBy("State", "Color")
  .agg(count("Count").alias("Total"))
  .orderBy(desc("Total"))
   
// show the resulting aggregation for California
caCountMnMDF.show(10)
