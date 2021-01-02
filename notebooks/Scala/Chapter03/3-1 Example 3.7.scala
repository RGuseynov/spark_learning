// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Example 3.7

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{col, expr, when, concat, lit}

val jsonFile = "/databricks-datasets/learning-spark-v2/blogs.json"

val schema = StructType(Array(StructField("Id", IntegerType, false),
  StructField("First", StringType, false),
  StructField("Last", StringType, false),
  StructField("Url", StringType, false),
  StructField("Published", StringType, false),
  StructField("Hits", IntegerType, false),
  StructField("Campaigns", ArrayType(StringType), false)))

val blogsDF = spark.read.schema(schema).json(jsonFile)

blogsDF.show(false)
// print the schemas
print(blogsDF.printSchema)
print(blogsDF.schema)

// COMMAND ----------

blogsDF.createOrReplaceTempView("blogs")

// COMMAND ----------

// MAGIC %scala
// MAGIC spark.table("blogs").schema.toDDL

// COMMAND ----------

blogsDF.select(expr("Hits") * 2).show(2)

// COMMAND ----------

blogsDF.select(expr("Hits") + expr("Id")).show(false)

// COMMAND ----------

blogsDF.withColumn("Big Hitters", (expr("Hits") > 10000)).show()

// COMMAND ----------

blogsDF.withColumn("AuthorsId", (concat(expr("First"), expr("Last"), expr("Id")))).select(expr("AuthorsId")).show(4)
