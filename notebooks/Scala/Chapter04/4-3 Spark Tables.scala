// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Spark Tables
// MAGIC 
// MAGIC This notebook shows how to use Spark Catalog Interface API to query databases, tables, and columns.
// MAGIC 
// MAGIC A full list of documented methods is available [here](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Catalog)

// COMMAND ----------

val us_flights_file = "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv"

// COMMAND ----------

// MAGIC %md
// MAGIC ### Create Managed Tables

// COMMAND ----------

// Create database and managed tables
spark.sql("DROP DATABASE IF EXISTS learn_spark_db CASCADE")
spark.sql("CREATE DATABASE learn_spark_db")
spark.sql("USE learn_spark_db")
spark.sql("CREATE TABLE us_delay_flights_tbl (date STRING, delay INT, distance INT, origin STRING, destination STRING)")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Display the databases

// COMMAND ----------

display(spark.catalog.listDatabases())

// COMMAND ----------

// MAGIC %md
// MAGIC ## Read our US Flights table

// COMMAND ----------

val df = spark
  .read
  .format("csv")
  .schema("`date` STRING, `delay` INT, `distance` INT, `origin` STRING, `destination` STRING")
  .option("header", "true")
  .option("path", "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv")
  .load()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Save into our table

// COMMAND ----------

df.write.mode("overwrite").saveAsTable("us_delay_flights_tbl")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Cache the Table

// COMMAND ----------

// MAGIC %sql
// MAGIC CACHE TABLE us_delay_flights_tbl

// COMMAND ----------

// MAGIC %md
// MAGIC Check if the table is cached

// COMMAND ----------

spark.catalog.isCached("us_delay_flights_tbl")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Display tables within a Database
// MAGIC 
// MAGIC Note that the table is MANGED by Spark

// COMMAND ----------

display(spark.catalog.listTables(dbName="learn_spark_db"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Display Columns for a table

// COMMAND ----------

display(spark.catalog.listColumns("us_delay_flights_tbl"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Create Unmanaged Tables

// COMMAND ----------

// Drop the database and create unmanaged tables
spark.sql("DROP DATABASE IF EXISTS learn_spark_db CASCADE")
spark.sql("CREATE DATABASE learn_spark_db")
spark.sql("USE learn_spark_db")
spark.sql("CREATE TABLE us_delay_flights_tbl (date INT, delay INT, distance INT, origin STRING, destination STRING) USING csv OPTIONS (path '/databricks-datasets/learning-spark-v2/flights/departuredelays.csv')")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Display Tables
// MAGIC 
// MAGIC **Note**: The table type here that tableType='EXTERNAL', which indicates it's unmanaged by Spark, whereas above the tableType='MANAGED'

// COMMAND ----------

display(spark.catalog.listTables(dbName="learn_spark_db"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Display Columns for a table

// COMMAND ----------

display(spark.catalog.listColumns("us_delay_flights_tbl"))
