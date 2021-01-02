// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Spark Data Sources
// MAGIC 
// MAGIC This notebook shows how to use Spark Data Sources Interface API to read file formats:
// MAGIC  * Parquet
// MAGIC  * JSON
// MAGIC  * CSV
// MAGIC  * Avro
// MAGIC  * ORC
// MAGIC  * Image
// MAGIC  * Binary
// MAGIC 
// MAGIC A full list of DataSource methods is available [here](https://docs.databricks.com/spark/latest/data-sources/index.html#id1)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Define paths for the various data sources

// COMMAND ----------

val parquetFile = "/databricks-datasets/learning-spark-v2/flights/summary-data/parquet/2010-summary.parquet"
val jsonFile = "/databricks-datasets/learning-spark-v2/flights/summary-data/json/*"
val csvFile = "/databricks-datasets/learning-spark-v2/flights/summary-data/csv/*"
val orcFile = "/databricks-datasets/learning-spark-v2/flights/summary-data/orc/*"
val avroFile = "/databricks-datasets/learning-spark-v2/flights/summary-data/avro/*"
val schema = "DEST_COUNTRY_NAME STRING, ORIGIN_COUNTRY_NAME STRING, count INT"

// COMMAND ----------

// MAGIC %md
// MAGIC ## Parquet Data Source

// COMMAND ----------

val df = spark
  .read
  .format("parquet")
  .option("path", parquetFile)
  .load()

// COMMAND ----------

// MAGIC %md
// MAGIC Another way to read this same data using a variation of this API

// COMMAND ----------

val df2 = spark.read.parquet(parquetFile)

// COMMAND ----------

df.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use SQL
// MAGIC 
// MAGIC This will create an _unmanaged_ temporary view

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW us_delay_flights_tbl
// MAGIC     USING parquet
// MAGIC     OPTIONS (
// MAGIC       path "/databricks-datasets/definitive-guide/data/flight-data/parquet/2010-summary.parquet"
// MAGIC     )

// COMMAND ----------

// MAGIC %md
// MAGIC Use SQL to query the table
// MAGIC 
// MAGIC The outcome should be the same as one read into the DataFrame above

// COMMAND ----------

spark.sql("SELECT * FROM us_delay_flights_tbl").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## JSON Data Source

// COMMAND ----------

val df = spark
  .read
  .format("json")
  .option("path", jsonFile)
  .load()

// COMMAND ----------

df.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use SQL
// MAGIC 
// MAGIC This will create an _unmanaged_ temporary view

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW us_delay_flights_tbl
// MAGIC     USING json
// MAGIC     OPTIONS (
// MAGIC       path "/databricks-datasets/learning-spark-v2/flights/summary-data/json/*"
// MAGIC     )

// COMMAND ----------

// MAGIC %md
// MAGIC Use SQL to query the table
// MAGIC 
// MAGIC The outcome should be the same as one read into the DataFrame above

// COMMAND ----------

spark.sql("SELECT * FROM us_delay_flights_tbl").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## CSV Data Source

// COMMAND ----------

val df  = spark
  .read
  .format("csv")
  .option("header", "true")
  .schema(schema)
  .option("mode", "FAILFAST")  // exit if any errors
  .option("nullValue", "")	  // replace any null data field with “”
  .option("path", csvFile)
  .load()

// COMMAND ----------

df.show(10, false)

// COMMAND ----------

(df.write.format("parquet")
  .mode("overwrite")
  .option("path", "/tmp/data/parquet/df_parquet")
  .option("compression", "snappy")
  .save())

// COMMAND ----------

// MAGIC %fs ls /tmp/data/parquet/df_parquet

// COMMAND ----------

val df2 = spark
  .read
  .option("header", "true")
  .option("mode", "FAILFAST") // exit if any errors
  .option("nullValue", "")
  .schema(schema)
  .csv(csvFile)

// COMMAND ----------

df2.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use SQL
// MAGIC 
// MAGIC This will create an _unmanaged_ temporary view

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW us_delay_flights_tbl
// MAGIC     USING csv
// MAGIC     OPTIONS (
// MAGIC       path "/databricks-datasets/learning-spark-v2/flights/summary-data/csv/*",
// MAGIC       header "true",
// MAGIC       inferSchema "true",
// MAGIC       mode "FAILFAST"
// MAGIC     )

// COMMAND ----------

// MAGIC %md
// MAGIC Use SQL to query the table
// MAGIC 
// MAGIC The outcome should be the same as one read into the DataFrame above

// COMMAND ----------

spark.sql("SELECT * FROM us_delay_flights_tbl").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## ORC Data Source

// COMMAND ----------

val df = spark.read
  .format("orc")
  .option("path", orcFile)
  .load()

// COMMAND ----------

df.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use SQL
// MAGIC 
// MAGIC This will create an _unmanaged_ temporary view

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW us_delay_flights_tbl
// MAGIC     USING orc
// MAGIC     OPTIONS (
// MAGIC       path "/databricks-datasets/learning-spark-v2/flights/summary-data/orc/*"
// MAGIC     )

// COMMAND ----------

// MAGIC %md
// MAGIC Use SQL to query the table
// MAGIC 
// MAGIC The outcome should be the same as one read into the DataFrame above

// COMMAND ----------

spark.sql("SELECT * FROM us_delay_flights_tbl").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Avro Data Source

// COMMAND ----------

val df = spark.read
  .format("avro")
  .option("path", avroFile)
  .load()

// COMMAND ----------

df.show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use SQL
// MAGIC 
// MAGIC This will create an _unmanaged_ temporary view

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW us_delay_flights_tbl
// MAGIC     USING avro
// MAGIC     OPTIONS (
// MAGIC       path "/databricks-datasets/learning-spark-v2/flights/summary-data/avro/*"
// MAGIC     )

// COMMAND ----------

// MAGIC %md
// MAGIC Use SQL to query the table
// MAGIC 
// MAGIC The outcome should be the same as the one read into the DataFrame above

// COMMAND ----------

spark.sql("SELECT * FROM us_delay_flights_tbl").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Image

// COMMAND ----------

import org.apache.spark.ml.source.image

val imageDir = "/databricks-datasets/cctvVideos/train_images/"
val imagesDF = spark.read.format("image").load(imageDir)

imagesDF.printSchema
imagesDF.select("image.height", "image.width", "image.nChannels", "image.mode", "label").show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Binary

// COMMAND ----------

val path = "/databricks-datasets/learning-spark-v2/cctvVideos/train_images/"
val binaryFilesDF = spark.read.format("binaryFile")
  .option("pathGlobFilter", "*.jpg")
  .load(path)

binaryFilesDF.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC To ignore any partitioning data discovery in a directory, you can set the `recursiveFileLookup` to `true`.

// COMMAND ----------

val binaryFilesDF = spark.read.format("binaryFile")
  .option("pathGlobFilter", "*.jpg")
  .option("recursiveFileLookup", "true")
  .load(path)
binaryFilesDF.show(5)
