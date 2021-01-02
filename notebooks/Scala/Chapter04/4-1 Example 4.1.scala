// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Example 4.1
// MAGIC 
// MAGIC This notebook shows Example 4.1 from the book showing how to use SQL on a US Flights Dataset dataset.

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

// COMMAND ----------

// MAGIC %md
// MAGIC Define a UDF to convert the date format into a legible format.
// MAGIC 
// MAGIC *Note*: the date is a string with year missing, so it might be difficult to do any queries using SQL `year()` function

// COMMAND ----------

def toDateFormatUDF(dStr:String) : String  = {
  return s"${dStr(0)}${dStr(1)}${'/'}${dStr(2)}${dStr(3)}${' '}${dStr(4)}${dStr(5)}${':'}${dStr(6)}${dStr(7)}"
}

// test  it
toDateFormatUDF("02190925")

// COMMAND ----------

// MAGIC %md
// MAGIC Register the UDF

// COMMAND ----------

spark.udf.register("toDateFormatUDF", toDateFormatUDF(_:String):String)

// COMMAND ----------

// MAGIC %md
// MAGIC Read our US departure flight data

// COMMAND ----------

val df = spark
  .read
  .format("csv")
  .schema("date STRING, delay INT, distance INT, origin STRING, destination STRING")
  .option("header", "true")
  .option("path", "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv")
  .load()

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC Test our UDF

// COMMAND ----------

df.selectExpr("toDateFormatUDF(date) as data_format").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC Create a temporary view to which we can issue SQL queries

// COMMAND ----------

df.createOrReplaceTempView("us_delay_flights_tbl")

// COMMAND ----------

// MAGIC %md
// MAGIC Cache Table so queries are expedient

// COMMAND ----------

// MAGIC %sql
// MAGIC CACHE TABLE us_delay_flights_tbl

// COMMAND ----------

// MAGIC %md
// MAGIC Convert all `date` to `date_fm` so it's more eligible
// MAGIC 
// MAGIC Note: we are using UDF to convert it on the fly. 

// COMMAND ----------

spark.sql("SELECT *, date, toDateFormatUDF(date) AS date_fm FROM us_delay_flights_tbl").show(10, false)

// COMMAND ----------

spark.sql("SELECT COUNT(*) FROM us_delay_flights_tbl").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Query 1:
// MAGIC 
// MAGIC  Find out all flights whose distance between origin and destination is greater than 1000 

// COMMAND ----------

spark.sql("SELECT distance, origin, destination FROM us_delay_flights_tbl WHERE distance > 1000 ORDER BY distance DESC").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC A DataFrame equivalent query

// COMMAND ----------

df.select("distance", "origin", "destination").where(col("distance") > 1000).orderBy(desc("distance")).show(10, false)

// COMMAND ----------

df.select("distance", "origin", "destination").where($"distance" > 1000).orderBy(desc("distance")).show(10, false)

// COMMAND ----------

df.select("distance", "origin", "destination").where($"distance" > 1000).orderBy($"distance".desc).show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Query 2:
// MAGIC 
// MAGIC  Find out all flights with 2 hour delays between San Francisco and Chicago  

// COMMAND ----------

spark.sql("""
SELECT date, delay, origin, destination 
FROM us_delay_flights_tbl 
WHERE delay > 120 AND ORIGIN = 'SFO' AND DESTINATION = 'ORD' 
ORDER by delay DESC
""").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Query 3:
// MAGIC 
// MAGIC A more complicated query in SQL, let's label all US flights originating from airports with _high_, _medium_, _low_, _no delays_, regardless of destinations.

// COMMAND ----------

spark.sql("""SELECT delay, origin, destination,
              CASE
                  WHEN delay > 360 THEN 'Very Long Delays'
                  WHEN delay > 120 AND delay < 360 THEN  'Long Delays '
                  WHEN delay > 60 AND delay < 120 THEN  'Short Delays'
                  WHEN delay > 0 and delay < 60  THEN   'Tolerable Delays'
                  WHEN delay = 0 THEN 'No Delays'
                  ELSE 'No Delays'
               END AS Flight_Delays
               FROM us_delay_flights_tbl
               ORDER BY origin, delay DESC""").show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Some Side Queries

// COMMAND ----------

val df1 =  spark.sql("SELECT date, delay, origin, destination FROM us_delay_flights_tbl WHERE origin = 'SFO'")

// COMMAND ----------

df1.createOrReplaceGlobalTempView("us_origin_airport_SFO_tmp_view")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM global_temp.us_origin_airport_SFO_tmp_view

// COMMAND ----------

// MAGIC %sql
// MAGIC DROP VIEW IF EXISTS global_temp.us_origin_airport_JFK_tmp_view

// COMMAND ----------

val df2 = spark.sql("SELECT date, delay, origin, destination from us_delay_flights_tbl WHERE origin = 'JFK'")

// COMMAND ----------

df2.createOrReplaceTempView("us_origin_airport_JFK_tmp_view")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM us_origin_airport_JFK_tmp_view

// COMMAND ----------

// MAGIC %sql
// MAGIC DROP VIEW IF EXISTS us_origin_airport_JFK_tmp_view

// COMMAND ----------

display(spark.catalog.listTables(dbName="global_temp"))
