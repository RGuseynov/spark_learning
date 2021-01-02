// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # San Francisco Fire Calls
// MAGIC 
// MAGIC This notebook is the end-to-end example from Chapter 3, showing how to use DataFrame and Spark SQL for common data analytics patterns and operations on a [San Francisco Fire Department Calls ](https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3) dataset.

// COMMAND ----------

// MAGIC %md
// MAGIC Inspect location where the SF Fire Department Fire calls data set is stored in the public dataset S3 bucket

// COMMAND ----------

// MAGIC %fs ls /databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

// COMMAND ----------

// MAGIC %md
// MAGIC Define the location of the public dataset on the S3 bucket

// COMMAND ----------

import org.apache.spark.sql.types._ 
import org.apache.spark.sql.functions._ 

val sfFireFile = "/databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv"

// COMMAND ----------

// MAGIC %md
// MAGIC Inspect the data looks like before defining a schema

// COMMAND ----------

// MAGIC %fs head databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

// COMMAND ----------

// MAGIC %md
// MAGIC Define our schema as the file has 4 million records. Inferring the schema is expensive for large files.

// COMMAND ----------

val fireSchema = StructType(Array(StructField("CallNumber", IntegerType, true),
  StructField("UnitID", StringType, true),
  StructField("IncidentNumber", IntegerType, true),
  StructField("CallType", StringType, true),                  
  StructField("CallDate", StringType, true),      
  StructField("WatchDate", StringType, true),
  StructField("CallFinalDisposition", StringType, true),
  StructField("AvailableDtTm", StringType, true),
  StructField("Address", StringType, true),       
  StructField("City", StringType, true),       
  StructField("Zipcode", IntegerType, true),       
  StructField("Battalion", StringType, true),                 
  StructField("StationArea", StringType, true),       
  StructField("Box", StringType, true),       
  StructField("OriginalPriority", StringType, true),       
  StructField("Priority", StringType, true),       
  StructField("FinalPriority", IntegerType, true),       
  StructField("ALSUnit", BooleanType, true),       
  StructField("CallTypeGroup", StringType, true),
  StructField("NumAlarms", IntegerType, true),
  StructField("UnitType", StringType, true),
  StructField("UnitSequenceInCallDispatch", IntegerType, true),
  StructField("FirePreventionDistrict", StringType, true),
  StructField("SupervisorDistrict", StringType, true),
  StructField("Neighborhood", StringType, true),
  StructField("Location", StringType, true),
  StructField("RowID", StringType, true),
  StructField("Delay", FloatType, true)))

// COMMAND ----------

val fireDF = spark
  .read
  .schema(fireSchema)
  .option("header", "true")
  .csv(sfFireFile)

// COMMAND ----------

// MAGIC %md
// MAGIC Cache the DataFrame since we will be performing some operations on it.

// COMMAND ----------

fireDF.cache()

// COMMAND ----------

fireDF.count()

// COMMAND ----------

fireDF.printSchema()

// COMMAND ----------

display(fireDF.limit(5))

// COMMAND ----------

// MAGIC %md
// MAGIC Filter out "Medical Incident" call types
// MAGIC 
// MAGIC Note that `filter()` and `where()` methods on the DataFrame are similar. Check relevant documentation for their respective argument types.

// COMMAND ----------

val fewFireDF = fireDF
  .select("IncidentNumber", "AvailableDtTm", "CallType") 
  .where($"CallType" =!= "Medical Incident")

fewFireDF.show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-1) How many distinct types of calls were made to the Fire Department?**
// MAGIC 
// MAGIC To be sure, let's not count "null" strings in that column.

// COMMAND ----------

fireDF.select("CallType").where(col("CallType").isNotNull).distinct().count()

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-2) What are distinct types of calls were made to the Fire Department?**
// MAGIC 
// MAGIC These are all the distinct type of call to the SF Fire Department

// COMMAND ----------

fireDF.select("CallType").where(col("CallType").isNotNull).distinct().show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-3) Find out all response or delayed times greater than 5 mins?**
// MAGIC 
// MAGIC 1. Rename the column Delay - > ReponseDelayedinMins
// MAGIC 2. Returns a new DataFrame
// MAGIC 3. Find out all calls where the response time to the fire site was delayed for more than 5 mins

// COMMAND ----------

val newFireDF = fireDF.withColumnRenamed("Delay", "ResponseDelayedinMins")
newFireDF.select("ResponseDelayedinMins").where($"ResponseDelayedinMins" > 5).show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC Let's do some ETL:
// MAGIC 
// MAGIC 1. Transform the string dates to Spark Timestamp data type so we can make some time-based queries later
// MAGIC 2. Returns a transformed query
// MAGIC 3. Cache the new DataFrame

// COMMAND ----------

val fireTSDF = newFireDF
  .withColumn("IncidentDate", to_timestamp(col("CallDate"), "MM/dd/yyyy")).drop("CallDate") 
  .withColumn("OnWatchDate", to_timestamp(col("WatchDate"), "MM/dd/yyyy")).drop("WatchDate") 
  .withColumn("AvailableDtTS", to_timestamp(col("AvailableDtTm"), "MM/dd/yyyy hh:mm:ss a")).drop("AvailableDtTm")

// COMMAND ----------

fireTSDF.cache()
fireTSDF.columns

// COMMAND ----------

// MAGIC %md
// MAGIC Check the transformed columns with Spark Timestamp type

// COMMAND ----------

fireTSDF.select("IncidentDate", "OnWatchDate", "AvailableDtTS").show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-4) What were the most common call types?**
// MAGIC 
// MAGIC List them in descending order

// COMMAND ----------

fireTSDF
  .select("CallType")
  .where(col("CallType").isNotNull)
  .groupBy("CallType")
  .count()
  .orderBy(desc("count"))
  .show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-4a) What zip codes accounted for most common calls?**
// MAGIC 
// MAGIC Let's investigate what zip codes in San Francisco accounted for most fire calls and what type where they.
// MAGIC 
// MAGIC 1. Filter out by CallType
// MAGIC 2. Group them by CallType and Zip code
// MAGIC 3. Count them and display them in descending order
// MAGIC 
// MAGIC It seems like the most common calls were all related to Medical Incident, and the two zip codes are 94102 and 94103.

// COMMAND ----------

fireTSDF
  .select("CallType", "ZipCode")
  .where(col("CallType").isNotNull)
  .groupBy("CallType", "Zipcode")
  .count()
  .orderBy(desc("count"))
  .show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-4b) What San Francisco neighborhoods are in the zip codes 94102 and 94103**
// MAGIC 
// MAGIC Let's find out the neighborhoods associated with these two zip codes. In all likelihood, these are some of the contested 
// MAGIC neighborhood with high reported crimes.

// COMMAND ----------

fireTSDF.select("Neighborhood", "Zipcode").where((col("Zipcode") === 94102) || (col("Zipcode") === 94103)).distinct().show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC **Q-5) What was the sum of all calls, average, min and max of the response times for calls?**
// MAGIC 
// MAGIC Let's use the built-in Spark SQL functions to compute the sum, avg, min, and max of few columns:
// MAGIC 
// MAGIC * Number of Total Alarms
// MAGIC * What were the min and max the delay in response time before the Fire Dept arrived at the scene of the call

// COMMAND ----------

fireTSDF.select(sum("NumAlarms"), avg("ResponseDelayedinMins"), min("ResponseDelayedinMins"), max("ResponseDelayedinMins")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ** Q-6a) How many distinct years of data is in the CSV file?**
// MAGIC 
// MAGIC We can use the `year()` SQL Spark function off the Timestamp column data type IncidentDate.
// MAGIC 
// MAGIC In all, we have fire calls from years 2000-2018

// COMMAND ----------

fireTSDF.select(year($"IncidentDate")).distinct().orderBy(year($"IncidentDate")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ** Q-6b) What week of the year in 2018 had the most fire calls?**
// MAGIC 
// MAGIC **Note**: Week 1 is the New Years' week and week 25 is the July 4 the week. Loads of fireworks, so it makes sense the higher number of calls.

// COMMAND ----------

fireTSDF.filter(year($"IncidentDate") === 2018).groupBy(weekofyear($"IncidentDate")).count().orderBy(desc("count")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ** Q-7) What neighborhoods in San Francisco had the worst response time in 2018?**
// MAGIC 
// MAGIC It appears that if you living in Presidio Heights, the Fire Dept arrived in less than 3 mins, while Mission Bay took more than 6 mins.

// COMMAND ----------

fireTSDF.select("Neighborhood", "ResponseDelayedinMins").filter(year($"IncidentDate") === 2018).show(10, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ** Q-8a) How can we use Parquet files or SQL table to store data and read it back?**

// COMMAND ----------

fireTSDF.write.format("parquet").mode("overwrite").save("/tmp/fireServiceParquet/")

// COMMAND ----------

// MAGIC %fs ls /tmp/fireServiceParquet/

// COMMAND ----------

// MAGIC %md
// MAGIC ** Q-8b) How can we use Parquet SQL table to store data and read it back?**

// COMMAND ----------

fireTSDF.write.format("parquet").mode("overwrite").saveAsTable("FireServiceCalls")

// COMMAND ----------

// MAGIC %sql
// MAGIC CACHE TABLE FireServiceCalls

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM FireServiceCalls LIMIT 10

// COMMAND ----------

// MAGIC %md
// MAGIC ** Q-8c) How can read data from Parquet file?**
// MAGIC 
// MAGIC Note we don't have to specify the schema here since it's stored as part of the Parquet metadata

// COMMAND ----------

val fileParquetDF = spark.read.format("parquet").load("/tmp/fireServiceParquet/")

// COMMAND ----------

display(fileParquetDF.limit(10))
