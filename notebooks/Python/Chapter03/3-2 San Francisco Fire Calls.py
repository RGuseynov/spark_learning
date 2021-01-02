# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # San Francisco Fire Calls
# MAGIC 
# MAGIC This notebook is the end-to-end example from Chapter 3, showing how to use DataFrame and Spark SQL for common data analytics patterns and operations on a [San Francisco Fire Department Calls ](https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3) dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC Inspect location where the SF Fire Department Fire calls data set is stored in the public dataset S3 bucket

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

# COMMAND ----------

# MAGIC %md
# MAGIC Define the location of the public dataset on the S3 bucket

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

sf_fire_file = "/databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC Inspect the data looks like before defining a schema

# COMMAND ----------

# MAGIC %fs head databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

# COMMAND ----------

# MAGIC %md
# MAGIC Define our schema as the file has 4 million records. Inferring the schema is expensive for large files.

# COMMAND ----------

fire_schema = StructType([StructField('CallNumber', IntegerType(), True),
                     StructField('UnitID', StringType(), True),
                     StructField('IncidentNumber', IntegerType(), True),
                     StructField('CallType', StringType(), True),                  
                     StructField('CallDate', StringType(), True),      
                     StructField('WatchDate', StringType(), True),
                     StructField('CallFinalDisposition', StringType(), True),
                     StructField('AvailableDtTm', StringType(), True),
                     StructField('Address', StringType(), True),       
                     StructField('City', StringType(), True),       
                     StructField('Zipcode', IntegerType(), True),       
                     StructField('Battalion', StringType(), True),                 
                     StructField('StationArea', StringType(), True),       
                     StructField('Box', StringType(), True),       
                     StructField('OriginalPriority', StringType(), True),       
                     StructField('Priority', StringType(), True),       
                     StructField('FinalPriority', IntegerType(), True),       
                     StructField('ALSUnit', BooleanType(), True),       
                     StructField('CallTypeGroup', StringType(), True),
                     StructField('NumAlarms', IntegerType(), True),
                     StructField('UnitType', StringType(), True),
                     StructField('UnitSequenceInCallDispatch', IntegerType(), True),
                     StructField('FirePreventionDistrict', StringType(), True),
                     StructField('SupervisorDistrict', StringType(), True),
                     StructField('Neighborhood', StringType(), True),
                     StructField('Location', StringType(), True),
                     StructField('RowID', StringType(), True),
                     StructField('Delay', FloatType(), True)])

# COMMAND ----------

fire_df = spark.read.csv(sf_fire_file, header=True, schema=fire_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC Cache the DataFrame since we will be performing some operations on it.

# COMMAND ----------

fire_df.cache()

# COMMAND ----------

fire_df.count()

# COMMAND ----------

fire_df.printSchema()

# COMMAND ----------

display(fire_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Filter out "Medical Incident" call types
# MAGIC 
# MAGIC Note that `filter()` and `where()` methods on the DataFrame are similar. Check relevant documentation for their respective argument types.

# COMMAND ----------

few_fire_df = (fire_df.select("IncidentNumber", "AvailableDtTm", "CallType") 
              .where(col("CallType") != "Medical Incident"))

few_fire_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-1) How many distinct types of calls were made to the Fire Department?**
# MAGIC 
# MAGIC To be sure, let's not count "null" strings in that column.

# COMMAND ----------

fire_df.select("CallType").where(col("CallType").isNotNull()).distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-2) What are distinct types of calls were made to the Fire Department?**
# MAGIC 
# MAGIC These are all the distinct type of call to the SF Fire Department

# COMMAND ----------

fire_df.select("CallType").where(col("CallType").isNotNull()).distinct().show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-3) Find out all response or delayed times greater than 5 mins?**
# MAGIC 
# MAGIC 1. Rename the column Delay - > ReponseDelayedinMins
# MAGIC 2. Returns a new DataFrame
# MAGIC 3. Find out all calls where the response time to the fire site was delayed for more than 5 mins

# COMMAND ----------

new_fire_df = fire_df.withColumnRenamed("Delay", "ResponseDelayedinMins")
new_fire_df.select("ResponseDelayedinMins").where(col("ResponseDelayedinMins") > 5).show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's do some ETL:
# MAGIC 
# MAGIC 1. Transform the string dates to Spark Timestamp data type so we can make some time-based queries later
# MAGIC 2. Returns a transformed query
# MAGIC 3. Cache the new DataFrame

# COMMAND ----------

fire_ts_df = (new_fire_df
              .withColumn("IncidentDate", to_timestamp(col("CallDate"), "MM/dd/yyyy")).drop("CallDate") 
              .withColumn("OnWatchDate",   to_timestamp(col("WatchDate"), "MM/dd/yyyy")).drop("WatchDate")
              .withColumn("AvailableDtTS", to_timestamp(col("AvailableDtTm"), "MM/dd/yyyy hh:mm:ss a")).drop("AvailableDtTm"))          

# COMMAND ----------

fire_ts_df.cache()
fire_ts_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC Check the transformed columns with Spark Timestamp type

# COMMAND ----------

fire_ts_df.select("IncidentDate", "OnWatchDate", "AvailableDtTS").show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-4) What were the most common call types?**
# MAGIC 
# MAGIC List them in descending order

# COMMAND ----------

(fire_ts_df
 .select("CallType").where(col("CallType").isNotNull())
 .groupBy("CallType")
 .count()
 .orderBy("count", ascending=False)
 .show(n=10, truncate=False))

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-4a) What zip codes accounted for most common calls?**
# MAGIC 
# MAGIC Let's investigate what zip codes in San Francisco accounted for most fire calls and what type where they.
# MAGIC 
# MAGIC 1. Filter out by CallType
# MAGIC 2. Group them by CallType and Zip code
# MAGIC 3. Count them and display them in descending order
# MAGIC 
# MAGIC It seems like the most common calls were all related to Medical Incident, and the two zip codes are 94102 and 94103.

# COMMAND ----------

(fire_ts_df
 .select("CallType", "ZipCode")
 .where(col("CallType").isNotNull())
 .groupBy("CallType", "Zipcode")
 .count()
 .orderBy("count", ascending=False)
 .show(10, truncate=False))

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-4b) What San Francisco neighborhoods are in the zip codes 94102 and 94103**
# MAGIC 
# MAGIC Let's find out the neighborhoods associated with these two zip codes. In all likelihood, these are some of the contested 
# MAGIC neighborhood with high reported crimes.

# COMMAND ----------

fire_ts_df.select("Neighborhood", "Zipcode").where((col("Zipcode") == 94102) | (col("Zipcode") == 94103)).distinct().show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Q-5) What was the sum of all calls, average, min and max of the response times for calls?**
# MAGIC 
# MAGIC Let's use the built-in Spark SQL functions to compute the sum, avg, min, and max of few columns:
# MAGIC 
# MAGIC * Number of Total Alarms
# MAGIC * What were the min and max the delay in response time before the Fire Dept arrived at the scene of the call

# COMMAND ----------

fire_ts_df.select(sum("NumAlarms"), avg("ResponseDelayedinMins"), min("ResponseDelayedinMins"), max("ResponseDelayedinMins")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ** Q-6a) How many distinct years of data is in the CSV file?**
# MAGIC 
# MAGIC We can use the `year()` SQL Spark function off the Timestamp column data type IncidentDate.
# MAGIC 
# MAGIC In all, we have fire calls from years 2000-2018

# COMMAND ----------

fire_ts_df.select(year('IncidentDate')).distinct().orderBy(year('IncidentDate')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ** Q-6b) What week of the year in 2018 had the most fire calls?**
# MAGIC 
# MAGIC **Note**: Week 1 is the New Years' week and week 25 is the July 4 the week. Loads of fireworks, so it makes sense the higher number of calls.

# COMMAND ----------

fire_ts_df.filter(year('IncidentDate') == 2018).groupBy(weekofyear('IncidentDate')).count().orderBy('count', ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ** Q-7) What neighborhoods in San Francisco had the worst response time in 2018?**
# MAGIC 
# MAGIC It appears that if you living in Presidio Heights, the Fire Dept arrived in less than 3 mins, while Mission Bay took more than 6 mins.

# COMMAND ----------

fire_ts_df.select("Neighborhood", "ResponseDelayedinMins").filter(year("IncidentDate") == 2018).show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ** Q-8a) How can we use Parquet files or SQL table to store data and read it back?**

# COMMAND ----------

fire_ts_df.write.format("parquet").mode("overwrite").save("/tmp/fireServiceParquet/")

# COMMAND ----------

# MAGIC %fs ls /tmp/fireServiceParquet/

# COMMAND ----------

# MAGIC %md
# MAGIC ** Q-8b) How can we use Parquet SQL table to store data and read it back?**

# COMMAND ----------

fire_ts_df.write.format("parquet").mode("overwrite").saveAsTable("FireServiceCalls")

# COMMAND ----------

# MAGIC %sql
# MAGIC CACHE TABLE FireServiceCalls

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM FireServiceCalls LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ** Q-8c) How can read data from Parquet file?**
# MAGIC 
# MAGIC Note we don't have to specify the schema here since it's stored as part of the Parquet metadata

# COMMAND ----------

file_parquet_df = spark.read.format("parquet").load("/tmp/fireServiceParquet/")

# COMMAND ----------

display(file_parquet_df.limit(10))
