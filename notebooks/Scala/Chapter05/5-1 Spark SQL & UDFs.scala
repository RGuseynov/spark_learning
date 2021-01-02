// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC ## Chapter 5: Spark SQL and DataFrames: Interacting with External Data Sources
// MAGIC This notebook contains for code samples for *Chapter 5: Spark SQL and DataFrames: Interacting with External Data Sources*.

// COMMAND ----------

// MAGIC %md
// MAGIC ### User Defined Functions
// MAGIC While Apache Spark has a plethora of functions, the flexibility of Spark allows for data engineers and data scientists to define their own functions (i.e., user-defined functions or UDFs).  

// COMMAND ----------

// Create cubed function
val cubed = (s: Long) => {
  s * s * s
}

// Register UDF
spark.udf.register("cubed", cubed)

// Create temporary view
spark.range(1, 9).createOrReplaceTempView("udf_test")

// COMMAND ----------

spark.sql("SELECT id, cubed(id) AS id_cubed FROM udf_test").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Speeding up and Distributing PySpark UDFs with Pandas UDFs
// MAGIC One of the previous prevailing issues with using PySpark UDFs was that it had slower performance than Scala UDFs.  This was because the PySpark UDFs required data movement between the JVM and Python working which was quite expensive.   To resolve this problem, pandas UDFs (also known as vectorized UDFs) were introduced as part of Apache Spark 2.3. It is a UDF that uses Apache Arrow to transfer data and utilizes pandas to work with the data. You simplify define a pandas UDF using the keyword pandas_udf as the decorator or to wrap the function itself.   Once the data is in Apache Arrow format, there is no longer the need to serialize/pickle the data as it is already in a format consumable by the Python process.  Instead of operating on individual inputs row-by-row, you are operating on a pandas series or dataframe (i.e. vectorized execution).

// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC # Import various pyspark SQL functions including pandas_udf
// MAGIC from pyspark.sql.functions import col, pandas_udf
// MAGIC from pyspark.sql.types import LongType
// MAGIC 
// MAGIC # Declare the cubed function 
// MAGIC def cubed(a: pd.Series) -> pd.Series:
// MAGIC     return a * a * a
// MAGIC 
// MAGIC # Create the pandas UDF for the cubed function 
// MAGIC cubed_udf = pandas_udf(cubed, returnType=LongType())

// COMMAND ----------

// MAGIC %md
// MAGIC ### Using pandas dataframe

// COMMAND ----------

// MAGIC %python
// MAGIC # Create a Pandas series
// MAGIC x = pd.Series([1, 2, 3])
// MAGIC 
// MAGIC # The function for a pandas_udf executed with local Pandas data
// MAGIC print(cubed(x))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Using Spark DataFrame

// COMMAND ----------

// MAGIC %python
// MAGIC # Create a Spark DataFrame
// MAGIC df = spark.range(1, 4)
// MAGIC 
// MAGIC # Execute function as a Spark vectorized UDF
// MAGIC df.select("id", cubed_udf(col("id"))).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Higher Order Functions in DataFrames and Spark SQL
// MAGIC 
// MAGIC Because complex data types are an amalgamation of simple data types, it is tempting to manipulate complex data types directly. As noted in the post *Introducing New Built-in and Higher-Order Functions for Complex Data Types in Apache Spark 2.4* there are typically two solutions for the manipulation of complex data types.
// MAGIC 1. Exploding the nested structure into individual rows, applying some function, and then re-creating the nested structure as noted in the code snippet below (see Option 1)  
// MAGIC 1. Building a User Defined Function (UDF)

// COMMAND ----------

// Create an array dataset
val arrayData = Seq(
  Row(1, List(1, 2, 3)),
  Row(2, List(2, 3, 4)),
  Row(3, List(3, 4, 5))
)

// Create schema
import org.apache.spark.sql._
import org.apache.spark.sql.types._

val arraySchema = new StructType()
  .add("id", IntegerType)
  .add("values", ArrayType(IntegerType))

// Create DataFrame
val df = spark.createDataFrame(spark.sparkContext.parallelize(arrayData), arraySchema)
df.createOrReplaceTempView("table")
df.printSchema()
df.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Option 1: Explode and Collect
// MAGIC In this nested SQL statement, we first `explode(values)` which creates a new row (with the id) for each element (`value`) within values.  

// COMMAND ----------

spark.sql("""
SELECT id, collect_list(value + 1) AS newValues
  FROM  (SELECT id, explode(values) AS value
        FROM table) x
 GROUP BY id
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Option 2: User Defined Function
// MAGIC To perform the same task (adding a value of 1 to each element in `values`), we can also create a user defined function (UDF) that uses map to iterate through each element (`value`) to perform the addition operation.

// COMMAND ----------

// Create UDF
def addOne(values: Seq[Int]): Seq[Int] = {
    values.map(value => value + 1)
}

// Register UDF
val plusOneInt = spark.udf.register("plusOneInt", addOne(_: Seq[Int]): Seq[Int])

// Query data
spark.sql("SELECT id, plusOneInt(values) AS values FROM table").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Higher-Order Functions
// MAGIC In addition to the previously noted built-in functions, there are high-order functions that take anonymous lambda functions as arguments. 

// COMMAND ----------

// In Scala
// Create DataFrame with two rows of two arrays (tempc1, tempc2)
val t1 = Array(35, 36, 32, 30, 40, 42, 38)
val t2 = Array(31, 32, 34, 55, 56)
val tC = Seq(t1, t2).toDF("celsius")
tC.createOrReplaceTempView("tC")

// Show the DataFrame
tC.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Transform
// MAGIC 
// MAGIC `transform(array<T>, function<T, U>): array<U>`
// MAGIC 
// MAGIC The transform function produces an array by applying a function to each element of an input array (similar to a map function).

// COMMAND ----------

// Calculate Fahrenheit from Celsius for an array of temperatures
spark.sql("""SELECT celsius, transform(celsius, t -> ((t * 9) div 5) + 32) as fahrenheit FROM tC""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Filter
// MAGIC 
// MAGIC `filter(array<T>, function<T, Boolean>): array<T>`
// MAGIC 
// MAGIC The filter function produces an array where the boolean function is true.

// COMMAND ----------

// Filter temperatures > 38C for array of temperatures
spark.sql("""SELECT celsius, filter(celsius, t -> t > 38) as high FROM tC""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Exists
// MAGIC 
// MAGIC `exists(array<T>, function<T, V, Boolean>): Boolean`
// MAGIC 
// MAGIC The exists function returns true if the boolean function holds for any element in the input array.

// COMMAND ----------

// Is there a temperature of 38C in the array of temperatures
spark.sql("""
SELECT celsius, exists(celsius, t -> t = 38) as threshold
FROM tC
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Reduce
// MAGIC 
// MAGIC `reduce(array<T>, B, function<B, T, B>, function<B, R>)`
// MAGIC 
// MAGIC The reduce function reduces the elements of the array to a single value  by merging the elements into a buffer B using function<B, T, B> and by applying a finishing function<B, R> on the final buffer.

// COMMAND ----------

// Calculate average temperature and convert to F
spark.sql("""
SELECT celsius, 
       reduce(
          celsius, 
          0, 
          (t, acc) -> t + acc, 
          acc -> (acc div size(celsius) * 9 div 5) + 32
        ) as avgFahrenheit 
  FROM tC
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## DataFrames and Spark SQL Common Relational Operators
// MAGIC 
// MAGIC The power of Spark SQL is that it contains many DataFrame Operations (also known as Untyped Dataset Operations). 
// MAGIC 
// MAGIC For the full list, refer to [Spark SQL, Built-in Functions](https://spark.apache.org/docs/latest/api/sql/index.html).
// MAGIC 
// MAGIC In the next section, we will focus on the following common relational operators:
// MAGIC * Unions and Joins
// MAGIC * Windowing
// MAGIC * Modifications

// COMMAND ----------

import org.apache.spark.sql.functions._

// Set File Paths
val delaysPath = "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv"
val airportsPath = "/databricks-datasets/learning-spark-v2/flights/airport-codes-na.txt"

// Obtain airports dataset
val airports = spark
  .read
  .options(
    Map(
      "header" -> "true", 
      "inferSchema" ->  "true", 
      "sep" -> "\t"))
  .csv(airportsPath)

airports.createOrReplaceTempView("airports_na")

// Obtain departure Delays data
val delays = spark
  .read
  .option("header","true")
  .csv(delaysPath)
  .withColumn("delay", expr("CAST(delay as INT) as delay"))
  .withColumn("distance", expr("CAST(distance as INT) as distance"))

delays.createOrReplaceTempView("departureDelays")

// Create temporary small table
val foo = delays
  .filter(
    expr("""
         origin == 'SEA' AND 
         destination == 'SFO' AND 
         date like '01010%' AND delay > 0
         """))

foo.createOrReplaceTempView("foo")

// COMMAND ----------

spark.sql("SELECT * FROM airports_na LIMIT 10").show()

// COMMAND ----------

spark.sql("SELECT * FROM departureDelays LIMIT 10").show()

// COMMAND ----------

spark.sql("SELECT * FROM foo LIMIT 10").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Unions

// COMMAND ----------

// Union two tables
val bar = delays.union(foo)
bar.createOrReplaceTempView("bar")
bar.filter(expr("origin == 'SEA' AND destination == 'SFO' AND date LIKE '01010%' AND delay > 0")).show()

// COMMAND ----------

spark.sql("""
SELECT * 
FROM bar 
WHERE origin = 'SEA' 
   AND destination = 'SFO' 
   AND date LIKE '01010%' 
   AND delay > 0
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Joins
// MAGIC By default, it is an `inner join`.  There are also the options: `inner, cross, outer, full, full_outer, left, left_outer, right, right_outer, left_semi, and left_anti`.
// MAGIC 
// MAGIC More info available at:
// MAGIC * [PySpark Join](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=join)

// COMMAND ----------

// Join Departure Delays data (foo) with flight info
foo.join(
  airports.as('air), 
  $"air.IATA" === $"origin"
).select("City", "State", "date", "delay", "distance", "destination").show()

// COMMAND ----------

spark.sql("""
SELECT a.City, a.State, f.date, f.delay, f.distance, f.destination 
  FROM foo f
  JOIN airports_na a
    ON a.IATA = f.origin
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Windowing Functions
// MAGIC 
// MAGIC Great reference: [Introduction Windowing Functions in Spark SQL](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)
// MAGIC 
// MAGIC > At its core, a window function calculates a return value for every input row of a table based on a group of rows, called the Frame. Every input row can have a unique frame associated with it. This characteristic of window functions makes them more powerful than other functions and allows users to express various data processing tasks that are hard (if not impossible) to be expressed without window functions in a concise way.

// COMMAND ----------

spark.sql("DROP TABLE IF EXISTS departureDelaysWindow")
spark.sql("""
CREATE TABLE departureDelaysWindow AS
SELECT origin, destination, sum(delay) as TotalDelays 
  FROM departureDelays 
 WHERE origin IN ('SEA', 'SFO', 'JFK') 
   AND destination IN ('SEA', 'SFO', 'JFK', 'DEN', 'ORD', 'LAX', 'ATL') 
 GROUP BY origin, destination
""")

spark.sql("""SELECT * FROM departureDelaysWindow""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC What are the top three total delays destinations by origin city of SEA, SFO, and JFK?

// COMMAND ----------

spark.sql("""
SELECT origin, destination, sum(TotalDelays) as TotalDelays
 FROM departureDelaysWindow
WHERE origin = 'SEA'
GROUP BY origin, destination
ORDER BY sum(TotalDelays) DESC
LIMIT 3
""").show()

// COMMAND ----------

spark.sql("""
SELECT origin, destination, TotalDelays, rank 
  FROM ( 
     SELECT origin, destination, TotalDelays, dense_rank() 
       OVER (PARTITION BY origin ORDER BY TotalDelays DESC) as rank 
       FROM departureDelaysWindow
  ) t 
 WHERE rank <= 3
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Modifications
// MAGIC 
// MAGIC Another common DataFrame operation is to perform modifications to the DataFrame. Recall that the underlying RDDs are immutable (i.e. they do not change) to ensure there is data lineage for Spark operations. Hence while DataFrames themselves are immutable, you can modify them through operations that create a new, different DataFrame with different columns, for example.  

// COMMAND ----------

// MAGIC %md
// MAGIC ### Adding New Columns

// COMMAND ----------

val foo2 = foo.withColumn("status", expr("CASE WHEN delay <= 10 THEN 'On-time' ELSE 'Delayed' END"))
foo2.show()

// COMMAND ----------

spark.sql("""SELECT *, CASE WHEN delay <= 10 THEN 'On-time' ELSE 'Delayed' END AS status FROM foo""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Dropping Columns

// COMMAND ----------

val foo3 = foo2.drop("delay")
foo3.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Renaming Columns

// COMMAND ----------

val foo4 = foo3.withColumnRenamed("status", "flight_status")
foo4.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Pivoting
// MAGIC Great reference [SQL Pivot: Converting Rows to Columns](https://databricks.com/blog/2018/11/01/sql-pivot-converting-rows-to-columns.html)

// COMMAND ----------

spark.sql("""SELECT destination, CAST(SUBSTRING(date, 0, 2) AS int) AS month, delay FROM departureDelays WHERE origin = 'SEA'""").show(10)

// COMMAND ----------

spark.sql("""
SELECT * FROM (
SELECT destination, CAST(SUBSTRING(date, 0, 2) AS int) AS month, delay 
  FROM departureDelays WHERE origin = 'SEA' 
) 
PIVOT (
  CAST(AVG(delay) AS DECIMAL(4, 2)) as AvgDelay, MAX(delay) as MaxDelay
  FOR month IN (1 JAN, 2 FEB, 3 MAR)
)
ORDER BY destination
""").show()

// COMMAND ----------

spark.sql("""
SELECT * FROM (
SELECT destination, CAST(SUBSTRING(date, 0, 2) AS int) AS month, delay 
  FROM departureDelays WHERE origin = 'SEA' 
) 
PIVOT (
  CAST(AVG(delay) AS DECIMAL(4, 2)) as AvgDelay, MAX(delay) as MaxDelay
  FOR month IN (1 JAN, 2 FEB)
)
ORDER BY destination
""").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Rollup
// MAGIC Refer to [What is the difference between cube, rollup and groupBy operators?](https://stackoverflow.com/questions/37975227/what-is-the-difference-between-cube-rollup-and-groupby-operators)