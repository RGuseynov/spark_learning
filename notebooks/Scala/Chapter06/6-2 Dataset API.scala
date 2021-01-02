// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC # Datasets: The DataFrame Query Language vs. Lambdas
// MAGIC 
// MAGIC The Dataset API gives you the option to use both the DataFrame query language _and_ RDD-like lambda transformations.

// COMMAND ----------

case class Person(id: Integer, firstName: String, middleName: String, lastName: String, gender: String, birthDate: String, ssn: String, salary: String)

val personDS = spark
  .read
  .option("header", "true")
  .option("inferSchema", "true")
  .option("delimiter", ":")
  .csv("/mnt/training/dataframes/people-with-header-10m.txt")
  .as[Person]

personDS.cache().count

// COMMAND ----------

// DataFrame query DSL
println(personDS.filter($"firstName" === "Nell").distinct().count)

// COMMAND ----------

// Dataset, with a lambda
println(personDS.filter(x => x.firstName == "Nell").distinct().count)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Tungsten Encoders' effect on Catalyst Optimization
// MAGIC 
// MAGIC The Domain Specific Language (DSL) used by DataFrames and DataSets allows for data manipulation without having to deserialize that data from the Tungsten format. 
// MAGIC 
// MAGIC <img src="https://files.training.databricks.com/images/tuning/dsl-lambda.png" alt="Lambda serialization overhead"/><br/>
// MAGIC 
// MAGIC The advantage of this is that we avoid any *serialization / deserialization* overhead. <br/>
// MAGIC Datasets give users the ability to carry out data manipulation through lambdas which can be very powerful, especially with semi-structured data. The **downside** of lambda is that they can't directly work with the Tungsten format, thus deserialization is required adding an overhead to the process.
// MAGIC 
// MAGIC Avoiding frequent jumps between DSL and closures would mean that the *serialization / deserialization* to and from Tungsten format would be reduced, leading to a performance gain.

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC 
// MAGIC - Advantages of using lambdas:
// MAGIC     - Good for semi-structured data
// MAGIC     - Very powerful
// MAGIC - Disadvantages:
// MAGIC     - Catalyst can't interpret lambdas until runtime. 
// MAGIC     - Lambdas are opaque to Catalyst. Since it doesn't know what a lambda is doing, it can't move it elsewhere in the processing.
// MAGIC     - Jumping between lambdas and the DataFrame query API can hurt performance.
// MAGIC     - Working with lambdas means that we need to `deserialize` from Tungsten's format to an object and then reserialize back to Tungsten format when the lambda is done.
// MAGIC     
// MAGIC If you _have_ to use lambdas, chaining them together can help.

// COMMAND ----------

// define the year 40 years ago for the below query
import java.util.Calendar
val earliestYear = Calendar.getInstance.get(Calendar.YEAR) - 40

personDS
  .filter(x => x.birthDate.split("-")(0).toInt > earliestYear) // everyone above 40
  .filter($"salary" > 80000) // everyone earning more than 80K
  .filter(x => x.lastName.startsWith("J")) // last name starts with J
  .filter($"firstName".startsWith("D")) // first name starts with D
  .count()

// COMMAND ----------

import org.apache.spark.sql.functions._

personDS
  .filter(year($"birthDate") > earliestYear) // everyone above 40
  .filter($"salary" > 80000) // everyone earning more than 80K
  .filter($"lastName".startsWith("J")) // last name starts with J
  .filter($"firstName".startsWith("D")) // first name starts with D
  .count()

// COMMAND ----------

// MAGIC %md
// MAGIC Look at how much faster it is to use the DataFrame API!