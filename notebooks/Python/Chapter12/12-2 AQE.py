# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # Adaptive Query Execution Demo
# MAGIC 
# MAGIC Adaptive Query Execution (AQE) is query re-optimization that occurs during query execution based on runtime statistics. AQE in Spark 3.0 includes 3 main features:
# MAGIC * Dynamically coalescing shuffle partitions
# MAGIC * Dynamically switching join strategies
# MAGIC * Dynamically optimizing skew joins

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable AQE

# COMMAND ----------

# MAGIC %sql
# MAGIC set spark.sql.adaptive.enabled = true;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- For demo purpose only.
# MAGIC -- Not necessary in real-life usage.
# MAGIC 
# MAGIC set spark.sql.adaptive.coalescePartitions.minPartitionNum = 1;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Tables

# COMMAND ----------

dbutils.fs.rm("dbfs:/user/hive/warehouse/aqe_demo_db", True)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS aqe_demo_db;
# MAGIC USE aqe_demo_db;
# MAGIC 
# MAGIC DROP TABLE IF EXISTS items;
# MAGIC DROP TABLE IF EXISTS sales;
# MAGIC 
# MAGIC -- Create "items" table.
# MAGIC 
# MAGIC CREATE TABLE items
# MAGIC USING parquet
# MAGIC AS
# MAGIC SELECT id AS i_item_id,
# MAGIC CAST(rand() * 1000 AS INT) AS i_price
# MAGIC FROM RANGE(30000000);
# MAGIC 
# MAGIC -- Create "sales" table with skew.
# MAGIC -- Item with id 100 is in 80% of all sales.
# MAGIC 
# MAGIC CREATE TABLE sales
# MAGIC USING parquet
# MAGIC AS
# MAGIC SELECT CASE WHEN rand() < 0.8 THEN 100 ELSE CAST(rand() * 30000000 AS INT) END AS s_item_id,
# MAGIC CAST(rand() * 100 AS INT) AS s_quantity,
# MAGIC DATE_ADD(current_date(), - CAST(rand() * 360 AS INT)) AS s_date
# MAGIC FROM RANGE(1000000000);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dynamically Coalesce Shuffle Partitions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get the sums of sales quantity grouped by sales date.
# MAGIC -- The grouped result is very small.
# MAGIC 
# MAGIC SELECT s_date, sum(s_quantity) AS q
# MAGIC FROM sales
# MAGIC GROUP BY s_date
# MAGIC ORDER BY q DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC * The partition sizes after aggregation are very small: 13KB on average, 431KB in total (see the highlighted box **shuffle bytes written**).
# MAGIC * AQE combines these small partitions into one new partition (see the highlighted box **CustomShuffleReader**).
# MAGIC 
# MAGIC ![screenshot_coalesce](https://docs.databricks.com/_static/images/spark/aqe/coalesce_partitions.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dynamically Switch Join Strategies

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get total sales amount grouped by sales date for items with a price lower than 10.
# MAGIC -- The selectivity of the filter by price is not known in static planning, so the initial plan opts for sort merge join.
# MAGIC -- But in fact, the "items" table after filtering is very small, so the query can do a broadcast hash join instead.
# MAGIC 
# MAGIC -- Static explain shows the initial plan with sort merge join.
# MAGIC 
# MAGIC EXPLAIN
# MAGIC SELECT s_date, sum(s_quantity * i_price) AS total_sales
# MAGIC FROM sales
# MAGIC JOIN items ON s_item_id = i_item_id
# MAGIC WHERE i_price < 10
# MAGIC GROUP BY s_date
# MAGIC ORDER BY total_sales DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The runtime join strategy is changed to broadcast hash join.
# MAGIC 
# MAGIC SELECT s_date, sum(s_quantity * i_price) AS total_sales
# MAGIC FROM sales
# MAGIC JOIN items ON s_item_id = i_item_id
# MAGIC WHERE i_price < 10
# MAGIC GROUP BY s_date
# MAGIC ORDER BY total_sales DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC * The data size of the "items" table after filtering is very small 6.9 MB (see the highlighted box **data size**).
# MAGIC *  AQE changes the sort merge join to broadcast hash join at runtime (see the highlighted box **BroadcastHashJoin**).
# MAGIC 
# MAGIC ![screenshot_strategy](https://docs.databricks.com/_static/images/spark/aqe/join_strategy.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Dynamically Optimize Skew Join

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get the total sales amount grouped by sales date.
# MAGIC -- The partition in the "sales" table containing value "100" as "s_item_id" is much larger than other partitions.
# MAGIC -- AQE splits the skewed partition into smaller partitions before joining the "sales" table with the "items" table.
# MAGIC 
# MAGIC SELECT s_date, sum(s_quantity * i_price) AS total_sales
# MAGIC FROM sales
# MAGIC JOIN items ON i_item_id = s_item_id
# MAGIC GROUP BY s_date
# MAGIC ORDER BY total_sales DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC * There is a skewed partition from the "sales" table (see the highlighted box **number of skewed partitions**).
# MAGIC * AQE splits the skewed partition into smaller partitions (see the highlighted box **number of skewed partition splits**).
# MAGIC * The sort merge join operator is marked with a skew join flag (see the highlighted box **SortMergeJoin(isSkew=true)**).
# MAGIC 
# MAGIC ![screenshot_skew](https://docs.databricks.com/_static/images/spark/aqe/skew_join.png)