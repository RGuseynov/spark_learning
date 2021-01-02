# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC ## Example 2-1 M&M Count

# COMMAND ----------

from pyspark.sql.functions import *

mnm_file = "/databricks-datasets/learning-spark-v2/mnm_dataset.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read from the CSV and infer the schema

# COMMAND ----------

mnm_df = (spark
          .read
          .format("csv")
          .option("header", "true")
          .option("inferSchema", "true")
          .load(mnm_file))

display(mnm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate count of all colors and groupBy state and color, orderBy descending order

# COMMAND ----------

count_mnm_df = (mnm_df
                .select("State", "Color", "Count")
                .groupBy("State", "Color")
                .agg(count("Count").alias("Total"))
                .orderBy("Total", ascending=False))

count_mnm_df.show(n=60, truncate=False)
print("Total Rows = %d" % (count_mnm_df.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the aggregate count for California by filtering on State

# COMMAND ----------

ca_count_mnm_df = (mnm_df
                   .select("State", "Color", "Count") 
                   .where(mnm_df.State == "CA") 
                   .groupBy("State", "Color") 
                   .agg(count("Count").alias("Total")) 
                   .orderBy("Total", ascending=False))

ca_count_mnm_df.show(n=10, truncate=False)
