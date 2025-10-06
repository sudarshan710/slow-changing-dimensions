#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp, expr, when
from delta.tables import DeltaTable
import datetime

from ntbk_logger import get_notebook_logger

logger = get_notebook_logger(logfile="logs/test.log")

spark = SparkSession.builder \
    .appName("scd") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Input data
new_data = [
    {"id": 9,  "name": "Ivy", "age": 30, "department": "Marketing", "salary": 66000},
    {"id": 10, "name": "Jack", "age": 38, "department": "IT", "salary": 50000},
    {"id": 11, "name": "Karen", "age": 27, "department": "HR", "salary": 51000},
    {"id": 12, "name": "Leo", "age": 41, "department": "Engineering", "salary": 99000},
    {"id": 13, "name": "Maya", "age": 34, "department": "Sales", "salary": 61000},
    {"id": 14, "name": "Nina", "age": 29, "department": "Sales", "salary": 54000},
    {"id": 15, "name": "Oscar", "age": 32, "department": "HR", "salary": 54000},
    {"id": 16, "name": "Paul", "age": 36, "department": "IT", "salary": 87000},
    {"id": 17, "name": "Queenie", "age": 25, "department": "Marketing", "salary": 23445},
    {"id": 18, "name": "Raj", "age": 39, "department": "Sales", "salary": 53000},
    {"id": 19, "name": "Sara", "age": 28, "department": "HR", "salary": 50000},
    {"id": 20, "name": "Tom", "age": 45, "department": "Engineering", "salary": 223344}
]

new_data_df = spark.createDataFrame(new_data)

# Data validation and cleansing (4.3)
logger.info("Validating incoming data")

valid_data_df = new_data_df.filter(
    (col("id").isNotNull()) & 
    (col("salary").isNotNull()) & 
    (col("salary") > 0)
)

invalid_count = new_data_df.count() - valid_data_df.count()
if invalid_count > 0:
    logger.warning(f"Found and excluded {invalid_count} invalid records from incoming data.")

FAR_FUTURE_DATE = "9999-12-31"

try:
    delta_table_path = "data/employee"

    # Add audit columns and SCD2 metadata columns (4.2)
    incoming_scd2_df = valid_data_df.withColumn("start_time", current_timestamp()) \
                                    .withColumn("end_time", lit(FAR_FUTURE_DATE).cast("timestamp")) \
                                    .withColumn("cur_status", lit("active")) \
                                    .withColumn("created_by", lit("system")) \
                                    .withColumn("created_at", current_timestamp()) \
                                    .withColumn("updated_by", lit(None).cast("string")) \
                                    .withColumn("updated_at", lit(None).cast("timestamp"))

    if DeltaTable.isDeltaTable(spark, delta_table_path):
        logger.info("Delta table exists. Loading existing active records.")
        delta_table = DeltaTable.forPath(spark, delta_table_path)
        existing_df = delta_table.toDF().filter("cur_status = 'active'")

        # Detect changed or new records (based on id and relevant columns)
        compare_df = incoming_scd2_df.alias("incoming").join(
            existing_df.alias("existing"), on="id", how="left_outer"
        ).filter(
            (col("existing.id").isNull()) | 
            (col("incoming.salary") != col("existing.salary")) |
            (col("incoming.department") != col("existing.department")) |
            (col("incoming.name") != col("existing.name")) |
            (col("incoming.age") != col("existing.age"))
        ).select("incoming.*")

        # Use rdd.isEmpty() for efficient empty check (3.2)
        if compare_df.rdd.isEmpty():
            logger.info("No changes detected in incoming data. Skipping merge.")
        else:
            changed_ids = [row["id"] for row in compare_df.select("id").distinct().collect()]
            logger.info(f"Changes detected for IDs: {changed_ids}")

            # Expire old records for changed/new IDs
            delta_table.alias("target").merge(
                compare_df.alias("source"),
                "target.id = source.id AND target.cur_status = 'active'"
            ).whenMatchedUpdate(
                condition="""
                    target.department != source.department OR
                    target.salary != source.salary OR
                    target.name != source.name OR
                    target.age != source.age
                """,
                set={
                    "end_time": expr("current_timestamp()"),
                    "cur_status": lit("inactive"),
                    "updated_by": lit("system"),
                    "updated_at": expr("current_timestamp()")
                }
            ).whenNotMatchedInsertAll().execute()

            logger.info("Expired old records and inserted new/changed records.")

        # (4.1) Soft delete: expire active records not in incoming data (deleted records)
        incoming_ids = [row["id"] for row in incoming_scd2_df.select("id").distinct().collect()]
        existing_active_df = delta_table.toDF().filter("cur_status = 'active'")

        deleted_df = existing_active_df.filter(~col("id").isin(incoming_ids))
        if deleted_df.count() > 0:
            deleted_ids = [row["id"] for row in deleted_df.select("id").collect()]
            logger.info(f"Soft deleting records not present in incoming data (IDs: {deleted_ids})")

            # Mark missing records as inactive
            delta_table.alias("target").merge(
                deleted_df.alias("source"),
                "target.id = source.id AND target.cur_status = 'active'"
            ).whenMatchedUpdate(
                set={
                    "end_time": expr("current_timestamp()"),
                    "cur_status": lit("inactive"),
                    "updated_by": lit("system"),
                    "updated_at": expr("current_timestamp()")
                }
            ).execute()
            logger.info("Soft delete operation completed.")
        else:
            logger.info("No records to soft delete.")

    else:
        # First time write (full insert)
        incoming_scd2_df.write.format("delta").mode("overwrite").save(delta_table_path)
        logger.info("Delta table did not exist. Created new table with incoming data.")

except Exception as e:
    logger.error(f"Exception in SCD2 merge: {e}", exc_info=True)

# Read final table for verification
df = spark.read.format("delta").load(delta_table_path)
df.show()