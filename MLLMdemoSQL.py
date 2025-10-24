import sys
import os
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from util.utils import *
from util.cdencoder import *
import re
import json
from typing import List, Dict, Optional

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)
output_path = "./demoResult.csv"

spark = SparkSession.builder \
    .appName("LLM SQL Test") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "128g") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", 50000) \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()
    
set_global_state(spark, "pope")
spark.udf.register("LLM", llm_udf_embedding)
POPE_PATH = "/home/haikai/haikai/entropyTest/POPE.parquet"

start_time = time.time()

# Read POPE parquet and create temp view
pope_df = spark.read.parquet(POPE_PATH).limit(100)
pope_df.createOrReplaceTempView("pope")

# Build LLM SQL based on POPE schema
pope_columns = pope_df.columns
if len(pope_columns) == 0:
    raise ValueError("POPE parquet has no columns; cannot build LLM SQL query.")

# Escape column names in prompt and wrap column identifiers with backticks for SQL safety
prompt_fields = ", ".join([f"{{{col}}}" for col in pope_columns])
sql_columns = ", ".join([f"`{col}`" for col in pope_columns])
prompt_text = (
    f"Given the following fields from the POPE dataset {prompt_fields}, "
    f"provide a concise summary of the content and key details"
).replace("'", "''")

result_df = spark.sql("SELECT LLM('Given the text: {text:question} and image: {image:image} give me the answer to the question', question, image) as summary FROM pope LIMIT 10")
result_df.explain()
result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

end_time = time.time()