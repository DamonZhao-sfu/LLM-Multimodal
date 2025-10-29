import sys
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, lower, trim
from pyspark.sql.types import StringType
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.quick_greedy import *

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

output_path = "./demoResult.csv"

# Spark configuration
spark = SparkSession.builder \
    .appName("LLM SQL Test") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "128g") \
    .config("spark.executor.cores", "32") \
    .config("spark.executor.instances", "1") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.default.parallelism", "1") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.executor.memoryOverhead", "16g") \
    .config("spark.python.worker.memory", "32g") \
    .config("spark.rpc.message.maxSize", "512") \
    .getOrCreate()

# Global variables for model configuration
API_URL = "http://localhost:8000/v1/chat/completions"
KEEP_RATIO = 0.5
RECOVERY_RATIO = 0.0
MODEL_PATH = "/scratch/hpc-prf-haqc/haikai/hf-cache/llava-1.5-7b-hf"

def create_llm_udf_with_embeddings():    
    @pandas_udf(StringType())
    def llm_udf_embedding_batch(
        prompts: pd.Series,
        *args: pd.Series
    ) -> pd.Series:        
        # Your processing logic
        results = []
        prompt_template = prompts.iloc[0]
        print(f"Processing batch on PID {os.getpid()}")
        typed_fields = parse_typed_fields(prompt_template)

        if len(args) != len(typed_fields):
            raise ValueError(
                f"Expected {len(typed_fields)} column(s) for fields {[f[0] for f in typed_fields]}, "
                f"but got {len(args)}."
            )
        
        data_dict = {}
        for i, (field_name, field_type) in enumerate(typed_fields):
            arg = args[i]
            if isinstance(arg, pd.DataFrame):
                data_dict[field_name] = arg.values.tolist()
            elif isinstance(arg, pd.Series):
                data_dict[field_name] = arg.tolist()
            else:
                data_dict[field_name] = list(arg)

        merged_df = pd.DataFrame(data_dict)
        fields_list = merged_df.to_dict('records')
        
        outputs = execute_batch_v2_with_pruned_embeddings(
            #model=model,
            modelname="llava-hf/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            typed_fields=typed_fields,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            guided_choice=["Yes", "No"],
            base_url="http://localhost:8000/v1"
        )
        
        return pd.Series(outputs)
    
    return llm_udf_embedding_batch



def extract_image_binary_from_pope_data(image_data):
    """Extract image binary from POPE data format."""
    if isinstance(image_data, (list, tuple)):
        return image_data[0] if len(image_data) > 0 else image_data
    return image_data

# Register UDF
llm_udf = create_llm_udf_with_embeddings()
spark.udf.register("LLM", llm_udf)

# Main execution
start_time = time.time()

# Read POPE parquet

if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

POPE_PATH = "/scratch/hpc-prf-haqc/haikai/dataset/POPE/random-00000-of-00001.parquet"
pope_df = spark.read.parquet(POPE_PATH)
pope_df.createOrReplaceTempView("pope")
print(f"Total records: {pope_df.count()}")

algo_begin_time = time.time()

from pyspark.sql.functions import to_json, col
from pyspark.sql.types import StructType, MapType, ArrayType

for field in pope_df.schema.fields:
    if isinstance(field.dataType, (StructType, MapType, ArrayType)):
        pope_df = pope_df.withColumn(field.name, to_json(col(field.name)))

pope_pandas_df = pope_df.toPandas()

pope_pandas_df, _ = QuickGreedy().reorder(
            pope_pandas_df,
            early_stop=100000,
            col_merge=[],
            one_way_dep=[],
            distinct_value_threshold=0.7,
        )
pope_df = spark.createDataFrame(pope_pandas_df)
algo_end_time = time.time()
print(f"\nTotal algo time: {algo_end_time - algo_begin_time:.2f} seconds")


# Execute query with proper column references - include all relevant columns
result_df = spark.sql("""
    SELECT 
        id,
        question_id,
        question,
        answer,
        image_source,
        LLM('Given the text: {text:question} and image: {image:image} give me the answer to the question', question, image) as predicted
    FROM pope
""")

# Normalize both answer and predicted columns for comparison (case-insensitive, trimmed)
result_df_with_comparison = result_df.withColumn(
    "is_correct",
    when(
        lower(trim(col("predicted"))) == lower(trim(col("answer"))),
        1
    ).otherwise(0)
)

# Write results to CSV
result_df_with_comparison.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
end_time = time.time()

# Calculate and display accuracy
total_count = result_df_with_comparison.count()
correct_count = result_df_with_comparison.filter(col("is_correct") == 1).count()
accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

print("\n" + "="*60)
print("PREDICTION ACCURACY RESULTS")
print("="*60)
print(f"Total predictions: {total_count}")
print(f"Correct predictions: {correct_count}")
print(f"Incorrect predictions: {total_count - correct_count}")
print(f"Accuracy: {accuracy:.2f}%")
print("="*60)
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")



spark.stop()