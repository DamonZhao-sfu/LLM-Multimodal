import sys
import os
import time
import pandas as pd
from typing import Tuple
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, lower, trim
from pyspark.sql.types import StringType
from util.mllm import *
from util.utils import *
from util.cdencoder import *

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)


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
RECOVERY_RATIO = 0.0

def create_llm_udf_with_embeddings(keep_ratio):    
    @pandas_udf(StringType())
    def llm_udf_embedding_batch(
        prompts: pd.Series,
        *args: pd.Series
    ) -> pd.Series:        
        prompt_template = prompts.iloc[0]
        print(f"Processing batch on PID {os.getpid()} with keep_ratio={keep_ratio}")
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
            modelname="llava-hf/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            keep_ratio=keep_ratio,
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


def run_experiment(keep_ratio: float, dataset_name: str = "POPE_random") -> Tuple[str, float]:
    """Run experiment with specific keep_ratio and save results."""
    print(f"\n{'='*80}")
    print(f"Running experiment with keep_ratio={keep_ratio}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Register UDF with current keep_ratio
    llm_udf = create_llm_udf_with_embeddings(keep_ratio)
    spark.udf.register("LLM", llm_udf)
    
    # Read POPE parquet
    POPE_PATH = "/scratch/hpc-prf-haqc/haikai/dataset/POPE/random-00000-of-00001.parquet"
    pope_df = spark.read.parquet(POPE_PATH)
    pope_df.createOrReplaceTempView("pope")
    print(f"Total records: {pope_df.count()}")
    
    # Execute query with proper column references
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
    
    # Normalize both answer and predicted columns for comparison
    result_df_with_comparison = result_df.withColumn(
        "is_correct",
        when(
            lower(trim(col("predicted"))) == lower(trim(col("answer"))),
            1
        ).otherwise(0)
    )
    
    # Write results to CSV
    output_path = f"./{dataset_name}_{keep_ratio}.csv"
    result_df_with_comparison.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time for keep_ratio={keep_ratio}: {execution_time:.2f} seconds")
    
    # Write execution time to text file
    time_log_path = f"./{dataset_name}_{keep_ratio}_execution_time.txt"
    with open(time_log_path, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Keep Ratio: {keep_ratio}\n")
        f.write(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"{'='*50}\n")
    
    print(f"Execution time logged to: {time_log_path}")
    
    return output_path, execution_time


def calculate_accuracy(csv_path: str, keep_ratio: float) -> float:
    """Read CSV and calculate accuracy."""
    # Find the actual CSV file in the directory (Spark creates a folder)
    if os.path.isdir(csv_path):
        csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv') and not f.startswith('.')]
        if csv_files:
            actual_csv = os.path.join(csv_path, csv_files[0])
        else:
            print(f"Warning: No CSV file found in {csv_path}")
            return None
    else:
        actual_csv = csv_path
    
    # Read CSV
    df = pd.read_csv(actual_csv)
    
    # Calculate accuracy
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Results for keep_ratio={keep_ratio}")
    print(f"{'='*80}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")
    
    return accuracy


# Main execution
if __name__ == "__main__":
    keep_ratios = [0.056, 0.111, 0.222]
    dataset_name = "POPE_random"
    
    overall_start = time.time()
    results = {}
    execution_times = {}
    
    for keep_ratio in keep_ratios:
        output_path, exec_time = run_experiment(keep_ratio, dataset_name)
        execution_times[keep_ratio] = exec_time
        
        accuracy = calculate_accuracy(output_path, keep_ratio)
        results[keep_ratio] = accuracy
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # Write summary to text file
    summary_path = f"./{dataset_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EXPERIMENT SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Overall Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"DETAILED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Keep Ratio':<15} {'Accuracy':<15} {'Exec Time (s)':<20} {'Status':<15}\n")
        f.write(f"{'-'*65}\n")
        for keep_ratio in keep_ratios:
            accuracy = results.get(keep_ratio)
            exec_time = execution_times.get(keep_ratio, 0)
            if accuracy is not None:
                f.write(f"{keep_ratio:<15.3f} {accuracy:<15.2f}% {exec_time:<20.2f} {'✓':<15}\n")
            else:
                f.write(f"{keep_ratio:<15.3f} {'N/A':<15} {exec_time:<20.2f} {'✗':<15}\n")
        f.write(f"{'='*80}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
    print(f"{'Keep Ratio':<15} {'Accuracy':<15} {'Exec Time (s)':<20} {'Status':<15}")
    print(f"{'-'*65}")
    for keep_ratio in keep_ratios:
        accuracy = results.get(keep_ratio)
        exec_time = execution_times.get(keep_ratio, 0)
        if accuracy is not None:
            print(f"{keep_ratio:<15.3f} {accuracy:<15.2f}% {exec_time:<20.2f} {'✓':<15}")
        else:
            print(f"{keep_ratio:<15.3f} {'N/A':<15} {exec_time:<20.2f} {'✗':<15}")
    print(f"{'='*80}\n")
    
    spark.stop()