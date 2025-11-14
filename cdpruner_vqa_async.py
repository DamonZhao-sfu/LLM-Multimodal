import sys
import os
import time
import pandas as pd
from typing import Tuple, List, Dict, Any, Callable
import json
import asyncio
import csv
from threading import Lock

from util.utils import _generate_prompt

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, lower, trim
from pyspark.sql.types import StringType
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.cdpruner import *
from util.trimTokenator import *

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
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.executor.memoryOverhead", "16g") \
    .config("spark.python.worker.memory", "32g") \
    .config("spark.rpc.message.maxSize", "512") \
    .getOrCreate()

# Global variables for model configuration
API_URL = "http://localhost:8000/v1/chat/completions"
RECOVERY_RATIO = 0.0

# Global variables to track timing and CSV file
TIMING_CSV_FILE = None
TIMING_CSV_LOCK = Lock()
INVOCATION_COUNTER = 0


def initialize_timing_csv(keep_ratio: float, dataset_name: str):
    """Initialize CSV file for timing records."""
    global TIMING_CSV_FILE, INVOCATION_COUNTER
    
    INVOCATION_COUNTER = 0
    TIMING_CSV_FILE = f"./{dataset_name}_{keep_ratio}_timing.csv"
    
    # Create CSV with headers
    with open(TIMING_CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['invocation_id', 'keep_ratio', 'preprocess_time', 'encode_time', 'prune_time', 'total_time'])
    
    print(f"Timing CSV initialized: {TIMING_CSV_FILE}")


def record_timing(keep_ratio: float, preprocess_time: float, encode_time: float, prune_time: float):
    """Record timing information to CSV file."""
    global TIMING_CSV_FILE, TIMING_CSV_LOCK, INVOCATION_COUNTER
    
    if TIMING_CSV_FILE is None:
        return
    
    INVOCATION_COUNTER += 1
    total_time = preprocess_time + encode_time + prune_time
    
    with open(TIMING_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            INVOCATION_COUNTER,
            keep_ratio,
            f"{preprocess_time:.6f}",
            f"{encode_time:.6f}",
            f"{prune_time:.6f}",
            f"{total_time:.6f}"
        ])


def execute_batch_pope_with_pruned_embeddings(
    modelname: str,
    fields: List[Dict[str, Any]],
    query: str,
    keep_ratio: float,
    typed_fields: List[Tuple[str, str]],
    reordered_columns: List[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
) -> List[str]:
    """Returns: outputs"""
    vision_tower, model, tokenizer = load_vision_models(device='cuda')
    
    try:
        # Build user prompts and generate pruned embeddings
        user_prompts = []
        all_pruned_embeddings = []
        
        for field_dict in fields:
            # Initialize prompt with empty string - we'll build it from scratch
            user_prompt = ""
            pruned_embeddings_for_this_prompt = []
            
            # Build prompt following the REORDERED column sequence
            for field_name in reordered_columns:
                # Find the field type for this field name
                field_type = None
                for fname, ftype in typed_fields:
                    if fname == field_name:
                        field_type = ftype
                        break
                
                if field_type is None:
                    continue  # Skip if field not found in typed_fields
                if field_type == "text":
                    value = field_dict.get(field_name, "")
                    user_prompt += f"{field_name}: {value}\n"
                
                elif field_type == "image":
                    user_prompt += f"{field_name}: [image]\n"
                    image_data = field_dict.get(field_name)
                    
                    if image_data is not None:
                        image_binary = extract_image_binary_from_pope_data(image_data)
                        
                        if keep_ratio == 1:
                            reduced_tokens, preprocess_time, encode_time  = getOriginalVisualToken(
                                model,
                                vision_tower,
                                image_binary
                            )
                            # Record timing with zeros for no pruning
                            record_timing(keep_ratio, preprocess_time, encode_time, 0.0)
                        else:
                            # Time the pruning operation
                            reduced_tokens, preprocess_time, encode_time, prune_time = getCDPrunedVisualToken(
                                model,
                                vision_tower,
                                image_binary,
                                user_prompt,
                                keep_ratio=keep_ratio
                            )
                            # Record timing for this invocation
                            record_timing(keep_ratio, preprocess_time, encode_time, prune_time)
                            
                        pruned_embeddings_for_this_prompt.append(reduced_tokens.to(torch.float16))
            
            user_prompts.append(user_prompt.strip())  # Remove trailing newline
            all_pruned_embeddings.append(
                pruned_embeddings_for_this_prompt[0] if pruned_embeddings_for_this_prompt else None
            )
        
        # Generate full prompts
        prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
                   for user_prompt in user_prompts]
        
        outputs = []
        if base_url:            
            async def fetch_all():
                """
                Concurrently runs all post_http_request_with_embeds calls.
                """
                tasks = []
                for i, prompt in enumerate(prompts):
                    # Use asyncio.to_thread to run the synchronous 
                    # post_http_request_with_embeds in a separate thread.
                    task = asyncio.to_thread(
                        post_http_request_with_embeds,
                        modelname,
                        [prompt],
                        temperature=0,
                        api_url=(base_url + "/chat/completions"),
                        guided_choice=guided_choice,
                        image_embeddings=[all_pruned_embeddings[i]] if all_pruned_embeddings[i] is not None else None
                    )
                    tasks.append(task)
                
                # Gather all responses concurrently
                responses = await asyncio.gather(*tasks)
                
                # Process responses in order
                processed_outputs = []
                for response in responses:
                    try:
                        # Assuming response has a .content attribute like a requests.Response
                        request_output = json.loads(response.content) 
                        choices = request_output.get('choices', [])
                        
                        if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                            processed_outputs.append(choices[0]['message']['content'])
                        else:
                            # Log error or empty response
                            print(f"Warning: No valid content in response. Output: {request_output}")
                            processed_outputs.append(None)
                    except Exception as e:
                        # Log exception
                        print(f"Error processing response: {e}. Response content: {getattr(response, 'content', 'N/A')}")
                        processed_outputs.append(None)
                        
                return processed_outputs

            # Run the async main function from our synchronous context
            # This will block until all concurrent requests are complete
            outputs = asyncio.run(fetch_all())            
            return outputs
    
    finally:
        torch.cuda.empty_cache()


def create_llm_udf_with_embeddings(
    keep_ratio: float
):
    @pandas_udf(StringType())
    def llm_udf_embedding_batch(
        prompts: pd.Series,
        *args: pd.Series
    ) -> pd.Series:      
        print(f"Batch size: {len(prompts)}")  
        prompt_template = prompts.iloc[0]
        typed_fields = parse_typed_fields(prompt_template)

        if len(args) != len(typed_fields):
            raise ValueError(
                f"Expected {len(typed_fields)} column(s) for fields {[f[0] for f in typed_fields]}, "
                f"but got {len(args)}."
            )
        
        # Build initial data dictionary
        data_dict = {}
        for i, (field_name, field_type) in enumerate(typed_fields):
            arg = args[i]
            if isinstance(arg, pd.DataFrame):
                data_dict[field_name] = arg.values.tolist()
            elif isinstance(arg, pd.Series):
                data_dict[field_name] = arg.tolist()
            else:
                data_dict[field_name] = list(arg)

        # Create DataFrame
        merged_df = pd.DataFrame(data_dict)
        
        reordered_columns = list(merged_df.columns)
        
        # Convert to records for processing
        fields_list = merged_df.to_dict('records')
        
        outputs = execute_batch_pope_with_pruned_embeddings(
            modelname="/data/models/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            keep_ratio=keep_ratio,
            typed_fields=typed_fields,
            reordered_columns=reordered_columns,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            base_url="http://localhost:8000/v1"
        )
                
        return pd.Series(outputs)
    
    return llm_udf_embedding_batch


def extract_image_binary_from_pope_data(image_data):
    """Extract image binary from POPE data format."""
    if isinstance(image_data, (list, tuple)):
        return image_data[0] if len(image_data) > 0 else image_data
    return image_data


def run_experiment(keep_ratio: float, dataset_name: str = "POPE_random") -> Tuple[str, float, float]:
    """Run experiment with specific keep_ratio and save results.
    Returns: (output_path, execution_time, pruning_time)
    """
    print(f"\n{'='*80}")
    print(f"Running experiment with keep_ratio={keep_ratio}")
    print(f"{'='*80}\n")
    
    # Initialize timing CSV for this experiment
    initialize_timing_csv(keep_ratio, dataset_name)
    
    start_time = time.time()
    
    # Register UDF with current keep_ratio
    llm_udf = create_llm_udf_with_embeddings(keep_ratio)
    spark.udf.register("LLM", llm_udf)
    
    # Read POPE parquet
    POPE_PATH = "/home/haikai/LLM-Multimodal/VQAv2/validation-00000-of-00068.parquet"
    pope_df = spark.read.parquet(POPE_PATH)
    pope_df.createOrReplaceTempView("pope")
    
    # Execute query with proper column references
    result_df = spark.sql("""
        SELECT 
            multiple_choice_answer,
            LLM('Given the question: {text:question} and candidate answers {text:answers} and {text:answer_type} and image: {image:image}, give me the answer to the question', question, answers, answer_type, image) as predicted
        FROM pope
    """)
    
    # Add comparison column
    result_df_with_comparison = result_df.withColumn(
        "is_correct",
        when(
            lower(col("predicted")).contains(lower(trim(col("multiple_choice_answer")))),
            1
        ).otherwise(0)
    ).drop("predicted")
    
    # Write results to CSV
    output_path = f"./{dataset_name}_{keep_ratio}.csv"
    result_df_with_comparison.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate pruning time from timing CSV
    pruning_time = 0
    if TIMING_CSV_FILE and os.path.exists(TIMING_CSV_FILE):
        timing_df = pd.read_csv(TIMING_CSV_FILE)
        pruning_time = timing_df['total_time'].sum()
    
    print(f"\nExecution time for keep_ratio={keep_ratio}: {execution_time:.2f} seconds")
    print(f"Pruning time for keep_ratio={keep_ratio}: {pruning_time:.2f} seconds")
    print(f"Pruning percentage: {pruning_time/execution_time*100:.2f}%")
    print(f"Total invocations: {INVOCATION_COUNTER}")
    
    # Write execution time to text file
    time_log_path = f"./{dataset_name}_{keep_ratio}_execution_time.txt"
    with open(time_log_path, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Keep Ratio: {keep_ratio}\n")
        f.write(f"Total Invocations: {INVOCATION_COUNTER}\n")
        f.write(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
        f.write(f"Total Pruning Time: {pruning_time:.2f} seconds ({pruning_time/60:.2f} minutes)\n")
        f.write(f"Pruning Time Percentage: {pruning_time/execution_time*100:.2f}%\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Timing CSV: {TIMING_CSV_FILE}\n")
        f.write(f"{'='*50}\n")
    
    print(f"Execution time logged to: {time_log_path}")
    print(f"Detailed timing logged to: {TIMING_CSV_FILE}")
    
    return output_path, execution_time, pruning_time


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
    try:
        df = pd.read_csv(actual_csv)
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {actual_csv}")
        return None
    
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


def generate_timing_summary(keep_ratios: List[float], dataset_name: str):
    """Generate summary statistics from all timing CSV files."""
    summary_path = f"./{dataset_name}_timing_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(f"TIMING SUMMARY\n")
        f.write(f"{'='*100}\n\n")
        
        for keep_ratio in keep_ratios:
            timing_csv = f"./{dataset_name}_{keep_ratio}_timing.csv"
            
            if not os.path.exists(timing_csv):
                continue
            
            df = pd.read_csv(timing_csv)
            
            f.write(f"Keep Ratio: {keep_ratio}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"Total Invocations: {len(df)}\n")
            f.write(f"\nPreprocess Time:\n")
            f.write(f"  Mean: {df['preprocess_time'].mean():.6f}s, Median: {df['preprocess_time'].median():.6f}s\n")
            f.write(f"  Min: {df['preprocess_time'].min():.6f}s, Max: {df['preprocess_time'].max():.6f}s\n")
            f.write(f"  Total: {df['preprocess_time'].sum():.6f}s\n")
            f.write(f"\nEncode Time:\n")
            f.write(f"  Mean: {df['encode_time'].mean():.6f}s, Median: {df['encode_time'].median():.6f}s\n")
            f.write(f"  Min: {df['encode_time'].min():.6f}s, Max: {df['encode_time'].max():.6f}s\n")
            f.write(f"  Total: {df['encode_time'].sum():.6f}s\n")
            f.write(f"\nPrune Time:\n")
            f.write(f"  Mean: {df['prune_time'].mean():.6f}s, Median: {df['prune_time'].median():.6f}s\n")
            f.write(f"  Min: {df['prune_time'].min():.6f}s, Max: {df['prune_time'].max():.6f}s\n")
            f.write(f"  Total: {df['prune_time'].sum():.6f}s\n")
            f.write(f"\nTotal Time:\n")
            f.write(f"  Mean: {df['total_time'].mean():.6f}s, Median: {df['total_time'].median():.6f}s\n")
            f.write(f"  Min: {df['total_time'].min():.6f}s, Max: {df['total_time'].max():.6f}s\n")
            f.write(f"  Total: {df['total_time'].sum():.6f}s\n")
            f.write(f"\n{'='*100}\n\n")
    
    print(f"Timing summary saved to: {summary_path}")


# Main execution
if __name__ == "__main__":
    keep_ratios = [0.111, 0.222]
    dataset_name = "vqav2_cdpruner"
    
    overall_start = time.time()
    results = {}
    execution_times = {}
    pruning_times = {}
    
    for keep_ratio in keep_ratios:
        output_path, exec_time, prune_time = run_experiment(
            keep_ratio, 
            dataset_name
        )
        
        execution_times[keep_ratio] = exec_time
        pruning_times[keep_ratio] = prune_time
        
        accuracy = calculate_accuracy(output_path, keep_ratio)
        results[keep_ratio] = accuracy
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    total_pruning_time = sum(pruning_times.values())
    
    # Generate timing summary
    generate_timing_summary(keep_ratios, dataset_name)
    
    # Write summary to text file
    summary_path = f"./{dataset_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EXPERIMENT SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Overall Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
        f.write(f"Total Accumulated Pruning Time: {total_pruning_time:.2f} seconds ({total_pruning_time/60:.2f} minutes)\n")
        f.write(f"Pruning Time Percentage: {total_pruning_time/overall_time*100:.2f}%\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"DETAILED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Keep Ratio':<12} {'Accuracy':<12} {'Exec Time (s)':<15} {'Prune Time (s)':<16} {'Prune %':<10} {'Status':<10}\n")
        f.write(f"{'-'*80}\n")
        for keep_ratio in keep_ratios:
            accuracy = results.get(keep_ratio)
            exec_time = execution_times.get(keep_ratio, 0)
            prune_time = pruning_times.get(keep_ratio, 0)
            prune_pct = (prune_time / exec_time * 100) if exec_time > 0 else 0.0
            if accuracy is not None:
                f.write(f"{keep_ratio:<12.3f} {accuracy:<12.2f}% {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}% {'✓':<10}\n")
            else:
                f.write(f"{keep_ratio:<12.3f} {'N/A':<12} {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}% {'✗':<10}\n")
        f.write(f"{'='*80}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Total pruning time: {total_pruning_time:.2f} seconds ({total_pruning_time/60:.2f} minutes)")
    print(f"Pruning time percentage: {total_pruning_time/overall_time*100:.2f}%\n")
    print(f"{'Keep Ratio':<12} {'Accuracy':<12} {'Exec Time (s)':<15} {'Prune Time (s)':<16} {'Prune %':<10} {'Status':<10}")
    print(f"{'-'*80}")
    for keep_ratio in keep_ratios:
        accuracy = results.get(keep_ratio)
        exec_time = execution_times.get(keep_ratio, 0)
        prune_time = pruning_times.get(keep_ratio, 0)
        prune_pct = (prune_time / exec_time * 100) if exec_time > 0 else 0.0
        if accuracy is not None:
            print(f"{keep_ratio:<12.3f} {accuracy:<12.2f}% {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}% {'✓':<10}")
        else:
            print(f"{keep_ratio:<12.3f} {'N/A':<12} {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}% {'✗':<10}")
    print(f"{'='*80}\n")
    
    spark.stop()