# import sys
# import os
# import time
# import pandas as pd
# from typing import Tuple, List, Dict, Any, Callable
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import pandas_udf, col, when, lower, trim
# from pyspark.sql.types import StringType
# from util.mllm import *
# from util.utils import *
# from util.cdencoder import *
# from util.utils import _generate_prompt
# from util.quick_greedy import *

# # Get the absolute path of the project root
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
# sys.path.insert(0, project_root)


# # Spark configuration
# spark = SparkSession.builder \
#     .appName("LLM SQL Test") \
#     .config("spark.driver.memory", "64g") \
#     .config("spark.executor.memory", "128g") \
#     .config("spark.executor.cores", "32") \
#     .config("spark.executor.instances", "1") \
#     .config("spark.dynamicAllocation.enabled", "false") \
#     .config("spark.default.parallelism", "1") \
#     .config("spark.sql.shuffle.partitions", "1") \
#     .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000") \
#     .config("spark.driver.maxResultSize", "4g") \
#     .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
#     .config("spark.executor.memoryOverhead", "16g") \
#     .config("spark.python.worker.memory", "32g") \
#     .config("spark.rpc.message.maxSize", "512") \
#     .getOrCreate()

# # Global variables for model configuration
# API_URL = "http://localhost:8000/v1/chat/completions"
# RECOVERY_RATIO = 0.0


# def execute_batch_pope_with_pruned_embeddings(
#     modelname: str,
#     fields: List[Dict[str, Any]],
#     query: str,
#     keep_ratio: float,
#     typed_fields: List[Tuple[str, str]],
#     reordered_columns: List[str],  # NEW: List of columns in reordered sequence
#     system_prompt: str = DEFAULT_SYSTEM_PROMPT,
#     guided_choice: List[str] = None,
#     base_url: str = "http://localhost:8000/v1",
# ) -> List[str]:
#     # Load models at the beginning
#     vision_tower, model = load_vision_models(device='cuda:0')
    
#     try:
#         # Build user prompts and generate pruned embeddings
#         user_prompts = []
#         all_pruned_embeddings = []
        
#         for field_dict in fields:
#             # Initialize prompt with empty string - we'll build it from scratch
#             user_prompt = ""
#             pruned_embeddings_for_this_prompt = []
            
#             # Build prompt following the REORDERED column sequence
#             for field_name in reordered_columns:
#                 # Find the field type for this field name
#                 field_type = None
#                 for fname, ftype in typed_fields:
#                     if fname == field_name:
#                         field_type = ftype
#                         break
                
#                 if field_type is None:
#                     continue  # Skip if field not found in typed_fields
#                 if field_type == "text":
#                     value = field_dict.get(field_name, "")
#                     user_prompt += f"{field_name}: {value}\n"
                
#                 elif field_type == "image":
#                     user_prompt += f"{field_name}: [image]\n"
#                     image_data = field_dict.get(field_name)
                    
#                     if image_data is not None:
#                         image_binary = extract_image_binary_from_pope_data(image_data)
                        
#                         if keep_ratio == 1:
#                             reduced_tokens = getOriginalVisualToken(
#                                 model,
#                                 vision_tower,
#                                 image_binary
#                             )
#                         else:
#                             reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
#                                 model,
#                                 vision_tower,
#                                 image_binary,
#                                 user_prompt,
#                                 keep_ratio=keep_ratio
#                             )
                        
#                         pruned_embeddings_for_this_prompt.append(reduced_tokens.to(torch.float16))
            
#             user_prompts.append(user_prompt.strip())  # Remove trailing newline
#             all_pruned_embeddings.append(
#                 pruned_embeddings_for_this_prompt[0] if pruned_embeddings_for_this_prompt else None
#             )
        
#         # Generate full prompts
#         prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
#                    for user_prompt in user_prompts]
        
#         outputs = []
#         if base_url:
#             # Send requests
#             for i, prompt in enumerate(prompts):
#                 response = post_http_request_with_embeds(
#                     modelname,
#                     [prompt],
#                     temperature=0,
#                     api_url=(base_url + "/chat/completions"),
#                     guided_choice=guided_choice,
#                     image_embeddings=[all_pruned_embeddings[i]] if all_pruned_embeddings[i] is not None else None
#                 )
                
#                 request_output = json.loads(response.content)
#                 choices = request_output.get('choices', [])
                
#                 if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
#                     outputs.append(choices[0]['message']['content'])
#                 else:
#                     outputs.append(None)
            
#             return outputs
    
#     finally:
#         # Always clean up models, even if an error occurred
#         cleanup_vision_models(vision_tower, model)


# def create_llm_udf_with_embeddings(
#     keep_ratio: float,
#     reorder_function: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]] = None
# ):
#     @pandas_udf(StringType())
#     def llm_udf_embedding_batch(
#         prompts: pd.Series,
#         *args: pd.Series
#     ) -> pd.Series:        
#         prompt_template = prompts.iloc[0]
#         print(f"Processing batch on PID {os.getpid()} with keep_ratio={keep_ratio}")
#         typed_fields = parse_typed_fields(prompt_template)

#         if len(args) != len(typed_fields):
#             raise ValueError(
#                 f"Expected {len(typed_fields)} column(s) for fields {[f[0] for f in typed_fields]}, "
#                 f"but got {len(args)}."
#             )
        
#         # Build initial data dictionary
#         data_dict = {}
#         for i, (field_name, field_type) in enumerate(typed_fields):
#             arg = args[i]
#             if isinstance(arg, pd.DataFrame):
#                 data_dict[field_name] = arg.values.tolist()
#             elif isinstance(arg, pd.Series):
#                 data_dict[field_name] = arg.tolist()
#             else:
#                 data_dict[field_name] = list(arg)

#         # Create DataFrame
#         merged_df = pd.DataFrame(data_dict)
        
#         # Apply reordering if function is provided
#         if reorder_function is not None:
#             print(f"\n🔄 Applying reorder function to DataFrame with shape {merged_df.shape}")
#             print(f"Original column order: {list(merged_df.columns)}")
            
#             reordered_df, reordered_columns = reorder_function(merged_df)
            
#             print(f"Reordered column order: {reordered_columns}")
#             print(f"Reordered DataFrame shape: {reordered_df.shape}")
            
#             merged_df = reordered_df
#         else:
#             # If no reordering, maintain original column order
#             reordered_columns = list(merged_df.columns)
        
#         # Convert to records for processing
#         fields_list = merged_df.to_dict('records')
        
#         outputs = execute_batch_pope_with_pruned_embeddings(
#             modelname="/data/models/llava-1.5-7b-hf",
#             fields=fields_list,
#             query=prompt_template,
#             keep_ratio=keep_ratio,
#             typed_fields=typed_fields,
#             reordered_columns=reordered_columns,  
#             system_prompt=DEFAULT_SYSTEM_PROMPT,
#             guided_choice=["Yes", "No"],
#             base_url="http://localhost:8000/v1"
#         )
        
#         return pd.Series(outputs)
    
#     return llm_udf_embedding_batch


# def extract_image_binary_from_pope_data(image_data):
#     """Extract image binary from POPE data format."""
#     if isinstance(image_data, (list, tuple)):
#         return image_data[0] if len(image_data) > 0 else image_data
#     return image_data


# def run_experiment(
#     keep_ratio: float, 
#     dataset_name: str = "POPE_random",
#     reorder_function: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]] = None
# ) -> Tuple[str, float]:
#     """Run experiment with specific keep_ratio and save results."""
#     print(f"\n{'='*80}")
#     print(f"Running experiment with keep_ratio={keep_ratio}")
#     print(f"{'='*80}\n")
    
#     start_time = time.time()
    
#     # Register UDF with current keep_ratio
#     llm_udf = create_llm_udf_with_embeddings(keep_ratio, reorder_function=reorder_function)
#     spark.udf.register("LLM", llm_udf)
    
#     # Read POPE parquet
#     POPE_PATH = "/home/haikai/haikai/entropyTest/POPE.parquet"
#     pope_df = spark.read.parquet(POPE_PATH)
#     pope_df.createOrReplaceTempView("pope")
#     print(f"Total records: {pope_df.count()}")
    
#     # Execute query with proper column references
#     result_df = spark.sql("""
#         SELECT 
#             id,
#             question_id,
#             question,
#             answer,
#             image_source,
#             LLM('Given the text: {text:question} and image: {image:image} give me the answer to the question', question, image) as predicted
#         FROM pope
#     """)
    
#     # Normalize both answer and predicted columns for comparison
#     result_df_with_comparison = result_df.withColumn(
#         "is_correct",
#         when(
#             lower(trim(col("predicted"))) == lower(trim(col("answer"))),
#             1
#         ).otherwise(0)
#     )
    
#     # Write results to CSV
#     output_path = f"./{dataset_name}_{keep_ratio}.csv"
#     result_df_with_comparison.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"\nExecution time for keep_ratio={keep_ratio}: {execution_time:.2f} seconds")
    
#     # Write execution time to text file
#     time_log_path = f"./{dataset_name}_{keep_ratio}_execution_time.txt"
#     with open(time_log_path, 'w') as f:
#         f.write(f"Experiment Configuration\n")
#         f.write(f"{'='*50}\n")
#         f.write(f"Dataset: {dataset_name}\n")
#         f.write(f"Keep Ratio: {keep_ratio}\n")
#         f.write(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
#         f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
#         f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
#         f.write(f"{'='*50}\n")
    
#     print(f"Execution time logged to: {time_log_path}")
    
#     return output_path, execution_time

# def example_reorder_function(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
#     # Convert unhashable types (lists, arrays) to strings for reordering
#     df_for_reorder = df.copy()
#     unhashable_cols = []
    
#     # Identify columns with unhashable types and convert them
#     for col in df_for_reorder.columns:
#         try:
#             # Try to check if column is hashable
#             df_for_reorder[col].nunique()
#         except TypeError:
#             # If TypeError occurs, convert to string representation
#             unhashable_cols.append(col)
#             df_for_reorder[col] = df_for_reorder[col].apply(
#                 lambda x: str(x) if isinstance(x, (list, dict, np.ndarray)) else x
#             )
    
#     # Perform reordering on the string-converted DataFrame
#     reordered_df_temp, _ = QuickGreedy().reorder(
#         df_for_reorder,
#         early_stop=100000,
#         col_merge=[],
#         one_way_dep=[],
#         distinct_value_threshold=0.7,
#     )
    
#     # Extract the column order from the reordered dataframe
#     final_column_order = list(reordered_df_temp.columns)
    
#     # Apply the same column order to the ORIGINAL dataframe (preserving data types)
#     # But also apply the row reordering
#     reordered_df_original = df_for_reorder[final_column_order].copy()
    
#     # Restore original data types for unhashable columns
#     for col in unhashable_cols:
#         if col in final_column_order:
#             reordered_df_original[col] = df[col].values
    
#     return reordered_df_original, final_column_order


# def calculate_accuracy(csv_path: str, keep_ratio: float) -> float:
#     """Read CSV and calculate accuracy."""
#     # Find the actual CSV file in the directory (Spark creates a folder)
#     if os.path.isdir(csv_path):
#         csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv') and not f.startswith('.')]
#         if csv_files:
#             actual_csv = os.path.join(csv_path, csv_files[0])
#         else:
#             print(f"Warning: No CSV file found in {csv_path}")
#             return None
#     else:
#         actual_csv = csv_path
    
#     # Read CSV
#     df = pd.read_csv(actual_csv)
    
#     # Calculate accuracy
#     total = len(df)
#     correct = df['is_correct'].sum()
#     accuracy = (correct / total) * 100 if total > 0 else 0
    
#     print(f"\n{'='*80}")
#     print(f"Results for keep_ratio={keep_ratio}")
#     print(f"{'='*80}")
#     print(f"Total samples: {total}")
#     print(f"Correct predictions: {correct}")
#     print(f"Accuracy: {accuracy:.2f}%")
#     print(f"{'='*80}\n")
    
#     return accuracy


# # Main execution
# if __name__ == "__main__":
#     keep_ratios = [1, 0.056, 0.111, 0.222]
#     dataset_name = "POPE_V1"
    
#     overall_start = time.time()
#     results = {}
#     execution_times = {}
    
#     for keep_ratio in keep_ratios:
#         output_path, exec_time = run_experiment(
#             keep_ratio, 
#             dataset_name,
#             reorder_function=example_reorder_function  # Pass your reorder function here
#         )
        
#         execution_times[keep_ratio] = exec_time
        
#         accuracy = 0
#         results[keep_ratio] = accuracy
    
#     overall_end = time.time()
#     overall_time = overall_end - overall_start
    
#     # Write summary to text file
#     summary_path = f"./{dataset_name}_summary.txt"
#     with open(summary_path, 'w') as f:
#         f.write(f"EXPERIMENT SUMMARY\n")
#         f.write(f"{'='*80}\n")
#         f.write(f"Dataset: {dataset_name}\n")
#         f.write(f"Total Overall Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
#         f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}\n")
#         f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}\n")
#         f.write(f"\n{'='*80}\n")
#         f.write(f"DETAILED RESULTS\n")
#         f.write(f"{'='*80}\n\n")
#         f.write(f"{'Keep Ratio':<15} {'Accuracy':<15} {'Exec Time (s)':<20} {'Status':<15}\n")
#         f.write(f"{'-'*65}\n")
#         for keep_ratio in keep_ratios:
#             accuracy = results.get(keep_ratio)
#             exec_time = execution_times.get(keep_ratio, 0)
#             if accuracy is not None:
#                 f.write(f"{keep_ratio:<15.3f} {accuracy:<15.2f}% {exec_time:<20.2f} {'✓':<15}\n")
#             else:
#                 f.write(f"{keep_ratio:<15.3f} {'N/A':<15} {exec_time:<20.2f} {'✗':<15}\n")
#         f.write(f"{'='*80}\n")
    
#     print(f"\nSummary saved to: {summary_path}")
    
#     # Print summary to console
#     print(f"\n{'='*80}")
#     print("FINAL SUMMARY")
#     print(f"{'='*80}")
#     print(f"Dataset: {dataset_name}")
#     print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
#     print(f"{'Keep Ratio':<15} {'Accuracy':<15} {'Exec Time (s)':<20} {'Status':<15}")
#     print(f"{'-'*65}")
#     for keep_ratio in keep_ratios:
#         accuracy = results.get(keep_ratio)
#         exec_time = execution_times.get(keep_ratio, 0)
#         if accuracy is not None:
#             print(f"{keep_ratio:<15.3f} {accuracy:<15.2f}% {exec_time:<20.2f} {'✓':<15}")
#         else:
#             print(f"{keep_ratio:<15.3f} {'N/A':<15} {exec_time:<20.2f} {'✗':<15}")
#     print(f"{'='*80}\n")
    
#     spark.stop()


import sys
import os
import time
import pandas as pd
from typing import Tuple, List, Dict, Any, Callable
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, lower, trim
from pyspark.sql.types import StringType
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.utils import _generate_prompt
from util.quick_greedy import *

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

# Global variable to track reorder timing across batches
REORDER_TIMES = []

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


def execute_batch_pope_with_pruned_embeddings(
    modelname: str,
    fields: List[Dict[str, Any]],
    query: str,
    keep_ratio: float,
    typed_fields: List[Tuple[str, str]],
    reordered_columns: List[str],  # NEW: List of columns in reordered sequence
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
) -> List[str]:
    # Load models at the beginning
    vision_tower, model, _ = load_vision_models(device='cuda:0')
    
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
                            reduced_tokens = getOriginalVisualToken(
                                model,
                                vision_tower,
                                image_binary
                            )
                        else:
                            reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
                                model,
                                vision_tower,
                                image_binary,
                                user_prompt,
                                keep_ratio=keep_ratio
                            )
                        
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
            # Send requests
            for i, prompt in enumerate(prompts):
                response = post_http_request_with_embeds(
                    modelname,
                    [prompt],
                    temperature=0,
                    api_url=(base_url + "/chat/completions"),
                    guided_choice=guided_choice,
                    image_embeddings=[all_pruned_embeddings[i]] if all_pruned_embeddings[i] is not None else None
                )
                
                request_output = json.loads(response.content)
                choices = request_output.get('choices', [])
                
                if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                    outputs.append(choices[0]['message']['content'])
                else:
                    outputs.append(None)
            
            return outputs
    
    finally:
        # Always clean up models, even if an error occurred
        cleanup_vision_models(vision_tower, model)


def create_llm_udf_with_embeddings(
    keep_ratio: float,
    reorder_function: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]] = None
):
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
        
        # Apply reordering if function is provided
        if reorder_function is not None:
            print(f"\n🔄 Applying reorder function to DataFrame with shape {merged_df.shape}")
            print(f"Original column order: {list(merged_df.columns)}")
            
            # Time the reorder function
            reorder_start = time.time()
            reordered_df, reordered_columns = reorder_function(merged_df)
            reorder_end = time.time()
            reorder_time = reorder_end - reorder_start
            
            # Store timing globally
            global REORDER_TIMES
            REORDER_TIMES.append(reorder_time)
            
            print(f"Reordered column order: {reordered_columns}")
            print(f"Reordered DataFrame shape: {reordered_df.shape}")
            print(f"⏱️  Reorder function took: {reorder_time:.2f} seconds")
            
            merged_df = reordered_df
        else:
            # If no reordering, maintain original column order
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


def run_experiment(
    keep_ratio: float, 
    dataset_name: str = "POPE_random",
    reorder_function: Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]] = None
) -> Tuple[str, float, float]:
    """Run experiment with specific keep_ratio and save results."""
    global REORDER_TIMES
    REORDER_TIMES = []  # Reset for each experiment
    
    print(f"\n{'='*80}")
    print(f"Running experiment with keep_ratio={keep_ratio}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Register UDF with current keep_ratio
    llm_udf = create_llm_udf_with_embeddings(keep_ratio, reorder_function=reorder_function)
    spark.udf.register("LLM", llm_udf)
    
    # Read POPE parquet
    POPE_PATH = "/home/haikai/haikai/entropyTest/POPE.parquet"
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
    
    # Calculate total reorder time
    total_reorder_time = sum(REORDER_TIMES)
    num_reorder_calls = len(REORDER_TIMES)
    avg_reorder_time = total_reorder_time / num_reorder_calls if num_reorder_calls > 0 else 0
    
    print(f"\n⏱️  TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total reorder time: {total_reorder_time:.2f} seconds ({total_reorder_time/60:.2f} minutes)")
    print(f"Number of reorder calls: {num_reorder_calls}")
    print(f"Average reorder time per call: {avg_reorder_time:.2f} seconds")
    print(f"Reorder time percentage: {(total_reorder_time/execution_time)*100:.2f}%")
    print(f"{'='*80}\n")
    
    # Write execution time to text file
    time_log_path = f"./{dataset_name}_{keep_ratio}_execution_time.txt"
    with open(time_log_path, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Keep Ratio: {keep_ratio}\n")
        f.write(f"\nTIMING DETAILS\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
        f.write(f"Total Reorder Time: {total_reorder_time:.2f} seconds ({total_reorder_time/60:.2f} minutes)\n")
        f.write(f"Number of Reorder Calls: {num_reorder_calls}\n")
        f.write(f"Average Reorder Time per Call: {avg_reorder_time:.2f} seconds\n")
        f.write(f"Reorder Time Percentage: {(total_reorder_time/execution_time)*100:.2f}%\n")
        f.write(f"\nTIMESTAMPS\n")
        f.write(f"{'='*50}\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"{'='*50}\n")
    
    print(f"Execution time logged to: {time_log_path}")
    
    return output_path, execution_time, total_reorder_time

def example_reorder_function(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Convert unhashable types (lists, arrays) to strings for reordering
    df_for_reorder = df.copy()
    unhashable_cols = []
    
    # Identify columns with unhashable types and convert them
    for col in df_for_reorder.columns:
        try:
            # Try to check if column is hashable
            df_for_reorder[col].nunique()
        except TypeError:
            # If TypeError occurs, convert to string representation
            unhashable_cols.append(col)
            df_for_reorder[col] = df_for_reorder[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict, np.ndarray)) else x
            )
    
    # Perform reordering on the string-converted DataFrame
    reordered_df_temp, _ = QuickGreedy().reorder(
        df_for_reorder,
        early_stop=100000,
        col_merge=[],
        one_way_dep=[],
        distinct_value_threshold=0.7,
    )
    
    # Extract the column order from the reordered dataframe
    final_column_order = list(reordered_df_temp.columns)
    
    # Apply the same column order to the ORIGINAL dataframe (preserving data types)
    # But also apply the row reordering
    reordered_df_original = df_for_reorder[final_column_order].copy()
    
    # Restore original data types for unhashable columns
    for col in unhashable_cols:
        if col in final_column_order:
            reordered_df_original[col] = df[col].values
    
    return reordered_df_original, final_column_order


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
    keep_ratios = [1, 0.056, 0.111, 0.222]
    dataset_name = "POPE_V1"
    
    overall_start = time.time()
    results = {}
    execution_times = {}
    reorder_times = {}
    
    for keep_ratio in keep_ratios:
        output_path, exec_time, reorder_time = run_experiment(
            keep_ratio, 
            dataset_name,
            reorder_function=example_reorder_function  # Pass your reorder function here
        )
        
        execution_times[keep_ratio] = exec_time
        reorder_times[keep_ratio] = reorder_time
        
        accuracy = 0
        results[keep_ratio] = accuracy
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    total_reorder_time_all = sum(reorder_times.values())
    
    # Write summary to text file
    summary_path = f"./{dataset_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EXPERIMENT SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Overall Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
        f.write(f"Total Reorder Time (All Experiments): {total_reorder_time_all:.2f} seconds ({total_reorder_time_all/60:.2f} minutes)\n")
        f.write(f"Reorder Time Percentage: {(total_reorder_time_all/overall_time)*100:.2f}%\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"DETAILED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Keep Ratio':<12} {'Accuracy':<12} {'Exec Time (s)':<15} {'Reorder Time (s)':<18} {'Status':<10}\n")
        f.write(f"{'-'*75}\n")
        for keep_ratio in keep_ratios:
            accuracy = results.get(keep_ratio)
            exec_time = execution_times.get(keep_ratio, 0)
            reorder_time = reorder_times.get(keep_ratio, 0)
            if accuracy is not None:
                f.write(f"{keep_ratio:<12.3f} {accuracy:<12.2f}% {exec_time:<15.2f} {reorder_time:<18.2f} {'✓':<10}\n")
            else:
                f.write(f"{keep_ratio:<12.3f} {'N/A':<12} {exec_time:<15.2f} {reorder_time:<18.2f} {'✗':<10}\n")
        f.write(f"{'='*80}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Total reorder time: {total_reorder_time_all:.2f} seconds ({total_reorder_time_all/60:.2f} minutes)")
    print(f"Reorder time percentage: {(total_reorder_time_all/overall_time)*100:.2f}%\n")
    print(f"{'Keep Ratio':<12} {'Accuracy':<12} {'Exec Time (s)':<15} {'Reorder Time (s)':<18} {'Status':<10}")
    print(f"{'-'*75}")
    for keep_ratio in keep_ratios:
        accuracy = results.get(keep_ratio)
        exec_time = execution_times.get(keep_ratio, 0)
        reorder_time = reorder_times.get(keep_ratio, 0)
        if accuracy is not None:
            print(f"{keep_ratio:<12.3f} {accuracy:<12.2f}% {exec_time:<15.2f} {reorder_time:<18.2f} {'✓':<10}")
        else:
            print(f"{keep_ratio:<12.3f} {'N/A':<12} {exec_time:<15.2f} {reorder_time:<18.2f} {'✗':<10}")
    print(f"{'='*80}\n")
    
    spark.stop()