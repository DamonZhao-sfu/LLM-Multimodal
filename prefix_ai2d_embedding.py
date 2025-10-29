import sys
import os
import time
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, lower, trim
from pyspark.sql.types import StringType
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.utils import _generate_prompt

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

def extract_image_binary_from_pope_data(image_data):
    """Extract image binary from POPE data format."""
    if isinstance(image_data, dict):
        return image_data['bytes']
    if isinstance(image_data, (list, tuple)):
        return image_data[0] if len(image_data) > 0 else image_data
    return image_data


def preprocess_and_cache_pruned_embeddings(
    df: pd.DataFrame,
    image_column: str,
    question_column: str,
    image_id_column: str,
    keep_ratio: float,
    device: str = 'cuda:0'
) -> Dict[str, Dict]:
    # Load models once for all preprocessing
    print("Loading vision models...")
    vision_tower, model = load_vision_models(device=device)
    
    try:
        # Group questions by image
        print(f"\nGrouping questions by {image_id_column}...")
        image_groups = df.groupby(image_id_column)
        unique_images = len(image_groups)
        
        print(f"Found {unique_images} unique images")
        print(f"Total questions in dataset: {len(df)}")
        
        # Display image distribution
        image_counts = df.groupby(image_id_column).size().sort_values(ascending=False)
        print(f"\nTop 10 most frequently used images:")
        for img_id, count in image_counts.head(10).items():
            print(f"  {img_id}: {count} questions")
        
        print("\n" + "-" * 80)
        
        pruned_cache = {}
        total_pruning_time = 0
        successful_prunes = 0
        failed_prunes = 0
        
        # Process each unique image
        for image_idx, (image_id, image_group) in enumerate(image_groups):
            num_questions = len(image_group)
            
            print(f"\n[{image_idx + 1}/{unique_images}] Processing image: {image_id}")
            print(f"  Number of questions: {num_questions}")
            
            try:
                # Get image data (same for all rows with this image_id)
                image_data = image_group.iloc[0][image_column]
                # Extract image binary
                image_binary = extract_image_binary_from_pope_data(image_data)
                
                # Collect all questions for this image
                all_questions = image_group[question_column].tolist()
                
                # Create combined guidance prompt
                questions_text = ", ".join([f'"{q}"' for q in all_questions])
                combined_guidance = (
                    f"Extract the image's key information based on the below questions: "
                    f"{questions_text}"
                )
                
                print(f"  Combined guidance length: {len(combined_guidance)} chars")
                print(f"  Sample questions: {all_questions[:2]}...")
                
                # Prune image with combined guidance
                print(f"  🔧 Pruning image...")
                prune_start = time.time()
                
                if keep_ratio == 1:
                    # No pruning, use original tokens
                    reduced_tokens = getOriginalVisualToken(
                        model,
                        vision_tower,
                        image_binary
                    )
                else:
                    # Prune with combined guidance
                    reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
                        model,
                        vision_tower,
                        image_binary,
                        combined_guidance,
                        keep_ratio=keep_ratio,
                        important_ratio=0.6,
                        recovery_ratio=0.0  # Adjust as needed
                    )
                
                prune_end = time.time()
                prune_time = prune_end - prune_start
                total_pruning_time += prune_time
                
                # Cache the pruned embedding
                pruned_cache[image_id] = {
                    'embedding': reduced_tokens.to(torch.float16),
                    'prune_time': prune_time,
                    'original_tokens': 576,  # LLaVA default
                    'pruned_tokens': reduced_tokens.shape[1],
                    'num_questions': num_questions,
                    'guidance_length': len(combined_guidance)
                }
                
                successful_prunes += 1
                
                print(f"  ✅ Pruning successful!")
                print(f"  📊 Original: 576 tokens → Pruned: {reduced_tokens.shape[1]} tokens")
                print(f"  📉 Reduction: {((576 - reduced_tokens.shape[1]) / 576 * 100):.1f}%")
                print(f"  ⏱️  Time: {prune_time:.2f}s")
                
            except Exception as e:
                failed_prunes += 1
                print(f"  ❌ Error pruning image {image_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"✅ Successfully pruned: {successful_prunes} images")
        print(f"❌ Failed to prune: {failed_prunes} images")
        print(f"⏱️  Total pruning time: {total_pruning_time:.2f}s")
        print(f"⏱️  Average time per image: {total_pruning_time / successful_prunes:.2f}s")
        print("=" * 80)
        
        return pruned_cache, total_pruning_time
    
    finally:
        # Clean up models
        print("\nCleaning up vision models...")
        cleanup_vision_models(vision_tower, model)
        print("Cleanup complete.")


def inference_with_cached_embeddings(
    modelname: str,
    fields: List[Dict[str, Any]],
    query: str,
    typed_fields: List[Tuple[str, str]],
    embedding_cache: Dict[str, Dict],
    image_source_mapping: Dict[int, str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
) -> List[str]:
    user_prompts = []
    all_pruned_embeddings = []
    
    # Build prompts and retrieve cached embeddings
    for idx, field_dict in enumerate(fields):
        user_prompt = query
        pruned_embedding = None
        
        image_source = image_source_mapping.get(idx)
        
        for field_name, field_type in typed_fields:
            placeholder = f"{{{field_type}:{field_name}}}"
            
            if field_type == "text":
                value = field_dict.get(field_name, "")
                user_prompt = user_prompt.replace(placeholder, str(value))
            
            elif field_type == "image":
                user_prompt = user_prompt.replace(placeholder, "[image]")
                
                # Retrieve cached embedding instead of processing image
                if image_source and image_source in embedding_cache:
                    pruned_embedding = embedding_cache[image_source]['embedding']
                else:
                    print(f"Warning: No cached embedding found for image_id: {image_source}")
                    pruned_embedding = None
        
        user_prompts.append(user_prompt)
        all_pruned_embeddings.append(pruned_embedding)
    
    # Generate full prompts
    prompts = [
        _generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt)
        for user_prompt in user_prompts
    ]
    
    # Send requests to API
    outputs = []
    if base_url:
        for i, prompt in enumerate(prompts):
            # Get the guided_choice for this specific row
            
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


def create_llm_udf_with_cached_embeddings(embedding_cache: Dict[str, Dict], image_source_mapping: Dict[int, str]):
    @pandas_udf(StringType())
    def llm_udf_cached_embeddings(
        prompts: pd.Series,
        *args: pd.Series
    ) -> pd.Series:
        prompt_template = prompts.iloc[0]
        print(f"Processing batch on PID {os.getpid()} with cached embeddings")
        
        typed_fields = parse_typed_fields(prompt_template)
        
        # The last argument should be the options column (guided_choice)
        # All other arguments are the typed fields
        if len(args) != len(typed_fields) + 1:
            raise ValueError(
                f"Expected {len(typed_fields) + 1} column(s) (fields + options), "
                f"but got {len(args)}."
            )
        
        # Separate the options column from other fields
        field_args = args[:-1]  # All except last
        options_series = args[-1]  # Last one is options
        
        data_dict = {}
        for i, (field_name, field_type) in enumerate(typed_fields):
            arg = field_args[i]
            if isinstance(arg, pd.DataFrame):
                data_dict[field_name] = arg.values.tolist()
            elif isinstance(arg, pd.Series):
                data_dict[field_name] = arg.tolist()
            else:
                data_dict[field_name] = list(arg)
        
        merged_df = pd.DataFrame(data_dict)
        fields_list = merged_df.to_dict('records')
        
        # Convert options series to list of lists
        # Handle both string representation of lists and actual lists
        guided_choices = []
        for opt in options_series:
            if isinstance(opt, str):
                # If it's a string representation of a list, evaluate it
                try:
                    import ast
                    guided_choices.append(ast.literal_eval(opt))
                except:
                    # If parsing fails, split by comma
                    guided_choices.append([x.strip() for x in opt.split(',')])
            elif isinstance(opt, list):
                guided_choices.append(opt)
            else:
                guided_choices.append(None)
        
        outputs = inference_with_cached_embeddings(
            modelname="llava-hf/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            typed_fields=typed_fields,
            embedding_cache=embedding_cache,
            image_source_mapping=image_source_mapping,
            guided_choice=["0","1","2","3"],
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            base_url="http://localhost:8000/v1"
        )
        
        return pd.Series(outputs)
    
    return llm_udf_cached_embeddings

def run_experiment_with_cached_embeddings(
    keep_ratio: float,
    pope_pandas_df: pd.DataFrame,
    dataset_name: str = "POPE_image_prefix"
) -> Tuple[str, float]:
    """Run experiment with cached embeddings for a specific keep_ratio."""
    print(f"\n{'='*80}")
    print(f"Running experiment with keep_ratio={keep_ratio}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Preprocess and cache pruned embeddings
    print(f"Preprocessing images with keep_ratio={keep_ratio}...")
    embedding_cache, total_pruning_time = preprocess_and_cache_pruned_embeddings(
        df=pope_pandas_df,
        image_column='image',
        question_column='question',
        image_id_column='image_id',
        keep_ratio=keep_ratio,
        device='cuda:0'
    )
    
    # Create image source mapping
    image_source_mapping_reordered = {
        idx: row['image_id'] 
        for idx, row in pope_pandas_df.iterrows()
    }
    
    # Create and register UDF
    llm_udf = create_llm_udf_with_cached_embeddings(
        embedding_cache, 
        image_source_mapping_reordered
    )
    spark.udf.register("LLM", llm_udf)
    
    # Execute query - now pass options as the last argument
    result_df = spark.sql("""
        SELECT 
            answer,
            options,
            LLM('Given the question: {text:question} and the image: {image:image}. The candidate answers are: {text:options}. Return only the index number (0, 1, 2, or 3) of the correct answer.', 
                question, image, options, options) as predicted
        FROM pope
    """)
    
    # UDF to find the index of predicted answer in options
    from pyspark.sql.functions import udf, array_position
    from pyspark.sql.types import IntegerType
    import ast
    
    @udf(IntegerType())
    def get_answer_index(predicted, options):
        """Find the index (0-based) of the predicted answer in the options list."""
        try:
            # Parse options if it's a string
            if isinstance(options, str):
                options_list = ast.literal_eval(options)
            else:
                options_list = options
            
            # Clean predicted answer
            predicted_clean = str(predicted).strip().lower() if predicted else ""
            
            # Find matching option
            for idx, option in enumerate(options_list):
                if str(option).strip().lower() == predicted_clean:
                    return idx
            
            # If no match found, return -1
            return -1
        except:
            return -1
    
    result_df_with_comparison = result_df.withColumn(
        "is_correct",
        when(col("predicted") == col("answer"), 1).otherwise(0)
    ).drop("options")
    
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
        f.write(f"Total Pruning Time: {total_pruning_time:.2f} seconds ({total_pruning_time/60:.2f} minutes)\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"{'='*50}\n")
    
    print(f"Execution time logged to: {time_log_path}")
    
    return output_path, execution_time


def calculate_accuracy(csv_path: str, keep_ratio: float) -> float:
    """Read CSV and calculate accuracy with detailed metrics."""
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
    
    # Count invalid predictions (where predicted_index is -1)
    invalid_predictions = (df['predicted_index'] == -1).sum()
    
    # Create confusion-like statistics
    print(f"\n{'='*80}")
    print(f"Results for keep_ratio={keep_ratio}")
    print(f"{'='*80}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Invalid predictions (not in options): {invalid_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Show distribution of predictions by index
    if 'predicted_index' in df.columns and 'answer_index' in df.columns:
        print(f"\n{'-'*80}")
        print("Prediction Distribution:")
        print(f"{'-'*80}")
        pred_dist = df['predicted_index'].value_counts().sort_index()
        for idx, count in pred_dist.items():
            percentage = (count / total) * 100
            print(f"  Predicted index {idx}: {count} times ({percentage:.1f}%)")
        
        print(f"\n{'-'*80}")
        print("Correct Answer Distribution:")
        print(f"{'-'*80}")
        answer_dist = df['answer_index'].value_counts().sort_index()
        for idx, count in answer_dist.items():
            percentage = (count / total) * 100
            correct_for_idx = df[(df['answer_index'] == idx) & (df['is_correct'] == 1)].shape[0]
            accuracy_for_idx = (correct_for_idx / count * 100) if count > 0 else 0
            print(f"  Answer index {idx}: {count} times ({percentage:.1f}%) - Accuracy: {accuracy_for_idx:.1f}%")
        
        # Confusion matrix style output
        print(f"\n{'-'*80}")
        print("Confusion Matrix (Answer Index vs Predicted Index):")
        print(f"{'-'*80}")
        confusion_data = df.groupby(['answer_index', 'predicted_index']).size().reset_index(name='count')
        for _, row in confusion_data.iterrows():
            print(f"  Answer: {int(row['answer_index'])} -> Predicted: {int(row['predicted_index'])}: {int(row['count'])} times")
    
    print(f"{'='*80}\n")
    
    return accuracy


# Main execution
if __name__ == "__main__":
    keep_ratios = [0.056, 0.111, 0.222, 1]
    dataset_name = "AI2D_image_prefix"
    
    overall_start = time.time()
    
    # Read both POPE parquet files and union them
    POPE_PATH_1 = "/scratch/hpc-prf-haqc/haikai/dataset/ai2d/test-00000-of-00002.parquet"
    POPE_PATH_2 = "/scratch/hpc-prf-haqc/haikai/dataset/ai2d/test-00001-of-00002.parquet"
    
    pope_df_1 = spark.read.parquet(POPE_PATH_1)
    pope_df_2 = spark.read.parquet(POPE_PATH_2)
    pope_df = pope_df_1.union(pope_df_2)
    
    from pyspark.sql.window import Window
    from pyspark.sql.functions import dense_rank
    
    # Assuming there's an 'image' column that identifies the image
    # Create a window specification to rank images
    window_spec = Window.orderBy("image")
    pope_df = pope_df.withColumn("image_id", dense_rank().over(window_spec))
    
    pope_df.createOrReplaceTempView("pope")
    print(f"Total records: {pope_df.count()}")
    print(f"Unique images: {pope_df.select('image_id').distinct().count()}")
    
    # Convert to pandas once
    pope_pandas_df = pope_df.toPandas()
    
    results = {}
    execution_times = {}
    
    # Run experiments for each keep_ratio
    for keep_ratio in keep_ratios:
        output_path, exec_time = run_experiment_with_cached_embeddings(
            keep_ratio=keep_ratio,
            pope_pandas_df=pope_pandas_df,
            dataset_name=dataset_name
        )
        
        execution_times[keep_ratio] = exec_time
        
        # Calculate accuracy
        accuracy = 0
        results[keep_ratio] = accuracy
        
        # Clear GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
    
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