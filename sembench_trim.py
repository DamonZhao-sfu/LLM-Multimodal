import sys
import os
import time
import csv
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from util.utils import _generate_prompt
import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, udf
from pyspark.sql.types import StringType, ArrayType, FloatType
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.cdpruner import *
from util.trimTokenator import *
from util.visual_util import *
from transformers import CLIPProcessor, CLIPModel
import pickle

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

AQP_SIMILARITY_THRESHOLD = 0.2
_aqp_model = None
_aqp_processor = None


def get_aqp_model(device='cuda'):
    global _aqp_model, _aqp_processor
    if _aqp_model is None:
        print("Loading AQP Proxy Model (CLIP)...")
        model_id = "/scratch/hpc-prf-haqc/haikai/clip-vit-base-patch32"
        _aqp_model = CLIPModel.from_pretrained(model_id).to(device)
        _aqp_processor = CLIPProcessor.from_pretrained(model_id)
        _aqp_model.eval()
    return _aqp_model, _aqp_processor


def compute_text_embedding(text: str) -> np.ndarray:
    """Compute CLIP text embedding for a single text string."""
    model, processor = get_aqp_model()
    
    # Truncate text to fit CLIP context length
    truncated_text = text[:77] if text else ""
    
    inputs = processor(
        text=[truncated_text],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
        # Normalize
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    return text_embeds.cpu().numpy().flatten()


def compute_image_embedding(image_binary: bytes) -> np.ndarray:
    """Compute CLIP image embedding for a single image."""
    model, processor = get_aqp_model()
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_binary))
    
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    
    return image_embeds.cpu().numpy().flatten()


def compute_similarity_from_embeddings(text_embedding: np.ndarray, 
                                      image_embedding: np.ndarray) -> float:
    """Compute cosine similarity between precomputed embeddings."""
    # Both embeddings are already normalized, so dot product = cosine similarity
    similarity = np.dot(text_embedding, image_embedding)
    return float(similarity)


def preprocess_text_embeddings(spark: SparkSession, 
                               table_name: str,
                               text_columns: List[str],
                               output_path: str) -> None:
    """
    Precompute text embeddings for specified columns and save as intermediate table.
    
    Args:
        spark: SparkSession
        table_name: Name of the table to process
        text_columns: List of text column names to embed
        output_path: Path to save the embeddings table
    """
    print(f"Preprocessing text embeddings for table: {table_name}")
    
    df = spark.table(table_name)
    
    # Create UDF for text embedding
    @udf(ArrayType(FloatType()))
    def embed_text_udf(text: str) -> List[float]:
        if text is None or text == "":
            return [0.0] * 512  # CLIP embedding dimension
        embedding = compute_text_embedding(text)
        return embedding.tolist()
    
    # Add embedding columns for each text field
    for col_name in text_columns:
        embedding_col = f"{col_name}_embedding"
        df = df.withColumn(embedding_col, embed_text_udf(col(col_name)))
    
    # Save as parquet for efficient storage
    df.write.mode("overwrite").parquet(output_path)
    print(f"Text embeddings saved to: {output_path}")


def preprocess_image_embeddings(spark: SparkSession,
                               table_name: str,
                               image_column: str,
                               output_path: str) -> None:
    """
    Precompute image embeddings and save as intermediate table.
    
    Args:
        spark: SparkSession
        table_name: Name of the table containing images
        image_column: Name of the image filepath column
        output_path: Path to save the embeddings table
    """
    print(f"Preprocessing image embeddings for table: {table_name}")
    
    df = spark.table(table_name)
    
    # Create UDF for image embedding
    @udf(ArrayType(FloatType()))
    def embed_image_udf(image_path: str) -> List[float]:
        if image_path is None or image_path == "":
            return [0.0] * 512  # CLIP embedding dimension
        try:
            image_binary = extract_image_binary(image_path)
            embedding = compute_image_embedding(image_binary)
            return embedding.tolist()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return [0.0] * 512
    
    # Add embedding column
    embedding_col = f"{image_column}_embedding"
    df = df.withColumn(embedding_col, embed_image_udf(col(image_column)))
    
    # Save as parquet
    df.write.mode("overwrite").parquet(output_path)
    print(f"Image embeddings saved to: {output_path}")


def filter_by_precomputed_similarity(fields_list: List[Dict[str, Any]],
                                    typed_fields: List[Tuple[str, str]],
                                    reordered_columns: List[str],
                                    threshold: float = 0.2) -> Tuple[List[Dict], List[int]]:

    filtered_fields = []
    skipped_indices = []
    
    
    for idx, field_dict in enumerate(fields_list):
        text_embedding = None
        image_embedding = None
        # Extract embeddings from the field_dict
        for field_name in reordered_columns:
            # Check if this is an embedding column
            if field_name.endswith("_embedding"):
                base_name = field_name.replace("_embedding", "")
                field_type = next((ftype for fname, ftype in typed_fields if fname == base_name), None)
                
                if field_type == "text":
                    text_embedding = np.array(field_dict.get(field_name, []))
                elif field_type == "image":
                    image_embedding = np.array(field_dict.get(field_name, []))
        
        # Compute similarity if both embeddings exist
        is_relevant = True
        if text_embedding is not None and image_embedding is not None and \
           len(text_embedding) > 0 and len(image_embedding) > 0:
            similarity = compute_similarity_from_embeddings(text_embedding, image_embedding)
            if similarity < threshold:
                print(f"AQP Pruning: Row {idx} skipped. Similarity {similarity:.3f} < {threshold}")
                skipped_indices.append(idx)
                is_relevant = False
        
        if is_relevant:
            filtered_fields.append(field_dict)
    
    return filtered_fields, skipped_indices


# Spark configuration (same as before)
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

# Global variables
API_URL = "http://localhost:8000/v1/chat/completions"
RECOVERY_RATIO = 0.0
TOTAL_PRUNING_TIME = 0.0
TIMING_CSV_FILE = None
INVOCATION_COUNTER = 0


def initialize_timing_csv(keep_token: float, dataset_name: str):
    """Initialize CSV file for timing records."""
    global TIMING_CSV_FILE, INVOCATION_COUNTER
    
    INVOCATION_COUNTER = 0
    TIMING_CSV_FILE = f"./{dataset_name}_{keep_token}_timing.csv"
    
    with open(TIMING_CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['invocation_id', 'keep_token', 'preprocess_time', 'encode_time', 'prune_time', 'total_time'])
    
    print(f"Timing CSV initialized: {TIMING_CSV_FILE}")


def record_timing(keep_token: float, preprocess_time: float, encode_time: float, prune_time: float):
    """Record timing information to CSV file."""
    global TIMING_CSV_FILE, INVOCATION_COUNTER
    
    if TIMING_CSV_FILE is None:
        return
    
    INVOCATION_COUNTER += 1
    total_time = preprocess_time + encode_time + prune_time
    
    with open(TIMING_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            INVOCATION_COUNTER,
            keep_token,
            f"{preprocess_time:.6f}",
            f"{encode_time:.6f}",
            f"{prune_time:.6f}",
            f"{total_time:.6f}"
        ])


def execute_batch_pope_with_pruned_embeddings(
    modelname: str,
    fields: List[Dict[str, Any]],
    query: str,
    keep_token: int,
    typed_fields: List[Tuple[str, str]],
    reordered_columns: List[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
) -> List[str]:
    """Process batch with precomputed embeddings for AQP filtering."""
    tokenizer, model, processor = load_vision_models_llava_next(device='cuda')
    
    try:
        total_rows = len(fields)
        
        # Filter using precomputed embeddings
        filtered_fields, skipped_indices = filter_by_precomputed_similarity(
            fields, typed_fields, reordered_columns, threshold=AQP_SIMILARITY_THRESHOLD
        )
        
        print(f"AQP Filtering: {len(skipped_indices)} rows skipped out of {total_rows} ({len(skipped_indices)/total_rows*100:.1f}%)")
        
        # If all rows are skipped, return "No" for all
        if len(filtered_fields) == 0:
            return ["No"] * total_rows
        
        user_prompts = []
        all_pruned_embeddings = []
        valid_indices = []  # Track which original indices we're processing
        
        for idx, field_dict in enumerate(fields):
            # Skip rows that were filtered out by AQP
            if idx in skipped_indices:
                continue
            
            valid_indices.append(idx)
            user_prompt = ""
            current_image_binary = None
            
            for field_name in reordered_columns:
                # Skip embedding columns in prompt construction
                if field_name.endswith("_embedding"):
                    continue
                    
                field_type = next((ftype for fname, ftype in typed_fields if fname == field_name), None)
                if not field_type:
                    continue
                
                if field_type == "text":
                    val = field_dict.get(field_name, "")
                    user_prompt += f"{field_name}: {val}\n"
                elif field_type == "image":
                    user_prompt += f"{field_name}: [image]\n"
                    image_data = field_dict.get(field_name)
                    if image_data is not None:
                        current_image_binary = extract_image_binary(image_data)
            
            # Only process if we have an image (shouldn't be None for valid indices)
            if current_image_binary:
                pruned_tensor, pre_t, enc_t, pr_t = get_inputs_embeds_trim(
                    model, processor, current_image_binary, keep_token=keep_token
                )
                record_timing(keep_token, pre_t, enc_t, pr_t)
                row_embedding = pruned_tensor.detach().cpu().to(torch.float16)
                all_pruned_embeddings.append(row_embedding)
                user_prompts.append(user_prompt.strip())
        
        # Generate prompts
        prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
                   for user_prompt in user_prompts]
        
        # Run inference only on filtered rows
        inference_outputs = []
        if base_url:
            async def fetch_all():
                tasks = []
                for i, prompt in enumerate(prompts):
                    task = asyncio.to_thread(
                        post_http_request_with_embeds,
                        modelname,
                        [prompt],
                        temperature=0,
                        api_url=(base_url + "/chat/completions"),
                        guided_choice=guided_choice,
                        image_embeddings=[[all_pruned_embeddings[i]]]
                    )
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks)
                processed_outputs = []
                for response in responses:
                    try:
                        request_output = json.loads(response.content)
                        choices = request_output.get('choices', [])
                        
                        if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                            processed_outputs.append(choices[0]['message']['content'])
                        else:
                            processed_outputs.append(None)
                    except Exception as e:
                        print(f"Error processing response: {e}")
                        processed_outputs.append(None)
                
                return processed_outputs
            
            inference_outputs = asyncio.run(fetch_all())
        
        # Reconstruct full output array with skipped indices filled as "No"
        final_outputs = ["No"] * total_rows
        for i, original_idx in enumerate(valid_indices):
            if i < len(inference_outputs):
                final_outputs[original_idx] = inference_outputs[i]
        
        print(f"Inference performed on {len(inference_outputs)} rows, skipped {len(skipped_indices)} rows")
        
        return final_outputs
    
    finally:
        torch.cuda.empty_cache()


def create_llm_udf_with_embeddings(keep_token: int):
    @pandas_udf(StringType())
    def llm_udf_embedding_batch(prompts: pd.Series, *args: pd.Series) -> pd.Series:
        prompt_template = prompts.iloc[0]
        typed_fields = parse_typed_fields(prompt_template)
        
        # Expected number of columns should be doubled (original + embedding for each field)
        expected_cols = len(typed_fields) * 2
        if len(args) != expected_cols:
            raise ValueError(
                f"Expected {expected_cols} column(s) for fields {[f[0] for f in typed_fields]} "
                f"(original + embedding columns), but got {len(args)}."
            )
        
        data_dict = {}
        # Process args in pairs: original column + its embedding
        for i, (field_name, field_type) in enumerate(typed_fields):
            # Original column
            original_arg = args[i * 2]
            if isinstance(original_arg, pd.DataFrame):
                data_dict[field_name] = original_arg.values.tolist()
            elif isinstance(original_arg, pd.Series):
                data_dict[field_name] = original_arg.tolist()
            else:
                data_dict[field_name] = list(original_arg)
            
            # Embedding column
            embedding_col_name = f"{field_name}_embedding"
            embedding_arg = args[i * 2 + 1]
            if isinstance(embedding_arg, pd.DataFrame):
                data_dict[embedding_col_name] = embedding_arg.values.tolist()
            elif isinstance(embedding_arg, pd.Series):
                data_dict[embedding_col_name] = embedding_arg.tolist()
            else:
                data_dict[embedding_col_name] = list(embedding_arg)
        
        merged_df = pd.DataFrame(data_dict)
        reordered_columns = list(merged_df.columns)
        fields_list = merged_df.to_dict('records')
        
        # This now returns the full-length output with skipped rows filled as "No"
        outputs = execute_batch_pope_with_pruned_embeddings(
            modelname="llava-hf/llava-v1.6-mistral-7b-hf",
            fields=fields_list,
            query=prompt_template,
            keep_token=keep_token,
            typed_fields=typed_fields,
            reordered_columns=reordered_columns,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            guided_choice=["Yes", "No"],
            base_url="http://localhost:8000/v1"
        )
        
        # The outputs should already match the input length due to reconstruction
        # but keep this as a safety check
        num_input_rows = len(merged_df)
        if len(outputs) != num_input_rows:
            print(f"Warning: Output length mismatch. Expected {num_input_rows}, got {len(outputs)}")
            if len(outputs) < num_input_rows:
                outputs.extend(["No"] * (num_input_rows - len(outputs)))
            elif len(outputs) > num_input_rows:
                outputs = outputs[:num_input_rows]
        
        return pd.Series(outputs)
    
    return llm_udf_embedding_batch


def extract_image_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data


def run_experiment(keep_token: int, dataset_name: str = "POPE_random", 
                  precompute_embeddings: bool = True) -> Tuple[str, float]:
    """
    Run experiment with optional embedding precomputation.
    
    Args:
        keep_token: Token keep ratio
        dataset_name: Name of the dataset
        precompute_embeddings: If True, precompute and cache embeddings
    """
    initialize_timing_csv(keep_token, dataset_name)
    start_time = time.time()
    
    # Load base tables
    ap_warrior_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/scratch/hpc-prf-haqc/haikai/SemBench/data200/files/mmqa/data/sf_200/ap_warrior.csv")
    
    images_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/scratch/hpc-prf-haqc/haikai/SemBench/data200/files/mmqa/data/sf_200/thalamusdb_images.csv")
    
    tampa_international_airport_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/scratch/hpc-prf-haqc/haikai/SemBench/data200/files/mmqa/data/sf_200/tampa_international_airport.csv")
    
    ap_warrior_df.createOrReplaceTempView("ap_warrior_raw")
    images_df.createOrReplaceTempView("images_raw")
    tampa_international_airport_df.createOrReplaceTempView("tampa_international_airport_raw")
    
    # Precompute embeddings if requested
    if precompute_embeddings:
        embedding_start = time.time()
        
        # Precompute text embeddings for relevant tables
        preprocess_text_embeddings(
            spark, "ap_warrior_raw", ["Track"], 
            "./embeddings/ap_warrior_text_embeddings.parquet"
        )
        preprocess_text_embeddings(
            spark, "tampa_international_airport_raw", ["Airlines", "Destinations"],
            "./embeddings/tampa_text_embeddings.parquet"
        )
        
        # Precompute image embeddings
        preprocess_image_embeddings(
            spark, "images_raw", "image_filepath",
            "./embeddings/image_embeddings.parquet"
        )
        
        # Load embedding tables
        ap_warrior_embeddings = spark.read.parquet("./embeddings/ap_warrior_text_embeddings.parquet")
        tampa_embeddings = spark.read.parquet("./embeddings/tampa_text_embeddings.parquet")
        image_embeddings = spark.read.parquet("./embeddings/image_embeddings.parquet")
        
        # Replace original views with embedding-enhanced versions
        ap_warrior_embeddings.createOrReplaceTempView("ap_warrior")
        tampa_embeddings.createOrReplaceTempView("tampa_international_airport")
        image_embeddings.createOrReplaceTempView("images")
        ap_warrior_embeddings.show()
        
        embedding_end = time.time()
        print(f"Embedding precomputation took {embedding_end - embedding_start:.2f} seconds")
    
    # Register UDF
    llm_udf = create_llm_udf_with_embeddings(keep_token)
    spark.udf.register("LLM", llm_udf)
    
    # Execute query - MODIFIED to pass both original and embedding columns
    result_df = spark.sql("""
         SELECT t.Airlines, i.image_filepath
         FROM tampa_international_airport t, images i where LLM(
         "You will be provided with an airline name and an image. Determine for each airline if the image shows the logo of the airline. Airline: {text:Airlines} Image: {image:image_filepath} ", 
         t.Airlines, t.Airlines_embedding,
         i.image_filepath, i.image_filepath_embedding
         ) = "Yes";
         """)
    
    output_path = f"./{dataset_name}_{keep_token}.csv"
    result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    pruning_time = 0
    if TIMING_CSV_FILE and os.path.exists(TIMING_CSV_FILE):
        timing_df = pd.read_csv(TIMING_CSV_FILE)
        pruning_time = timing_df['total_time'].sum()
    
    print(f"\nExecution time for keep_token={keep_token}: {execution_time:.2f} seconds")
    print(f"Pruning time for keep_token={keep_token}: {pruning_time:.2f} seconds")
    
    # Write execution time log
    time_log_path = f"./{dataset_name}_{keep_token}_execution_time.txt"
    with open(time_log_path, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Keep Ratio: {keep_token}\n")
        f.write(f"Precomputed Embeddings: {precompute_embeddings}\n")
        f.write(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
        f.write(f"Total Pruning Time: {pruning_time:.2f} seconds ({pruning_time/60:.2f} minutes)\n")
        f.write(f"Pruning Time Percentage: {pruning_time/execution_time*100:.2f}%\n")
        f.write(f"{'='*50}\n")
    
    return output_path, execution_time, pruning_time


# Main execution
if __name__ == "__main__":
    keep_tokens = [128, 256, 512, 1024]
    dataset_name = "SEMbench_q7"
    
    overall_start = time.time()
    results = {}
    execution_times = {}
    pruning_times = {}
    
    for keep_token in keep_tokens:
        output_path, exec_time, prune_time = run_experiment(
            keep_token, 
            dataset_name,
            precompute_embeddings=True  # Enable precomputation
        )
        
        execution_times[keep_token] = exec_time
        pruning_times[keep_token] = prune_time
        results[keep_token] = 0
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    total_pruning_time = sum(pruning_times.values())
    
    # Write summary
    summary_path = f"./{dataset_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EXPERIMENT SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Overall Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
        f.write(f"Total Accumulated Pruning Time: {total_pruning_time:.2f} seconds ({total_pruning_time/60:.2f} minutes)\n")
        f.write(f"Pruning Time Percentage: {total_pruning_time/overall_time*100:.2f}%\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"DETAILED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Keep Ratio':<12} {'Exec Time (s)':<15} {'Prune Time (s)':<16} {'Prune %':<10}\n")
        f.write(f"{'-'*80}\n")
        for keep_token in keep_tokens:
            exec_time = execution_times.get(keep_token, 0)
            prune_time = pruning_times.get(keep_token, 0)
            prune_pct = (prune_time / exec_time * 100) if exec_time > 0 else 0
            f.write(f"{keep_token:<12} {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}%\n")
        f.write(f"{'='*80}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    spark.stop()