import sys
import os
import time
import csv
import pandas as pd
from typing import Tuple, List, Dict, Any
from util.utils import _generate_prompt
import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.cdpruner import *
from util.trimTokenator import *
from util.visual_util import *
from transformers import CLIPProcessor, CLIPModel
# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

AQP_SIMILARITY_THRESHOLD=0.2
_aqp_model = None
_aqp_processor = None


def get_aqp_model(device='cuda'):
    global _aqp_model, _aqp_processor
    if _aqp_model is None:
        print("Loading AQP Proxy Model (CLIP)...")
        # specific lightweight model
        model_id = "/scratch/hpc-prf-haqc/haikai/clip-vit-base-patch32" 
        _aqp_model = CLIPModel.from_pretrained(model_id).to(device)
        _aqp_processor = CLIPProcessor.from_pretrained(model_id)
        _aqp_model.eval()
    return _aqp_model, _aqp_processor


def check_aqp_similarity(image_binary, candidate_text, threshold=0.2):
    """
    Returns True if the image and text are similar enough to proceed.
    Returns False if they are too different (should be skipped).
    """
    model, processor = get_aqp_model()
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_binary))
    
    # Process inputs (Text + Image)
    # Truncate text to fit CLIP context length (77 tokens usually)
    inputs = processor(
        text=[candidate_text], 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        
        # CLIP calculates the dot product (logits_per_image)
        # We normalize specific to the model, but logits_per_image is usually the raw score
        logits_per_image = outputs.logits_per_image  # shape: [1, 1]
        probs = logits_per_image.softmax(dim=1) # standard softmax
        
        # Alternative: Normalized Cosine Similarity (more interpretable)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        similarity = torch.matmul(image_embeds, text_embeds.t()).item()

    # Decision Logic
    # 0.2 is a common heuristic threshold for CLIP ViT-B/32
    return similarity >= threshold, similarity

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

TOTAL_PRUNING_TIME = 0.0
TIMING_CSV_FILE = None
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
    global TIMING_CSV_FILE, INVOCATION_COUNTER
    
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
) -> Tuple[List[str], float]:
    """Returns tuple of (outputs, pruning_time)"""
    tokenizer, model, processor = load_vision_models_llava_next(device='cuda')
    try:
        user_prompts = []
        all_pruned_embeddings = []
        skipped_indices = []
        for idx, field_dict in enumerate(fields):
            user_prompt = ""
            
            row_embedding = None            
            current_image_binary = None
            current_text_concept = ""

            for field_name in reordered_columns:
                field_type = next((ftype for fname, ftype in typed_fields if fname == field_name), None)
                if not field_type: continue

                if field_type == "text":
                    val = field_dict.get(field_name, "")
                    user_prompt += f"{field_name}: {val}\n"
                    current_text_concept += f"{val} "
                
                elif field_type == "image":
                    user_prompt += f"{field_name}: [image]\n"
                    image_data = field_dict.get(field_name)
                    if image_data is not None:
                        current_image_binary = extract_image_binary(image_data)            

            is_relevant = True
            if current_image_binary and current_text_concept:
                is_relevant, score = check_aqp_similarity(
                    current_image_binary, 
                    current_text_concept, 
                    threshold=AQP_SIMILARITY_THRESHOLD 
                )
                
                if not is_relevant:
                    print(f"AQP Pruning: Row {idx} skipped. Similarity {score:.3f} < {AQP_SIMILARITY_THRESHOLD}")
                    skipped_indices.append(idx)
                else:
                    print(f"Similarity {score:.3f}")

            if idx not in skipped_indices and current_image_binary:
                pruned_tensor, pre_t, enc_t, pr_t = get_inputs_embeds_trim(
                    model, processor, current_image_binary, keep_ratio=keep_ratio
                )
                record_timing(keep_ratio, pre_t, enc_t, pr_t)
                row_embedding = pruned_tensor.detach().cpu().to(torch.float16)
                user_prompts.append(user_prompt.strip())
            
            if row_embedding is not None:
                all_pruned_embeddings.append(row_embedding)


        # if all_pruned_embeddings:
        #     max_len = max(emb.shape[0] for emb in all_pruned_embeddings)
        #     padded_embeddings = []
        #     for emb in all_pruned_embeddings:
        #         current_len = emb.shape[0]
        #         if current_len < max_len:
        #             # Pad the bottom with zeros: (pad_left, pad_right, pad_top, pad_bottom)
        #             pad_amount = max_len - current_len
        #             # F.pad format for 2D is (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
        #             # We want to pad dimension 0 (rows), so we pad the 2nd to last dimension.
        #             padded_emb = F.pad(emb, (0, 0, 0, pad_amount), "constant", 0)
        #             padded_embeddings.append(padded_emb)
        #         else:
        #             padded_embeddings.append(emb)
        
        #     # Replace the original list with the padded list
        #     all_pruned_embeddings = padded_embeddings

        prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
                   for user_prompt in user_prompts]
        print("len of prompts")
        print(len(prompts))
        outputs = []
        if base_url:    
            def fetch_all():
                processed_outputs = []
                for i, prompt in enumerate(prompts):
                    try:
                        print("send to inference")
                        response = post_http_request_with_embeds(
                            modelname,
                            [prompt],
                            temperature=0,
                            api_url=(base_url + "/chat/completions"),
                            guided_choice=guided_choice,
                            image_embeddings=[[all_pruned_embeddings[i]]]
                        )
                        
                        request_output = json.loads(response.content)
                        choices = request_output.get('choices', [])

                        if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                            print(choices[0]['message']['content'])
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

            outputs = fetch_all()
            return outputs
            # async def fetch_all():
            #     tasks = []
            #     for i, prompt in enumerate(prompts):
            #         task = asyncio.to_thread(
            #             post_http_request_with_embeds,
            #             modelname,
            #             [prompt],
            #             temperature=0,
            #             api_url=(base_url + "/chat/completions"),
            #             guided_choice=guided_choice,
            #             image_embeddings=[[all_pruned_embeddings[i]]]
            #         )
            #         tasks.append(task)
                
            #     responses = await asyncio.gather(*tasks)                
            #     processed_outputs = []
            #     for response in responses:
            #         try:
            #             # Assuming response has a .content attribute like a requests.Response
            #             request_output = json.loads(response.content) 
            #             choices = request_output.get('choices', [])
                        
            #             if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
            #                 processed_outputs.append(choices[0]['message']['content'])
            #             else:
            #                 # Log error or empty response
            #                 print(f"Warning: No valid content in response. Output: {request_output}")
            #                 processed_outputs.append(None)
            #         except Exception as e:
            #             # Log exception
            #             print(f"Error processing response: {e}. Response content: {getattr(response, 'content', 'N/A')}")
            #             processed_outputs.append(None)
                        
            #     return processed_outputs

            # outputs = asyncio.run(fetch_all())            
            # return outputs
    
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
            modelname="llava-hf/llava-v1.6-mistral-7b-hf",
            fields=fields_list,
            query=prompt_template,
            keep_ratio=keep_ratio,
            typed_fields=typed_fields,
            reordered_columns=reordered_columns,  # Pass reordered column sequence
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            guided_choice=["Yes", "No"],
            base_url="http://localhost:8000/v1"
        )
        
        return pd.Series(outputs)
    
    return llm_udf_embedding_batch

def extract_image_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

def run_experiment(keep_ratio: float, dataset_name: str = "POPE_random") -> Tuple[str, float]:
    # Reset pruning time for this experiment
    initialize_timing_csv(keep_ratio, dataset_name)

    start_time = time.time()
    
    # Register UDF with current keep_ratio
    llm_udf = create_llm_udf_with_embeddings(keep_ratio)
    spark.udf.register("LLM", llm_udf)
    
    ap_warrior_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/scratch/hpc-prf-haqc/haikai/SemBench/data200/files/mmqa/data/sf_200/ap_warrior.csv")

    images_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/scratch/hpc-prf-haqc/haikai/SemBench/data200/files/mmqa/data/sf_200/thalamusdb_images.csv").limit(10) \
    
    tampa_international_airport_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/scratch/hpc-prf-haqc/haikai/SemBench/data200/files/mmqa/data/sf_200/tampa_international_airport.csv") \
    

    ap_warrior_df.show()
    images_df.show()
    tampa_international_airport_df.show()
    
    ap_warrior_df.createOrReplaceTempView("ap_warrior")
    images_df.createOrReplaceTempView("images")
    tampa_international_airport_df.createOrReplaceTempView("tampa_international_airport")
    
    # Execute query with proper column references
    # Q2a
    result_df = spark.sql("""
        SELECT t.ID, i.image_filepath, LLM(
            'You will be provided with a horse racetrack name and an image.
            Determine if the image shows the logo of the racetrack.
            Racetrack: {text:Track} Image: {image:image_filepath}', t.Track, i.image_filepath
        ) FROM ap_warrior t, images i;
    """)    
    
    """
    SELECT t.ID, i.image_filepath, LLM(
            'You will be provided with a horse racetrack name and an image.
            Determine if the image shows the logo of the racetrack.
            Racetrack: {text:Track} Image: {image:image_filepath}', t.Track, i.image_filepath
        ) FROM ap_warrior t, images i
        WHERE LLM(
            'You will be provided with a horse racetrack name and an image.
            Determine if the image shows the logo of the racetrack.
            Racetrack: {text:Track} Image: {image:image_filepath}', t.Track, i.image_filepath
        ) = 'Yes'
    """
    
    # Q7
    # result_df = spark.sql("""
    #     SELECT t.Airlines
    #     FROM tampa_international_airport t, images i
    #     WHERE LLM(
    #     "You will be provided with an airline name and an image. Determine if the image shows the logo of the airline. Airline: {text:Airlines} Image: {image:image_filepath} ", t.Airlines, i.image_filepath
    #     ) = 'Yes';
    #     """)
    
    output_path = f"./{dataset_name}_{keep_ratio}.csv"
    result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    pruning_time = 0
    if TIMING_CSV_FILE and os.path.exists(TIMING_CSV_FILE):
        timing_df = pd.read_csv(TIMING_CSV_FILE)
        pruning_time = timing_df['total_time'].sum()
        
    print(f"\nExecution time for keep_ratio={keep_ratio}: {execution_time:.2f} seconds")
    print(f"Pruning time for keep_ratio={keep_ratio}: {pruning_time:.2f} seconds")
    print(f"Pruning percentage: {pruning_time/execution_time*100:.2f}%")
        
    # Write execution time to text file
    time_log_path = f"./{dataset_name}_{keep_ratio}_execution_time.txt"
    with open(time_log_path, 'w') as f:
        f.write(f"Experiment Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Keep Ratio: {keep_ratio}\n")
        f.write(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n")
        f.write(f"Total Pruning Time: {pruning_time:.2f} seconds ({pruning_time/60:.2f} minutes)\n")
        f.write(f"Pruning Time Percentage: {pruning_time/execution_time*100:.2f}%\n")
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
    keep_ratios = [0.2]
    dataset_name = "SEMbench_q2a"
    
    overall_start = time.time()
    results = {}
    execution_times = {}
    pruning_times = {}
    
    for keep_ratio in keep_ratios:
        output_path, exec_time = run_experiment(
            keep_ratio, 
            dataset_name
        )
        
        execution_times[keep_ratio] = exec_time
        #pruning_times[keep_ratio] = prune_time
        
        accuracy = 0 #calculate_accuracy(output_path, keep_ratio)
        results[keep_ratio] = accuracy
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    total_pruning_time = sum(pruning_times.values())
    
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
            prune_pct = (prune_time / exec_time * 100) if exec_time > 0 else 0
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
        prune_pct = (prune_time / exec_time * 100) if exec_time > 0 else 0
        if accuracy is not None:
            print(f"{keep_ratio:<12.3f} {accuracy:<12.2f}% {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}% {'✓':<10}")
        else:
            print(f"{keep_ratio:<12.3f} {'N/A':<12} {exec_time:<15.2f} {prune_time:<16.2f} {prune_pct:<10.2f}% {'✗':<10}")
    print(f"{'='*80}\n")
    
    spark.stop()

