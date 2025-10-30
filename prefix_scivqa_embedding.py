import sys
import os
import time
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, when, lower, trim
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

full_system_prompt = '''You are an expert scientific figure analyst specializing in academic publications.
Your task is to answer questions about scientific figures and their captions accurately and concisely.
Answer the given question based *solely* on the information visible in the figure and its provided caption.

The user message will include a 'Question Type'. Adhere strictly to the following rules for formatting your response based on the question type:

- For 'closed-ended finite answer set binary visual' or 'closed-ended finite answer set binary non-visual': 
  - Respond ONLY with 'Yes' or 'No'. 
  - Do NOT add any other text, explanations, or punctuation.
  - Your entire response must be exactly one word: either 'Yes' or 'No'.

- For 'closed-ended finite answer set non-binary visual' or 'closed-ended finite answer set non-binary non-visual': 
  - Identify the correct option(s) from the provided 'Answer Options'.
  - Respond ONLY with the letter(s) of the correct option(s) as listed.
  - For a single correct option, provide only its letter (e.g., 'B').
  - For multiple correct options, list ALL correct letters separated by commas with NO SPACES (e.g., 'A,C,D').
  - Ensure ALL correct options are listed and NO incorrect ones.
  - Do NOT add any other text, explanations, or surrounding punctuation.

- For 'closed-ended infinite answer set visual' or 'closed-ended infinite answer set non-visual': 
  - Provide a brief, direct answer.
  - This answer must be a value, a short phrase, a specific name, a label, or a list of values read directly from the figure or caption.
  - **For numerical values:** Read values as precisely as possible from the graph axes, data points, or labels. Include units ONLY if they appear in the figure.
  - **For non-numerical values:** Reproduce them EXACTLY as they appear in the figure or caption.
  - Do NOT add any introductory phrases, explanations, or surrounding text.

- For 'unanswerable': 
  - Respond ONLY with the exact phrase: 'It is not possible to answer this question based only on the provided data.'
  - Do NOT add any other text.

IMPORTANT: Your response should ONLY contain the answer in the correct format as specified above - nothing else.
Do NOT include any additional text, explanations, comments, or contextual information.
Your answer must be based solely on the information visible in the figure and its provided caption.

Below are examples of questions and answers similar to what you will receive. "
                "Study these examples carefully to understand the expected answer format. "
                "Your question will be in the user message after these examples:

Example 1:
Question: What is the approximate value of the red line at x=5?
Image: xxx.jpg
Caption: Figure 3: R10@1 for different ranges on E-commerce.
QA Type: closed-ended infinite answer set visual
Answer Options: []
Answer: 0.66

Example 2:
Question: Is the value of the red line higher than the value of the blue line at the cut range of 6?
QA Type: closed-ended finite answer set binary visual
Answer Options: []
Answer: Yes

Example 3:
Question: Which line represents the R10@1 metric?
QA Type: closed-ended finite answer set binary visual
Answer Options: [{"A": "The red line"}, {"B": "The blue line"}, {"C": null}, {"D": null}]
Answer: B

Example 4:
Question: What is the exact temperature shown in the image?
QA Type: unanswerable
Answer Options: []
Answer: It is not possible to answer this question based only on the provided data.


Please only return the Answer field without other field such as Question, QA Type.

'''

def extract_image_binary_from_pope_data(image_path):
    with open("/home/haikai/images_train/" + image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

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
                image_path = image_group.iloc[0][image_column]
                # Extract image binary
                image_binary = extract_image_binary_from_pope_data(image_path)
                
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
        
        return pruned_cache
    
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
            
            if field_name == "qa_pair_type":
                qa_pair_type = field_dict.get(field_name, "")
                answer_options = field_dict.get("answer_options", [])
                if qa_pair_type in ["closed-ended finite answer set binary visual", "closed-ended finite answer set binary non-visual"]:
                    system_prompt += "\n\nREMEMBER: Your entire answer must be EXACTLY 'Yes' or 'No' - nothing more, nothing less."
                elif qa_pair_type in ["closed-ended finite answer set non-binary visual", "closed-ended finite answer set non-binary non-visual"] and len(answer_options) == 4:
                    system_prompt += "\n\nREMEMBER: Your entire answer must be ONLY the letter(s) of the correct option(s) - e.g., 'A' or 'B,D'."
                elif qa_pair_type in ["closed-ended infinite answer set visual", "closed-ended infinite answer set non-visual"]:
                    system_prompt += "\n\nREMEMBER: Your answer must be concise and direct, with no explanatory text."
                elif qa_pair_type == "unanswerable":
                    system_prompt += "\n\nREMEMBER: Decide if the question is unanswerable based on the figure and caption. If it is, respond with 'It is not possible to answer this question based only on the provided data.'. If it is not, respond with the correct answer."


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
        _generate_prompt(user_prompt=user_prompt, system_prompt=full_system_prompt)
        for user_prompt in user_prompts
    ]
    
    # Send requests to API
    outputs = []
    if base_url:
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


def create_llm_udf_with_cached_embeddings(embedding_cache: Dict[str, Dict], image_source_mapping: Dict[int, str]):
    @pandas_udf(StringType())
    def llm_udf_cached_embeddings(
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

        outputs = inference_with_cached_embeddings(
            modelname="/data/models/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            typed_fields=typed_fields,
            embedding_cache=embedding_cache,
            image_source_mapping=image_source_mapping,
            system_prompt=full_system_prompt,
            base_url="http://localhost:8000/v1"
        )
        
        return pd.Series(outputs)
    
    return llm_udf_cached_embeddings

def run_experiment_with_cached_embeddings(
    keep_ratio: float,
    pope_pandas_df: pd.DataFrame,
    pope_spark_df,
    dataset_name: str = "POPE_image_prefix"
) -> Tuple[str, float]:
    """Run experiment with cached embeddings for a specific keep_ratio."""
    print(f"\n{'='*80}")
    print(f"Running experiment with keep_ratio={keep_ratio}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Preprocess and cache pruned embeddings
    print(f"Preprocessing images with keep_ratio={keep_ratio}...")
    embedding_cache = preprocess_and_cache_pruned_embeddings(
        df=pope_pandas_df,
        image_column='image_file',
        question_column='question',
        image_id_column='figure_id',
        keep_ratio=keep_ratio,
        device='cuda:0'
    )
    
    # Create image source mapping
    image_source_mapping_reordered = {
        idx: row['figure_id'] 
        for idx, row in pope_pandas_df.iterrows()
    }
    
    # Create and register UDF
    llm_udf = create_llm_udf_with_cached_embeddings(
        embedding_cache, 
        image_source_mapping_reordered
    )
    spark.udf.register("LLM", llm_udf)
    
    # Execute query
    result_df = spark.sql("""
    SELECT 
        answer,
        LLM('
Question: {text:question}
Image: {image:image_file}
Figure Caption: {text:caption}
Figure Type: {text:figure_type}
QA Type: {text:qa_pair_type}
Answer Options: {text:answer_options}

', question, image_file, caption, figure_type, qa_pair_type, answer_options) as predicted
    FROM pope
""")
    
    result_df = result_df.withColumn(
        "predicted",
        regexp_replace(regexp_replace(col("predicted"), "[\r\n]+", " "), "\\s+", " ")
    )

    result_df_with_comparison = result_df.withColumn(
    "is_correct",
    when(
        lower(col("predicted")).contains(lower(trim(col("answer")))),
        1
    ).otherwise(0))
    
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
    keep_ratios = [1, 0.056, 0.111, 0.222]
    dataset_name = "SciVQA_image_prefix"
    
    overall_start = time.time()
    
    # Read POPE parquet once
    POPE_PATH = "/home/haikai/train_2025-07-03_09-06.json"
    pope_df = spark.read \
        .option("multiLine", "true") \
        .option("encoding", "UTF-8") \
        .json(POPE_PATH) \
        .limit(3000) \
        .cache()
    pope_df.createOrReplaceTempView("pope")
    print(f"Total records: {pope_df.count()}")
    
    # Convert to pandas once
    pope_pandas_df = pope_df.toPandas()
    
    results = {}
    execution_times = {}
    
    # Run experiments for each keep_ratio
    for keep_ratio in keep_ratios:
        output_path, exec_time = run_experiment_with_cached_embeddings(
            keep_ratio=keep_ratio,
            pope_pandas_df=pope_pandas_df,
            pope_spark_df=pope_df,
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
