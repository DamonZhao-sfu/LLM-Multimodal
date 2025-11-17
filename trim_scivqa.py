import sys
import os
import time
import pandas as pd
import csv
from typing import Tuple, List, Dict, Any, Callable
from pyspark.sql.functions import array_contains, lower, trim, col, when, expr, regexp_replace, get_json_object

import json
import asyncio

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


def extract_image_binary_from_scivqa_data(image_path):
    with open("/home/haikai/images_train/" + image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

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
        
        outputs = execute_batch_scivqa_with_pruned_embeddings(
            modelname="/data/models/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            keep_ratio=keep_ratio,
            typed_fields=typed_fields,
            system_prompt=full_system_prompt,
            base_url="http://localhost:8000/v1"
        )
        
        return pd.Series(outputs)
    
    return llm_udf_embedding_batch



def execute_batch_scivqa_with_pruned_embeddings(
    modelname,
    fields: List[Dict[str, any]],
    query: str,
    keep_ratio: float,
    typed_fields: List[Tuple[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
) -> List[str]:
    """
    Execute batch queries with pruned image embeddings.
    Models are loaded at the start and cleaned up at the end.
    """
    # Load models at the beginning
    vision_tower, model, tokenizer = load_vision_models(device='cuda')
    
    try:
        # Build user prompts and generate pruned embeddings
        user_prompts = []
        all_pruned_embeddings = []
        
        for field_dict in fields:
            user_prompt = query
            pruned_embeddings_for_this_prompt = []
            
            for field_name, field_type in typed_fields:
                placeholder = f"{{{field_type}:{field_name}}}"   
                # if field_name == "qa_pair_type":
                #     qa_pair_type = field_dict.get(field_name, "")
                #     answer_options = field_dict.get("answer_options", [])
                #     if qa_pair_type in ["closed-ended finite answer set binary visual", "closed-ended finite answer set binary non-visual"]:
                #         system_prompt += "\n\nREMEMBER: Your entire answer must be EXACTLY 'Yes' or 'No' - nothing more, nothing less."
                #     elif qa_pair_type in ["closed-ended finite answer set non-binary visual", "closed-ended finite answer set non-binary non-visual"] and len(answer_options) == 4:
                #         system_prompt += "\n\nREMEMBER: Your entire answer must be ONLY the letter(s) of the correct option(s) - e.g., 'A' or 'B,D'."
                #     elif qa_pair_type in ["closed-ended infinite answer set visual", "closed-ended infinite answer set non-visual"]:
                #         system_prompt += "\n\nREMEMBER: Your answer must be concise and direct, with no explanatory text."
                #     elif qa_pair_type == "unanswerable":
                #         system_prompt += "\n\nREMEMBER: Decide if the question is unanswerable based on the figure and caption. If it is, respond with 'It is not possible to answer this question based only on the provided data.'. If it is not, respond with the correct answer."

                if field_type == "text":
                    value = field_dict.get(field_name, "")
                    user_prompt = user_prompt.replace(placeholder, str(value))
                
                elif field_type == "image":
                    user_prompt = user_prompt.replace(placeholder, "[image]")
                    image_data = field_dict.get(field_name)
                    
                    if image_data is not None:
                        image_binary = extract_image_binary_from_scivqa_data(image_data)
                        
                        if keep_ratio == 1:
                            reduced_tokens, preprocess_time, encode_time  = getOriginalVisualToken(
                                model,
                                vision_tower,
                                image_binary
                            )
                            # Record timing with zeros for no pruning
                            record_timing(keep_ratio, preprocess_time, encode_time, 0.0)
                        else:
                            reduced_tokens, preprocess_time, encode_time, prune_time = trimTokenatorPruning(
                                model,
                                vision_tower,
                                tokenizer,
                                image_binary,
                                user_prompt,
                                keep_ratio=keep_ratio
                            )
                            # Record timing for this invocation
                            record_timing(keep_ratio, preprocess_time, encode_time, prune_time)
                        
                        pruned_embeddings_for_this_prompt.append(reduced_tokens.to(torch.float16))
            
            user_prompts.append(user_prompt)
            all_pruned_embeddings.append(
                pruned_embeddings_for_this_prompt[0] if pruned_embeddings_for_this_prompt else None
            )
        
        # Generate full prompts
        prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
                   for user_prompt in user_prompts]
        
        outputs = []
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
                        image_embeddings=[all_pruned_embeddings[i]] if all_pruned_embeddings[i] is not None else None
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
                            print(f"Warning: No valid content in response. Output: {request_output}")
                            processed_outputs.append(None)
                    except Exception as e:
                        print(f"Error processing response: {e}. Response content: {getattr(response, 'content', 'N/A')}")
                        processed_outputs.append(None)
                        
                return processed_outputs

            outputs = asyncio.run(fetch_all())            
            return outputs
    
    finally:
        pass


def extract_image_binary_from_pope_data(image_data):
    """Extract image binary from POPE data format."""
    if isinstance(image_data, (list, tuple)):
        return image_data[0] if len(image_data) > 0 else image_data
    return image_data


def run_experiment(keep_ratio: float, dataset_name: str = "POPE_random") -> Tuple[str, float]:
    """Run experiment with specific keep_ratio and save results."""
    initialize_timing_csv(keep_ratio, dataset_name)

    print(f"\n{'='*80}")
    print(f"Running experiment with keep_ratio={keep_ratio}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Register UDF with current keep_ratio
    llm_udf = create_llm_udf_with_embeddings(keep_ratio)
    spark.udf.register("LLM", llm_udf)
    
    # Read POPE parquet
    POPE_PATH = "/home/haikai/train_2025-07-03_09-06.json"
    pope_df = spark.read \
        .option("multiLine", "true") \
        .option("encoding", "UTF-8") \
        .json(POPE_PATH) \
        .limit(1000) \
        .cache()
    pope_df.createOrReplaceTempView("pope")
    print(f"Total records: {pope_df.count()}")
    
    # Execute query with proper column references
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
        lower(trim(col("predicted"))).contains(lower(trim(col("answer")))),
        1
    ).otherwise(0))
    
    # Write results to CSV
    output_path = f"./{dataset_name}_{keep_ratio}.csv"
    result_df_with_comparison.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(output_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    pruning_time = 0
    if TIMING_CSV_FILE and os.path.exists(TIMING_CSV_FILE):
        timing_df = pd.read_csv(TIMING_CSV_FILE)
        pruning_time = timing_df['total_time'].sum()
    print(f"\nExecution time for keep_ratio={keep_ratio}: {execution_time:.2f} seconds")
    print(f"Pruning time for keep_ratio={keep_ratio}: {pruning_time:.2f} seconds")
    print(f"Pruning percentage: {pruning_time/execution_time*100:.2f}%")
    print(f"Total invocations: {INVOCATION_COUNTER}")
        
    pruning_time = 0
    if TIMING_CSV_FILE and os.path.exists(TIMING_CSV_FILE):
        timing_df = pd.read_csv(TIMING_CSV_FILE)
        pruning_time = timing_df['total_time'].sum()

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
        
    # Read CSV with semicolon delimiter
    df = pd.read_csv(actual_csv, sep=';')
        
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
    dataset_name = "SciVQA_image_trim"
    
    overall_start = time.time()
    results = {}
    execution_times = {}
    pruning_times = {}

    for keep_ratio in keep_ratios:
        output_path, exec_time,prune_time = run_experiment(keep_ratio, dataset_name)
        execution_times[keep_ratio] = exec_time
        pruning_times[keep_ratio] = prune_time

        accuracy = calculate_accuracy(output_path, keep_ratio)
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
