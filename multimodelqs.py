import sys
import os
import time
import pandas as pd
from typing import Tuple, List, Dict, Any, Callable
from util.utils import _generate_prompt

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, when, lower, trim
from util.mllm import *
from util.utils import *
from util.cdencoder import *
from util.divpruner import *
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType, LongType, BooleanType, DoubleType
from pyspark.sql.functions import to_json, col, explode, collect_list, concat_ws, first, array_join

# Initialize SparkSession
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

IMAGE_BASE_PATH = "/home/haikai/multimodalqa/dataset/final_dataset_images"

def extract_image_binary_from_path(image_path: str, base_path: str = IMAGE_BASE_PATH) -> bytes:
    """Read image binary from file path."""
    full_path = os.path.join(base_path, image_path)
    with open(full_path, 'rb') as f:
        return f.read()

def post_http_request_with_embeds(
    model: str,
    prompts: List[str],
    temperature: float = 1.0,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    guided_choice: List[str] = None,
    image_embeddings: List[List[torch.Tensor]] = None,  # Changed to List[List[torch.Tensor]] for multiple images
    answer_schema: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    messages_list = []
    
    for i, prompt in enumerate(prompts):
        content = []
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Handle multiple image embeddings per prompt
        if image_embeddings and i < len(image_embeddings) and image_embeddings[i] is not None:
            embeddings_for_prompt = image_embeddings[i]
            
            # Check if it's a list of embeddings or single embedding
            if isinstance(embeddings_for_prompt, list):
                # Multiple images
                image_embeddings = torch.stack(embeddings_for_prompt, dim=0)
                embedding = encode_image_embedding_to_base64(image_embeddings)
                content.append({
                    "type": "image_embeds",
                    "image_embeds": embedding
                })
            else:
                # Single image (backward compatibility)
                embedding = encode_image_embedding_to_base64(embeddings_for_prompt)
                content.append({
                    "type": "image_embeds",
                    "image_embeds": embedding
                })
        
        messages_list.append({
            "role": "user",
            "content": content
        })

    pload = {
        "model": model,
        "messages": messages_list,
        "temperature": temperature,
    }
    if answer_schema is not None:
        pload["guided_json"] = answer_schema
    
    if guided_choice is not None and len(guided_choice) > 0:
        pload["guided_choice"] = guided_choice

    headers = {"Content-Type": "application/json"}
    req = requests.Request('POST', api_url, headers=headers, data=json.dumps(pload))
    prepared = req.prepare()

    with requests.Session() as session:
        response = session.send(prepared)

    return response


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
    # Load models at the beginning
    vision_tower, model, tokenizer = load_vision_models(device="cuda")
    
    try:
        # Build user prompts and generate pruned embeddings
        user_prompts = []
        all_pruned_embeddings = []  # Will contain List[List[torch.Tensor]] for multiple images per prompt
        
        for field_dict in fields:
            user_prompt = ""
            pruned_embeddings_for_this_prompt = []  # Store multiple image embeddings for this prompt
            
            # Build prompt following the REORDERED column sequence
            for field_name in reordered_columns:
                # Find the field type for this field name
                field_type = None
                for fname, ftype in typed_fields:
                    if fname == field_name:
                        field_type = ftype
                        break
                
                if field_type is None:
                    continue
                
                if field_type == "text":
                    value = field_dict.get(field_name, "")
                    user_prompt += f"{field_name}: {value}\n"
                
                elif field_type == "image":
                    image_data = field_dict.get(field_name)
                    
                    # Handle both single image path (string) and multiple image paths (list)
                    if image_data is not None:
                        # Convert to list if it's a single string
                        if isinstance(image_data, np.ndarray):
                            image_paths = image_data.tolist()
                        # Handle single string
                        elif isinstance(image_data, str):
                            image_paths = [image_data]
                        elif isinstance(image_data, list):
                            image_paths = image_data
                        else:
                            print(f"Warning: Unexpected image data type: {type(image_data)}")
                            continue
                        
                        # Process each image
                        for idx, image_path in enumerate(image_paths):
                            if image_path:  # Skip empty strings
                                user_prompt += f"{field_name}_{idx+1}: [image]\n"
                                
                                try:
                                    # Read image binary from file
                                    image_binary = extract_image_binary_from_path(image_path)
                                    
                                    # Generate pruned embeddings
                                    if keep_ratio == 1:
                                        reduced_tokens = getOriginalVisualToken(
                                            model,
                                            vision_tower,
                                            image_binary
                                        )
                                    else:
                                        reduced_tokens = getCDPrunedVisualToken(
                                            model,
                                            vision_tower,
                                            image_binary,
                                            user_prompt,
                                            keep_ratio=keep_ratio
                                        )
                                    
                                    pruned_embeddings_for_this_prompt.append(reduced_tokens.to(torch.float16))
                                
                                except Exception as e:
                                    print(f"Error processing image {image_path}: {e}")
                                    continue
            
            user_prompts.append(user_prompt.strip())
            # Store all embeddings for this prompt (could be multiple images)
            all_pruned_embeddings.append(pruned_embeddings_for_this_prompt if pruned_embeddings_for_this_prompt else None)
        
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
#     vision_tower, model, tokenizer = load_vision_models(device="cuda")
    
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
#                     print(image_data)

#                     if image_data is not None:
#                         image_binary = extract_image_binary_from_pope_data(image_data)
                        
#                         if keep_ratio == 1:
#                             reduced_tokens = getOriginalVisualToken(
#                                 model,
#                                 vision_tower,
#                                 image_binary
#                             )
#                         else:
#                             reduced_tokens = getCDPrunedVisualToken(
      
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
#         torch.cuda.empty_cache()


# def create_llm_udf_with_embeddings(
#     keep_ratio: float
# ):
#     @pandas_udf(StringType())
#     def llm_udf_embedding_batch(
#         prompts: pd.Series,
#         *args: pd.Series
#     ) -> pd.Series:        
#         prompt_template = prompts.iloc[0]
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
        
#         reordered_columns = list(merged_df.columns)
#         # Convert to records for processing
#         fields_list = merged_df.to_dict('records')
        
#         outputs = execute_batch_pope_with_pruned_embeddings(
#             modelname="/data/models/llava-1.5-7b-hf",
#             fields=fields_list,
#             query=prompt_template,
#             keep_ratio=keep_ratio,
#             typed_fields=typed_fields,
#             reordered_columns=reordered_columns,  # Pass reordered column sequence
#             system_prompt=DEFAULT_SYSTEM_PROMPT,
#             guided_choice=["Yes", "No"],
#             base_url="http://localhost:8000/v1"
#         )
        
#         return pd.Series(outputs)
    
#     return llm_udf_embedding_batch



llm_udf = create_llm_udf_with_embeddings(1.0)
spark.udf.register("LLM", llm_udf)

# ===== 1. Read and prepare df_table =====
table_schema = StructType([
    StructField("title", StringType(), True),
    StructField("url", StringType(), True),
    StructField("id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("table", StructType([
        StructField("table_rows", ArrayType(ArrayType(StructType([
            StructField("text", StringType(), True),
            StructField("links", ArrayType(StructType([
                StructField("text", StringType(), True),
                StructField("wiki_title", StringType(), True),
                StructField("url", StringType(), True)
            ]), True), True)
        ]), True)), True),
        StructField("table_name", StringType(), True),
        StructField("header", ArrayType(StructType([
            StructField("column_name", StringType(), True),
            StructField("metadata", StructType([
                StructField("parsed_values", ArrayType(DoubleType(), True), True),
                StructField("type", StringType(), True),
                StructField("num_of_links", LongType(), True),
                StructField("ner_appearances_map", MapType(StringType(), LongType()), True),
                StructField("is_key_column", BooleanType(), True),
                StructField("image_associated_column", BooleanType(), True),
                StructField("entities_column", BooleanType(), True)
            ]), True)
        ]), True), True)
    ]), True)
])

df_table = spark.read.schema(table_schema).json("/home/haikai/multimodalqa/dataset/MMQA_tables.jsonl")
df_table = df_table.withColumn("table_full_json", to_json(col("table")))
df_table = df_table.select("title", "url", "id", "text", "table_full_json")
df_table.createOrReplaceTempView("df_table")

# ===== 2. Read df_images =====
images_schema = StructType([
    StructField("title", StringType(), True),
    StructField("url", StringType(), True),
    StructField("id", StringType(), True),
    StructField("path", StringType(), True)
])

df_images = spark.read.schema(images_schema).json("/home/haikai/multimodalqa/dataset/MMQA_images.jsonl")
df_images.createOrReplaceTempView("df_images")

# ===== 3. Read df_texts =====
texts_schema = StructType([
    StructField("title", StringType(), True),
    StructField("url", StringType(), True),
    StructField("id", StringType(), True),
    StructField("text", StringType(), True)
])

df_texts = spark.read.schema(texts_schema).json("/home/haikai/multimodalqa/dataset/MMQA_texts.jsonl")
df_texts.createOrReplaceTempView("df_texts")

# ===== 4. Read questions =====
questions_schema = StructType([
    StructField("qid", StringType(), True),
    StructField("question", StringType(), True),
    StructField("metadata", StructType([
        StructField("image_doc_ids", ArrayType(StringType()), True),
        StructField("text_doc_ids", ArrayType(StringType()), True),
        StructField("table_id", StringType(), True)
    ]), True)
])

df_questions = spark.read.schema(questions_schema).json("/home/haikai/multimodalqa/dataset/MMQA_test.jsonl").limit(10)
df_questions.createOrReplaceTempView("questions")


result_df = spark.sql("""
    WITH question_context AS (
        SELECT 
            q.qid,
            q.question,
            q.metadata.table_id as table_id,
            q.metadata.image_doc_ids as image_doc_ids,
            q.metadata.text_doc_ids as text_doc_ids
        FROM questions q
    ),
    -- Explode text doc ids
    text_docs_exploded AS (
        SELECT 
            qc.qid,
            qc.question,
            qc.table_id,
            qc.image_doc_ids,
            text_doc_id
        FROM question_context qc
        LATERAL VIEW explode(qc.text_doc_ids) txt_ids AS text_doc_id
    ),
    -- Join with texts
    text_docs_joined AS (
        SELECT 
            tde.qid,
            tde.question,
            tde.table_id,
            tde.image_doc_ids,
            txt.text as text_content
        FROM text_docs_exploded tde
        LEFT JOIN df_texts txt ON tde.text_doc_id = txt.id
    ),
    -- Aggregate texts (filter nulls before aggregating)
    texts_aggregated AS (
        SELECT 
            qid,
            first(question) as question,
            first(table_id) as table_id,
            first(image_doc_ids) as image_doc_ids,
            COALESCE(
                concat_ws('\n\n--- Document Separator ---\n\n', 
                          collect_list(CASE WHEN text_content IS NOT NULL THEN text_content END)),
                'No context documents available.'
            ) as all_text_docs
        FROM text_docs_joined
        GROUP BY qid
    ),
    -- Explode image doc ids
    image_docs_exploded AS (
        SELECT 
            ta.qid,
            ta.question,
            ta.table_id,
            ta.all_text_docs,
            image_doc_id
        FROM texts_aggregated ta
        LATERAL VIEW explode(ta.image_doc_ids) img_ids AS image_doc_id
    ),
    -- Join with images
    image_docs_joined AS (
        SELECT 
            ide.qid,
            ide.question,
            ide.table_id,
            ide.all_text_docs,
            img.path as image_path
        FROM image_docs_exploded ide
        LEFT JOIN df_images img ON ide.image_doc_id = img.id
    ),
    -- Aggregate images (filter out nulls and ensure we always have a list)
    images_aggregated AS (
        SELECT 
            qid,
            first(question) as question,
            first(table_id) as table_id,
            first(all_text_docs) as all_text_docs,
            CASE 
                WHEN COUNT(CASE WHEN image_path IS NOT NULL THEN 1 END) > 0 
                THEN collect_list(CASE WHEN image_path IS NOT NULL THEN image_path END)
                ELSE array()
            END as image_paths
        FROM image_docs_joined
        GROUP BY qid
    ),
    -- Join with table
    final_context AS (
        SELECT 
            ia.qid,
            ia.question,
            ia.all_text_docs,
            ia.image_paths,
            COALESCE(tbl.table_full_json, 'No table data available.') as table_data
        FROM images_aggregated ia
        LEFT JOIN df_table tbl ON ia.table_id = tbl.id
    )
    SELECT 
        qid,
        question,
        all_text_docs,
        table_data,
        image_paths,
        LLM('
Given the following question, context documents, table data, and images, please provide a comprehensive answer.

Question: {text:question}

Context Documents:
{text:all_text_docs}

Table Data:
{text:table_data}

Image Data:
{image:image_paths}

Please analyze all the provided information and answer the question thoroughly.
',
            question,
            all_text_docs,
            table_data,
            image_paths
        ) as answer
    FROM final_context
""")

# Show results
print("--- LLM Query Results ---")
result_df.show(truncate=False)

# Save results if needed
result_df.write.mode("overwrite").json("llm_answers.json")

spark.stop()
