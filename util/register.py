import json
import re
import time
import pandas as pd
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame as SparkDataFrame
from vllm import EngineArgs
from threading import Lock
from util.prompt import DEFAULT_SYSTEM_PROMPT
from util.vllm import vLLM
from util.utils import LLM, get_ordered_columns
from pandas import DataFrame
from algos.quick_greedy import QuickGreedy
from util.utils import *
from string import Formatter
import os
from pyspark.sql.functions import col, window, expr, sum, avg, count, when, lit, unix_timestamp, date_format, pandas_udf

# from pyflink.datastream import StreamExecutionEnvironment
# from pyflink.table import StreamTableEnvironment, DataTypes, EnvironmentSettings
# from pyflink.table.udf import udf
# from pyflink.common.typeinfo import Types
# from pyflink.common import Row
# from pyflink.datastream.functions import MapFunction, ProcessFunction
# from pyflink.datastream.state import ValueStateDescriptor
import time
import random
import csv
import os
from threading import Thread

class ModelRegistry:
    _instance = None
    _lock = Lock()
    _model = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def initialize_model(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    print("Initializing global model...")
                    engine_args = EngineArgs(
                        model="/data/models/Qwen2.5-Coder-7B-Instruct",
                        max_num_seqs=1024
                    )
                    self._model = vLLM(
                        engine_args=engine_args,
                        base_url="http://localhost:8002/v1"
                    )
                    self._tokenizer = get_tokenizer()
                    self._initialized = True
                    print("Global model initialized successfully.")

    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    @property
    def model(self) -> LLM:
        if not self._initialized:
            self.initialize_model()
        return self._model

# Global variables
model_registry = ModelRegistry()
global_spark = None
global_table_name = ""

algo_config = "quick_greedy"

base_path = os.path.dirname(os.path.abspath(__file__))
solver_config_path = dataset_config_path = os.path.join(base_path+"/../", "solver_configs", f"{algo_config}.yaml")
solver_config = read_yaml(solver_config_path)
algo = solver_config["algorithm"]
merged_cols = [] if not solver_config.get("colmerging", True) else data_config["merged_columns"]
one_deps = []
default_distinct_value_threshold = 0.0

processed_row = 0

def init(model_runner: LLM):
    global REGISTERED_MODEL
    REGISTERED_MODEL = model_runner

def set_global_state(spark, table_name):
    global global_spark, global_table_name
    global_spark = spark
    global_table_name = table_name
    # Initialize the model when setting global state
    model_registry.initialize_model()


def get_fields(user_prompt: str) -> List[str]:
    """Get the names of all the fields specified in the user prompt."""
    if not isinstance(user_prompt, str):
        raise ValueError("Expected a string for user_prompt, got: {}".format(type(user_prompt)))
    pattern = r"{(.*?)}"
    return re.findall(pattern, user_prompt)

def batchQuery(model: LLM, 
          prompt: str, 
          df: DataFrame, 
          system_prompt: str = DEFAULT_SYSTEM_PROMPT,
          ):


    df.drop_duplicates()
    # Returns a list of dicts, maintaining column order.
    records = df.to_dict(orient="records")
    print("Len of Records:", len(records))
    outputs = model.execute_batch_v2(
        fields=records,
        query=prompt,
        system_prompt=system_prompt
    )
    return outputs

def query(model: LLM, 
          prompt: str, 
          df: DataFrame, 
          reorder_columns: bool = True,
          reorder_rows: bool = True,
          system_prompt: str = DEFAULT_SYSTEM_PROMPT,
          ):
    
    fields = get_fields(prompt)
    
    for field in fields:
        if field not in df.columns:
            raise ValueError(f"Provided field {field} does not exist in dataframe")
    
    if reorder_columns:
        print("Column reorder : True")
        fields = get_ordered_columns(df, fields)
        df = df[fields]
        #df = df.drop_duplicates(subset=fields)

    else:
        # If reorder_columns is False, filter down to columns that appear in the prompt
        # but maintain original column order
        print("Column reorder : False")
        df = df[fields]

    df.drop_duplicates()

    if reorder_rows:
        print("reorder rows ...")
        df = df.sort_values(by=fields)
    # Returns a list of dicts, maintaining column order.
    records = df.to_dict(orient="records")
    outputs = model.execute_batch_sync(
        fields=records,
        query=prompt,
        system_prompt=system_prompt
        #guided_choice=["Yes", "No"]
    )
    return outputs


def batchQuery(model: LLM, 
          prompt: str, 
          df: DataFrame, 
          system_prompt: str = DEFAULT_SYSTEM_PROMPT,
          ):
    

    records = df.to_dict(orient="records")
    outputs = model.execute_batch_v2(
        fields=records,
        query=prompt,
        system_prompt=system_prompt
    )
    return outputs

from pyspark.sql.functions import udf # Import the standard udf decorator



@udf(StringType())
def llm_udf_v1(prompt: str, *dataFrames: pd.Series) -> pd.Series:

    outputs = []    
    # Get the global model instance
    model = model_registry.model
    
    # Process each Series in dataFrames
    fields = get_fields(prompt)
    if len(dataFrames) != len(fields):
        raise ValueError(
            f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
            f"but got {len(dataFrames)}."
        )

    # Extract fields dynamically from the user prompt
    merged_df = pd.DataFrame({field: col.apply(lambda x: x.split(':', 1)[1].strip() if ':' in x else x) 
                             for field, col in zip(fields, dataFrames)})
    start_time = time.time()
    print("\n=== Before Reordering ===")
    print("Columns:", merged_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(merged_df.head())

    merged_df, _ = QuickGreedy().reorder(
            merged_df,
            early_stop=solver_config["early_stop"],
            row_stop=solver_config.get("row_stop", None),
            col_stop=solver_config.get("col_stop", None),
            col_merge=merged_cols,
            one_way_dep=one_deps,
            distinct_value_threshold=solver_config.get("distinct_value_threshold", default_distinct_value_threshold),
        )
    after_reorder_time = time.time()
    print(f"\nReordering time: {after_reorder_time - start_time:.4f} seconds")

    print("Columns:", merged_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(merged_df.head())

    before_query_time = time.time()
    # Call the query function for batch execution using the global model
    outputs = batchQuery(
        model=model,
        prompt=prompt,
        df=merged_df,
        system_prompt=DEFAULT_SYSTEM_PROMPT
    )

    end_time = time.time()
    print(f"\nBatch query time: {end_time - before_query_time:.4f} seconds")
    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
    
    return pd.Series(outputs)



# Pyspark UDF
@pandas_udf(StringType(), PandasUDFType.SCALAR)
def llm_udf_v2(prompts: pd.Series, *dataFrames: pd.Series) -> pd.Series:
    print("len of prompts", len(prompts))
    outputs = []    
    prompt = prompts.iloc[0]
    
    # Get the global model instance
    model = model_registry.model
    
    # Process each Series in dataFrames
    fields = get_fields(prompts.iloc[0])
    if len(dataFrames) != len(fields):
        raise ValueError(
            f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
            f"but got {len(dataFrames)}."
        )

    # Extract fields dynamically from the user prompt
    merged_df = pd.DataFrame({field: col.apply(lambda x: x.split(':', 1)[1].strip() if ':' in x else x) 
                             for field, col in zip(fields, dataFrames)})
    start_time = time.time()
    print("\n=== Before Reordering ===")
    print("Columns:", merged_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(merged_df.head())

    merged_df, _ = QuickGreedy().reorder(
            merged_df,
            early_stop=solver_config["early_stop"],
            row_stop=solver_config.get("row_stop", None),
            col_stop=solver_config.get("col_stop", None),
            col_merge=merged_cols,
            one_way_dep=one_deps,
            distinct_value_threshold=solver_config.get("distinct_value_threshold", default_distinct_value_threshold),
        )
    after_reorder_time = time.time()
    print(f"\nReordering time: {after_reorder_time - start_time:.4f} seconds")

    print("Columns:", merged_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(merged_df.head())

    before_query_time = time.time()
    # Call the query function for batch execution using the global model
    outputs = batchQuery(
        model=model,
        prompt=prompt,
        df=merged_df,
        system_prompt=DEFAULT_SYSTEM_PROMPT
    )

    end_time = time.time()
    print(f"\nBatch query time: {end_time - before_query_time:.4f} seconds")
    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
    
    return pd.Series(outputs)

# Register the LLM UDF with Spark
def register_llm_udf():
    nondet_llm_udf = llm_udf_v2.asNondeterministic()
    global_spark.udf.register("LLM", nondet_llm_udf)


# class LLM(ProcessFunction):
#     def __init__(self, query):
#         # Get the global model instance
#         self.model = model_registry.model
#         self.query = query
#         self.base_url = "http://localhost:8002/v1"
#         self.model_name = is_server_running()
        

#     def generate_prompt(self, user_prompt: str, system_prompt: str) -> str:
#         messages = [
#             {"role": "user", "content": user_prompt},
#             {"role": "system", "content": system_prompt}
#         ]

#         successful_prompt_generation = False
#         while not successful_prompt_generation:
#             try:
#                 # Construct a prompt for the chosen model given OpenAI style messages.
#                 prompt = self.model.tokenizer.apply_chat_template(
#                     conversation=messages,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )
#             except Exception as e:
#                 if messages[0]["role"] == "system":
#                     # Try again without system prompt
#                     messages = messages[1:]
#                 else:
#                     raise e
#             else:
#                 successful_prompt_generation = True
        
#         return prompt

#     def process_element(self, row: Row, ctx: ProcessFunction.Context):
#         row_dict = row.as_dict()
#         try:
            
#             field_names = [fname for _, fname, _, _ in Formatter().parse(self.query) if fname]
#             prompt_data = {field: row_dict[field] for field in field_names}
#             user_prompt_template = f"{self.query} Given the following data:\n {prompt_data} \n answer the above query:"
#             prompts = [self.generate_prompt(user_prompt_template, DEFAULT_SYSTEM_PROMPT)]
#             #print(prompts[0])
#             #prompts = [user_prompt_template]
#             request_outputs = json.loads(post_http_request(self.model_name, prompts, temperature=0, api_url=(self.base_url + "/completions")).content)
#             #print("result: ", [choice['text'] for choice in request_outputs['choices']])

#         except KeyError as e:
#             print(f"Missing column in row data: {e}")



class SemanticExtension:
    """
    Extension class that adds semantic filtering capabilities to PySpark DataFrames
    by integrating with LLM serving API using batch processing.
    """
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        """
        Initialize the semantic filtering extension.
        
        Args:
            system_prompt: System prompt for the LLM model
        """
        self.system_prompt = system_prompt
        self._register_extension()
    
    def _extract_placeholders(self, template: str) -> List[str]:
        """Extract column placeholders from a template string like "{col1} and {col2}"""
        return re.findall(r'\{([^}]+)\}', template)
    
    def _register_extension(self):
        """Register the sem_filter method to DataFrame class"""
        def sem_filter_v2(dataframe, filter_template: str, output_col: str = "_keep_record") -> SparkDataFrame:
            system_prompt = self.system_prompt
            
            placeholders = self._extract_placeholders(filter_template)
            
            if not placeholders:
                raise ValueError(f"No column placeholders found in template: {filter_template}")
                
            for col_name in placeholders:
                if col_name not in dataframe.columns:
                    raise ValueError(f"Column '{col_name}' referenced in filter template not found in DataFrame")
            
            udf_args = [lit(filter_template)]

            for col_name in placeholders:
                udf_args.append(col(col_name))

            return dataframe.repartition(1).withColumn(
                output_col,
                llm_udf_v2(*udf_args) # Use * to unpack the list
            )

        SparkDataFrame.sem_filter_v2 = sem_filter_v2
