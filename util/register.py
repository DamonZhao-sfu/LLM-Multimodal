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
from util.utils import *
from string import Formatter
import os
from pyspark.sql.functions import col, window, expr, sum, avg, count, when, lit, unix_timestamp, date_format, pandas_udf

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
                        model="llava-hf/llava-1.5-7b-hf",
                        max_num_seqs=1024
                    )
                    self._model = vLLM(
                        engine_args=engine_args,
                        base_url="http://localhost:8000/v1/chat"
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
# solver_config_path = dataset_config_path = os.path.join(base_path+"/../", "solver_configs", f"{algo_config}.yaml")
# solver_config = read_yaml(solver_config_path)
# algo = solver_config["algorithm"]
# merged_cols = [] if not solver_config.get("colmerging", True) else data_config["merged_columns"]
# one_deps = []
# default_distinct_value_threshold = 0.0

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


def register_llm_udf():
    global_spark.udf.register("LLM", llm_udf)


def get_fields(user_prompt: str) -> List[str]:
    """Get the names of all the fields specified in the user prompt."""
    if not isinstance(user_prompt, str):
        raise ValueError("Expected a string for user_prompt, got: {}".format(type(user_prompt)))
    pattern = r"{(.*?)}"
    return re.findall(pattern, user_prompt)

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

    #df.drop_duplicates()

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

# @pandas_udf(StringType(), PandasUDFType.SCALAR)
# def llm_udf(prompts: pd.Series, *args: pd.Series) -> pd.Series:
#     model = model_registry.model
#     if model is None:
#         raise RuntimeError("Registered model is not initialized.")
    
#     # Extract the prompt template from the first element (all rows use the same template)
#     prompt_template = prompts.iloc[0]
    
#     # Extract the placeholder names from the prompt template.
#     fields = get_fields(prompt_template)
#     if len(args) != len(fields):
#         raise ValueError(
#             f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
#             f"but got {len(args)}."
#         )
    
#     merged_df = pd.DataFrame({
#         field: args[i] for i, field in enumerate(fields)
#     })

#     outputs = batchQuery(
#         model=model,
#         prompt=prompt_template,
#         df=merged_df,
#         system_prompt=DEFAULT_SYSTEM_PROMPT
#     )

#     # Convert the outputs to a Pandas Series
#     return pd.Series(outputs)


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def llm_udf(prompts: pd.Series, *args: pd.Series) -> pd.Series:
    """
    Enhanced LLM UDF with support for typed fields (text and image).
    
    Usage:
        SELECT LLM('Given the {text:question} and {image:image_col} give me the answer', 
                   question, image_col) as summary 
        FROM table
    """
    model = model_registry.model
    if model is None:
        raise RuntimeError("Registered model is not initialized.")
    
    # Extract the prompt template from the first element
    prompt_template = prompts.iloc[0]
    print(prompt_template)
    # Parse typed fields from the prompt template
    typed_fields = parse_typed_fields(prompt_template)

    if len(args) != len(typed_fields):
        raise ValueError(
            f"Expected {len(typed_fields)} column(s) for fields {[f[0] for f in typed_fields]}, "
            f"but got {len(args)}."
        )
    
    # Build dataframe with field values
    merged_df = pd.DataFrame({
        field_name: args[i] for i, (field_name, field_type) in enumerate(typed_fields)
    })
    
    # Convert dataframe rows to list of dictionaries
    fields_list = merged_df.to_dict('records')
    
    # Execute batch query with image support
    outputs = execute_batch_v2_with_images(
        model=model,
        fields=fields_list,
        query=prompt_template,
        typed_fields=typed_fields,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        base_url=model.base_url if hasattr(model, 'base_url') else None
    )
    
    return pd.Series(outputs)

# Pyspark UDF
# @pandas_udf(StringType(), PandasUDFType.SCALAR)
# def llm_udf_v2(prompts: pd.Series, *dataFrames: pd.Series) -> pd.Series:
#     """
#     UDF that processes batches of data with LLM queries.
    
#     Args:
#         prompts: Series containing the same prompt template for all rows
#         *dataFrames: Variable number of Series, each containing values from a column
    
#     Returns:
#         Series containing LLM responses
#     """
#     print(f"len of prompts: {len(prompts)}")
#     print(f"number of data columns: {len(dataFrames)}")
    
#     # Get the prompt template from the first row (same for all rows)
#     prompt_template = prompts.iloc[0]
    
#     # Get the global model instance
#     model = model_registry.model
    
#     # Extract field placeholders from the prompt (e.g., {id}, {question}, etc.)
#     fields = get_fields(prompt_template)
    
#     # Validate that we have the right number of columns
#     if len(dataFrames) != len(fields):
#         raise ValueError(
#             f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
#             f"but got {len(dataFrames)}."
#         )
    
#     # Create DataFrame from the column values
#     # Each Series in dataFrames contains values for one column across all rows
#     merged_df = pd.DataFrame({
#         field: dataFrames[i] for i, field in enumerate(fields)
#     })
    
#     print(f"Merged DataFrame shape: {merged_df.shape}")
#     print(f"Sample row:\n{merged_df.head(1)}")
    
#     # Execute batch query
#     before_query_time = time.time()
#     outputs = batchQuery(
#         model=model,
#         prompt=prompt_template,
#         df=merged_df,
#         system_prompt=DEFAULT_SYSTEM_PROMPT
#     )
#     end_time = time.time()
    
#     print(f"Batch query time: {end_time - before_query_time:.4f} seconds")
    
#     return pd.Series(outputs)


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def llm_udf_v2(prompts: pd.Series, *dataFrames: pd.Series) -> pd.Series:
    """
    UDF that processes batches of data with LLM queries.
    """
    import re
    import pandas as pd
    import time
    
    # Define get_fields inside the UDF to avoid pickling issues
    def get_fields_local(user_prompt: str):
        """Get the names of all the fields specified in the user prompt."""
        pattern = r"{(.*?)}"
        return re.findall(pattern, user_prompt)
    
    print(f"len of prompts: {len(prompts)}")
    print(f"number of data columns: {len(dataFrames)}")
    
    # Get the prompt template from the first row
    prompt_template = prompts.iloc[0]
    
    # Get the global model instance - import inside UDF
    from util.register import ModelRegistry
    model_reg = ModelRegistry()
    model = model_reg.model
    
    # Extract field placeholders from the prompt
    fields = get_fields_local(prompt_template)
    
    # Validate that we have the right number of columns
    if len(dataFrames) != len(fields):
        raise ValueError(
            f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
            f"but got {len(dataFrames)}."
        )
    
    # Create DataFrame from the column values
    merged_df = pd.DataFrame({
        field: dataFrames[i] for i, field in enumerate(fields)
    })
    
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(f"Sample row:\n{merged_df.head(1)}")
    
    # Execute batch query - import inside UDF
    from util.register import batchQuery
    from util.prompt import DEFAULT_SYSTEM_PROMPT
    
    before_query_time = time.time()
    outputs = batchQuery(
        model=model,
        prompt=prompt_template,
        df=merged_df,
        system_prompt=DEFAULT_SYSTEM_PROMPT
    )
    end_time = time.time()
    
    print(f"Batch query time: {end_time - before_query_time:.4f} seconds")
    
    return pd.Series(outputs)


# Register the LLM UDF with Spark
