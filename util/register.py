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
from pandas import DataFrame
from util.utils import *
from util.cdencoder import *
from util.mllm import *
from string import Formatter
import os
from pyspark.sql.functions import col, window, expr, sum, avg, count, when, lit, unix_timestamp, date_format, pandas_udf

import time
import random
import csv
import os
from threading import Thread

# class ModelRegistry:
#     _instance = None
#     _lock = Lock()
#     _model = None
#     _initialized = False

#     def __new__(cls):
#         if cls._instance is None:
#             with cls._lock:
#                 if cls._instance is None:
#                     cls._instance = super(ModelRegistry, cls).__new__(cls)
#         return cls._instance

#     def initialize_model(self):
#         if not self._initialized:
#             with self._lock:
#                 if not self._initialized:
#                     engine_args = EngineArgs(
#                         model="/data/models/llava-1.5-7b-hf",
#                         max_num_seqs=1024
#                     )
#                     self._model = vLLM(
#                         engine_args=engine_args,
#                         base_url="http://localhost:8000/v1"
#                     )
#                     self._tokenizer = get_tokenizer()
#                     self._initialized = True
#                     print("Global model initialized successfully.")

#     @property
#     def tokenizer(self):
#         if not hasattr(self, '_tokenizer'):
#             self._tokenizer = get_tokenizer()
#         return self._tokenizer

#     @property
#     def model(self) -> LLM:
#         if not self._initialized:
#             self.initialize_model()
#         return self._model

# # Global variables
# model_registry = ModelRegistry()
# global_spark = None
# global_table_name = ""

# algo_config = "quick_greedy"

# base_path = os.path.dirname(os.path.abspath(__file__))

# processed_row = 0

# # MODEL_PATH = "/data/models/llava-1.5-7b-hf"
# # model = LlavaForConditionalGeneration.from_pretrained(
# #     MODEL_PATH, 
# #     torch_dtype=torch.float16, 
# #     device_map="cuda",
# #     attn_implementation="eager"
# # )
# def init(model_runner: LLM):
#     global REGISTERED_MODEL
#     REGISTERED_MODEL = model_runner

# def set_global_state(spark, table_name):
#     global global_spark, global_table_name
#     global_spark = spark
#     global_table_name = table_name
#     # Initialize the model when setting global state
#     model_registry.initialize_model()


# def register_llm_udf():
#     global_spark.udf.register("LLM", llm_udf)


# def get_fields(user_prompt: str) -> List[str]:
#     """Get the names of all the fields specified in the user prompt."""
#     if not isinstance(user_prompt, str):
#         raise ValueError("Expected a string for user_prompt, got: {}".format(type(user_prompt)))
#     pattern = r"{(.*?)}"
#     return re.findall(pattern, user_prompt)


# @pandas_udf(StringType(), PandasUDFType.SCALAR)
# def llm_udf(prompts: pd.Series, *args: pd.Series) -> pd.Series:
#     """
#     Enhanced LLM UDF with support for typed fields (text and image).
    
#     Usage:
#         SELECT LLM('Given the {text:question} and {image:image_col} give me the answer', 
#                    question, image_col) as summary 
#         FROM table
#     """
#     model = model_registry.model
#     if model is None:
#         raise RuntimeError("Registered model is not initialized.")
    
#     # Extract the prompt template from the first element
#     prompt_template = prompts.iloc[0]
#     typed_fields = parse_typed_fields(prompt_template)

#     if len(args) != len(typed_fields):
#         raise ValueError(
#             f"Expected {len(typed_fields)} column(s) for fields {[f[0] for f in typed_fields]}, "
#             f"but got {len(args)}."
#         )
    
#     data_dict = {}
#     for i, (field_name, field_type) in enumerate(typed_fields):
#         arg = args[i]
#         if isinstance(arg, pd.DataFrame):
#             data_dict[field_name] = arg.values.tolist()
#         elif isinstance(arg, pd.Series):
#             data_dict[field_name] = arg.tolist()
#         else:
#             data_dict[field_name] = list(arg)

#     merged_df = pd.DataFrame(data_dict)
    
#     # Convert dataframe rows to list of dictionaries
#     fields_list = merged_df.to_dict('records')
    
#     # Execute batch query with image support
#     outputs = execute_batch_v2_with_images(
#         model=model,
#         fields=fields_list,
#         query=prompt_template,
#         typed_fields=typed_fields,
#         system_prompt=DEFAULT_SYSTEM_PROMPT,
#         guided_choice=["Yes", "No"],
#         base_url="http://localhost:8000/v1"
#     )
    
#     return pd.Series(outputs)

# @pandas_udf(StringType(), PandasUDFType.SCALAR)
# def llm_udf_embedding(prompts: pd.Series, *args: pd.Series) -> pd.Series:
#     """
#     Enhanced LLM UDF with support for typed fields (text and image).
    
#     Usage:
#         SELECT LLM('Given the {text:question} and {image:image_col} give me the answer', 
#                    question, image_col) as summary 
#         FROM table
#     """
#     # Extract the prompt template from the first element
#     prompt_template = prompts.iloc[0]
#     print(prompt_template)
#     # Parse typed fields from the prompt template
#     typed_fields = parse_typed_fields(prompt_template)

#     if len(args) != len(typed_fields):
#         raise ValueError(
#             f"Expected {len(typed_fields)} column(s) for fields {[f[0] for f in typed_fields]}, "
#             f"but got {len(args)}."
#         )
    
#     data_dict = {}
#     for i, (field_name, field_type) in enumerate(typed_fields):
#         arg = args[i]
#         if isinstance(arg, pd.DataFrame):
#             data_dict[field_name] = arg.values.tolist()
#         elif isinstance(arg, pd.Series):
#             data_dict[field_name] = arg.tolist()
#         else:
#             data_dict[field_name] = list(arg)

#     merged_df = pd.DataFrame(data_dict)
    
#     # Convert dataframe rows to list of dictionaries
#     fields_list = merged_df.to_dict('records')
    
#     # Execute batch query with image support
#     outputs = execute_batch_v2_with_pruned_embeddings(
#         model=model,
#         modelname="/data/models/llava-1.5-7b-hf",
#         fields=fields_list,
#         query=prompt_template,
#         typed_fields=typed_fields,
#         system_prompt=DEFAULT_SYSTEM_PROMPT,
#         guided_choice=["Yes", "No"],
#         base_url="http://localhost:8000/v1"
#     )
    
#     return pd.Series(outputs)
