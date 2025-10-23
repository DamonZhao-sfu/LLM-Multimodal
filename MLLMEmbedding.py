import sys
import os
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
import torch
import json
from typing import Iterator, Tuple
from util.register import *
from util.utils import *
from util.cdencoder import *

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

output_path = "./demoResult.csv"

# Spark configuration
spark = SparkSession.builder \
    .appName("LLM SQL Test") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "128g") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.executor.memoryOverhead", "16g") \
    .config("spark.python.worker.memory", "32g") \
    .config("spark.rpc.message.maxSize", "512") \
    .getOrCreate()

# Global variables for model configuration
MODEL_PATH = "/data/models/llava-1.5-7b-hf"
API_URL = "http://localhost:8000/v1/chat/completions"
KEEP_RATIO = 0.5
RECOVERY_RATIO = 0.0

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
                        model="/data/models/llava-1.5-7b-hf",
                        max_num_seqs=1024
                    )
                    self._model = vLLM(
                        engine_args=engine_args,
                        base_url="http://localhost:8000/v1"
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


def create_llm_udf_with_embeddings():    
    @pandas_udf(StringType())
    def llm_udf_embedding_batch(
        prompts: pd.Series,
        *args: pd.Series
    ) -> pd.Series:
        import torch
        import json
        import time
        import requests
        from typing import Optional
        
        if not hasattr(llm_udf_embedding_batch, 'model_initialized'):
            try:
                from util.utils import initialize_model_for_pruning                
                llm_udf_embedding_batch.model = initialize_model_for_pruning(MODEL_PATH)
                llm_udf_embedding_batch.model_initialized = True
                print(f"Model initialized on executor: {os.getpid()}")
            except Exception as e:
                print(f"Error initializing model: {e}")
                llm_udf_embedding_batch.model = None
                llm_udf_embedding_batch.model_initialized = False
        
        results = []

        prompt_template = prompts.iloc[0]
        print(prompt_template)
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
        
        # Convert dataframe rows to list of dictionaries
        fields_list = merged_df.to_dict('records')
        outputs = execute_batch_v2_with_pruned_embeddings(
            model=llm_udf_embedding_batch.model,
            modelname="/data/models/llava-1.5-7b-hf",
            fields=fields_list,
            query=prompt_template,
            typed_fields=typed_fields,
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

# Register UDF
llm_udf = create_llm_udf_with_embeddings()
spark.udf.register("LLM", llm_udf)

# Main execution
start_time = time.time()

# Read POPE parquet
POPE_PATH = "/home/haikai/haikai/entropyTest/POPE.parquet"
pope_df = spark.read.parquet(POPE_PATH).limit(100)
pope_df.createOrReplaceTempView("pope")

# Execute query with proper column references
result_df = spark.sql("SELECT LLM('Given the text: {text:question} and image: {image:image} give me the answer to the question', question, image) as summary FROM pope LIMIT 10")


# Show execution plan
result_df.explain(extended=True)

# Collect and save results
print("Collecting results...")
result_df.show(truncate=False)

# Write to CSV
result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
print(f"Results saved to: {output_path}")

spark.stop()
