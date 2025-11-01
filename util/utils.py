from typing import List, Dict
import re
import abc
import requests
import json
from pandas import DataFrame
import aiohttp
from transformers import AutoTokenizer
from aiohttp import ClientTimeout
from util.prompt import DEFAULT_SYSTEM_PROMPT
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List, Tuple
from pyspark.sql import SparkSession
from itertools import combinations
import yaml
import base64
import numpy as np

GPU_DEVICES = "4,5"

class LLM(abc.ABC):
    @abc.abstractmethod
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Executes the LLM query. 

        Args:
            fields: A dict mapping from column names to values for this particular row.
            query: The user query for the LLM call. The query should specify how the column fields should be used.
            system_prompt: The system prompt to use for the LLM.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_batch(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[str]:
        """Batched version of ``execute``."""

        raise NotImplementedError



def get_tokenizer():
    return AutoTokenizer.from_pretrained("/home/haikai/LLM-Multimodal/llama-tokenizer")

async def async_post_http_request(
    session: aiohttp.ClientSession,
    model: str,
    prompts: List[str],
    api_url: str = "http://localhost:8000/v1/completions",
    temperature: float = 0,
    guided_choice: List[str] = None,
) -> Dict:
    """Async version of the HTTP request"""
    pload = {
        "model": model,
        "prompt": prompts,
        #"max_tokens": 200,
        "temperature": temperature,
        "guided_choice": guided_choice
    }
    timeout = ClientTimeout(total=300)  # Adjust as needed

    async with session.post(api_url, json=pload,timeout=timeout) as response:
        response_text = await response.text()
        return json.loads(response_text)

def is_server_running(url="http://localhost:8000/v1/models"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            model = response.json()['data'][0]['id']
            return model
    except requests.ConnectionError:
        return None
    return None

def get_fields(user_prompt: str) -> str:
    """Get the names of all the fields specified in the user prompt."""
    pattern = r"{(.*?)}"
    return re.findall(pattern, user_prompt)

def get_field_score(df: DataFrame, field: str):
    num_distinct = df[field].nunique(dropna=True)
    avg_length = df[field].apply(lambda x: len(str(x))).mean()
    return avg_length / num_distinct

def get_ordered_columns(df: DataFrame, fields: List[str]):
    print("Get ordered columns ...")
    field_scores = {}

    for field in fields:
        field_scores[field] = get_field_score(df, field)
    
    reordered_fields = [field for field in sorted(fields, key=lambda field: field_scores[field], reverse=True)]

    return reordered_fields

def prepend_col_name(df: pd.DataFrame) -> pd.DataFrame:
    # just prepend column name to each value in the column
    df = df.apply(lambda x: x.name + ": " + x.astype(str))
    return df


def unprepend_col_name(df: pd.DataFrame) -> pd.DataFrame:
    # just prepend column name to each value in the column
    df = df.apply(lambda x: x.name.split(": ")[1])
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Unnamed" in df.columns:
        df = df.drop(columns=["Unnamed"])
    df = df.fillna("")
    return df

def read_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data

def parse_typed_fields(prompt_template: str) -> List[Tuple[str, str]]:
    """
    Parse prompt template to extract field names and types.
    
    Example: 'Given the {text:question} and {image:image_col}'
    Returns: [('question', 'text'), ('image_col', 'image')]
    """
    pattern = r'\{(text|image):(\w+)\}'
    matches = re.findall(pattern, prompt_template)
    return [(field_name, field_type) for field_type, field_name in matches]

def _generate_prompt(user_prompt: str, system_prompt: str) -> str:
    """Generate the full prompt with system and user components."""
    return f"{system_prompt}\n\n{user_prompt}"
