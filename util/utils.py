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
    return AutoTokenizer.from_pretrained("/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/llama-tokenizer")
    #return AutoTokenizer.from_pretrained("/home/haikai/LLM-SQL/llama-tokenizer")
    #return AutoTokenizer.from_pretrained("/home/haikai/llama-tokenizer")

def post_http_request(
    model: str,
    prompts: List[str],
    temperature: float = 1.0,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    guided_choice: List[str] = None,
    image_urls: List[str] = None,
) -> requests.Response:
    """
    Send POST request to chat completions endpoint.
    
    Args:
        model: Model name/identifier
        prompts: List of text prompts
        temperature: Sampling temperature
        api_url: API endpoint URL
        guided_choice: Optional guided choices
        image_urls: Optional list of image URLs (one per prompt, or None for text-only)
    """
    # Construct messages for each prompt
    messages_list = []
    
    for i, prompt in enumerate(prompts):
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add image if provided
        if image_urls and i < len(image_urls) and image_urls[i]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_urls[i]
                }
            })
        
        messages_list.append({
            "role": "user",
            "content": content
        })
    
    # Construct the payload
    pload = {
        "model": model,
        "messages": messages_list,
        "temperature": temperature,
    }
    
    if guided_choice:
        pload["guided_choice"] = guided_choice

    headers = {"Content-Type": "application/json"}

    req = requests.Request('POST', api_url, headers=headers, data=json.dumps(pload))
    prepared = req.prepare()

    with requests.Session() as session:
        response = session.send(prepared)

    return response


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
    else:
        # If reorder_columns is False, filter down to columns that appear in the prompt
        # but maintain original column order
        print("Column reorder : False")
        original_columns = df.columns
        filtered_columns = [column for column in original_columns if column in fields]
        df = df[filtered_columns]
    
    if reorder_rows:
        print("reorder rows ...")
        df = df.sort_values(by=fields)

    # Returns a list of dicts, maintaining column order.
    records = df.to_dict(orient="records")
    outputs = model.execute_batch(
        fields=records,
        query=prompt,
        system_prompt=system_prompt
    )

    return outputs


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
