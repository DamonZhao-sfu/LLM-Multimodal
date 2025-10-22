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
    #return AutoTokenizer.from_pretrained("/home/haikai/LLM-SQL/llama-tokenizer")
    #return AutoTokenizer.from_pretrained("/home/haikai/llama-tokenizer")

# def post_http_request(
#     model: str,
#     prompts: List[str],
#     temperature: float = 1.0,
#     api_url: str = "http://localhost:8000/v1/chat/completions",
#     guided_choice: List[str] = None,
#     image_urls: List[str] = None,
# ) -> requests.Response:
#     messages_list = []
    
#     for i, prompt in enumerate(prompts):
#         content = []
        
#         # Add text content
#         content.append({
#             "type": "text",
#             "text": prompt
#         })
        
#         # Add image if provided
#         if image_urls and i < len(image_urls) and image_urls[i]:
#             content.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": image_urls[i]
#                 }
#             })
        
#         messages_list.append({
#             "role": "user",
#             "content": content
#         })
    
#     # Construct the payload
#     pload = {
#         "model": model,
#         "messages": messages_list,
#         "temperature": temperature,
#     }
    
#     if guided_choice:
#         pload["guided_choice"] = guided_choice

#     headers = {"Content-Type": "application/json"}

#     req = requests.Request('POST', api_url, headers=headers, data=json.dumps(pload))
#     prepared = req.prepare()

#     with requests.Session() as session:
#         response = session.send(prepared)

#     return response


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



def parse_typed_fields(prompt_template: str) -> List[Tuple[str, str]]:
    """
    Parse prompt template to extract field names and types.
    
    Example: 'Given the {text:question} and {image:image_col}'
    Returns: [('question', 'text'), ('image_col', 'image')]
    """
    pattern = r'\{(text|image):(\w+)\}'
    matches = re.findall(pattern, prompt_template)
    return [(field_name, field_type) for field_type, field_name in matches]

# def convert_image_to_base64_url(image_binary: bytes) -> str:
#     """
#     Convert binary image data to base64 data URL.
#     """
#     if image_binary is None:
#         return None
#     image_base64 = base64.b64encode(image_binary).decode('utf-8')
#     return f"data:image/jpeg;base64,{image_base64}"


def convert_image_to_base64_url(image_binary) -> str:
    """
    Convert binary image data to base64 data URL.
    Handles both bytes and list/array inputs.
    """
    if image_binary is None:
        return None
    
    # Convert list or array to bytes if needed
    if isinstance(image_binary, list):
        # If it's a list of integers (byte values), convert to bytes
        image_binary = bytes(image_binary)
    elif isinstance(image_binary, np.ndarray):
        # If it's a numpy array, convert to bytes
        image_binary = image_binary.tobytes()
    elif not isinstance(image_binary, bytes):
        # If it's some other type, try to convert it
        try:
            image_binary = bytes(image_binary)
        except TypeError:
            raise TypeError(f"Cannot convert {type(image_binary)} to bytes")
    
    
    # Now encode to base64
    image_base64 = base64.b64encode(image_binary).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

def post_http_request(
    model: str,
    prompts: List[str],
    temperature: float = 1.0,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    guided_choice: List[str] = None,
    image_urls: List[List[str]] = None,  # Changed: List of lists for multiple images per prompt
) -> requests.Response:
    """
    Send POST request to chat completions endpoint.
    
    Args:
        model: Model name/identifier
        prompts: List of text prompts
        temperature: Sampling temperature
        api_url: API endpoint URL
        guided_choice: Optional guided choices
        image_urls: Optional list of image URL lists (one list per prompt)
    """
    messages_list = []
    
    for i, prompt in enumerate(prompts):
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add images if provided
        if image_urls and i < len(image_urls) and image_urls[i]:
            for img_url in image_urls[i]:
                if img_url:  # Skip None values
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
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

def execute_batch_v2_with_images(
    model,
    fields: List[Dict[str, any]],
    query: str,
    typed_fields: List[Tuple[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1"
) -> List[str]:
    """
    Execute batch queries with support for text and image fields.
    
    Args:
        model: The LLM model
        fields: List of dictionaries containing field values
        query: The query template with typed placeholders
        typed_fields: List of (field_name, field_type) tuples
        system_prompt: System prompt
        guided_choice: Optional guided choices
        base_url: API base URL
    """
    # Build user prompts and collect image URLs
    user_prompts = []
    all_image_urls = []
    
    for field_dict in fields:
        # Replace text placeholders in the query
        user_prompt = query
        image_urls_for_this_prompt = []
        
        for field_name, field_type in typed_fields:
            placeholder = f"{{{field_type}:{field_name}}}"
            
            if field_type == "text":
                value = field_dict.get(field_name, "")
                user_prompt = user_prompt.replace(placeholder, str(value))
            
            elif field_type == "image":
                user_prompt = user_prompt.replace(placeholder, "[image]")
                image_binary = field_dict.get(field_name)[0]
                if image_binary is not None:
                    image_url = convert_image_to_base64_url(image_binary)
                    image_urls_for_this_prompt.append(image_url)
        
        user_prompts.append(user_prompt)
        all_image_urls.append(image_urls_for_this_prompt if image_urls_for_this_prompt else None)
    
    # Generate full prompts with system prompt
    prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
               for user_prompt in user_prompts]
    
    outputs = []
    if base_url:
        # For each prompt, send a separate HTTP POST request
        for i, prompt in enumerate(prompts):
            print(prompt)
            response = post_http_request(
                model.model,
                [prompt],
                temperature=0,
                api_url=(base_url + "/chat/completions"),  # Changed endpoint
                guided_choice=guided_choice,
                image_urls=[all_image_urls[i]] if all_image_urls[i] else None
            )
            request_output = json.loads(response.content)
            choices = request_output.get('choices', [])
            
            if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                outputs.append(choices[0]['message']['content'])
            else:
                outputs.append(None)
        
        return outputs
    else:
        # Use local engine (assuming it supports images)
        request_outputs = model.engine.generate(
            prompts=prompts,
            sampling_params=model.sampling_params
        )
        return [output for output in request_outputs]


# Helper function (placeholder - implement based on your existing code)
def _generate_prompt(user_prompt: str, system_prompt: str) -> str:
    """Generate the full prompt with system and user components."""
    return f"{system_prompt}\n\n{user_prompt}"