import sys
import os
import time
import pandas as pd
from typing import Tuple, List, Dict, Any, Callable
from pyspark.sql.functions import array_contains, lower, trim, col, when, expr, regexp_replace, get_json_object
import pandas as pd
import torch
from PIL import Image
import io
import csv
import json
import asyncio
import numpy as np

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

def extract_image_binary_from_scivqa_data(image_path, base_path="/home/haikai/images_train/"):
    """Extract image binary from file path."""
    full_path = os.path.join(base_path, image_path)
    with open(full_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

def extract_image_binary_from_pope_data(image_data):
    """Extract image binary from POPE data format."""
    # Handle dictionary format
    if isinstance(image_data, dict):
        # Common keys in HuggingFace datasets
        if 'bytes' in image_data:
            return image_data['bytes']
        elif 'path' in image_data:
            with open(image_data['path'], 'rb') as f:
                return f.read()
        elif 'content' in image_data:
            return image_data['content']
        else:
            # Print the keys to help debug
            print(f"Unknown dict keys: {image_data.keys()}")
            raise ValueError(f"Cannot extract image from dict with keys: {image_data.keys()}")
    
    # Handle list/tuple format
    if isinstance(image_data, (list, tuple)):
        return image_data[0] if len(image_data) > 0 else image_data
    
    # Handle direct bytes
    if isinstance(image_data, bytes):
        return image_data
    
    # Handle PIL Image
    if hasattr(image_data, 'save'):  # PIL Image object
        buffer = io.BytesIO()
        image_data.save(buffer, format='PNG')
        return buffer.getvalue()
    
    return image_data

def convert_to_string(value):
    """Convert value to string, handling lists, arrays, and other types."""
    # Handle None/NaN
    if value is None:
        return ""
    
    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        # Convert array elements to string and join
        return " ".join([str(item) for item in value.flatten() if item is not None])
    
    # Handle pandas NA types
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        # pd.isna might fail on arrays/lists, continue processing
        pass
    
    # Handle list/tuple types
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        # Convert each element to string and join
        return " ".join([str(item) for item in value if item is not None])
    
    # Handle other types
    return str(value)

@torch.no_grad()
def getTokenCount(model, vision_tower, tokenizer, image_binary, text):
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    
    model_device = vision_tower.device
    
    # Extract visual features
    image_forward_outs = vision_tower.vision_tower(
        images.to(device=model_device, dtype=vision_tower.dtype), 
        output_hidden_states=True
    )
    image_outputs = vision_tower.feature_select(image_forward_outs)
    image_features = image_outputs.to(images.dtype)
    
    B, N, C = image_features.shape

    text_inputs = tokenizer(
            text=text, 
            return_tensors="pt", 
            padding=True
            ).to(device=model_device)
        
    text_embeds = model.get_input_embeddings()(text_inputs.input_ids)
    text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)
    M = text_embeds.shape[1]  # Number of text tokens
    
    # Calculate total pixels
    width, height = image.size
    total_pixels = width * height
    
    return N, M, total_pixels

def process_parquet_file(model, vision_tower, tokenizer, parquet_path, 
                        image_column, text_columns, 
                        image_id_column=None,
                        output_csv="token_counts.csv",
                        output_redundancy_csv=None,
                        text_separator=" "):
    """
    Process parquet file (for POPE/VQAv2/VQAtext format with embedded images).
    
    Args:
        image_id_column: Column name to use for calculating image redundancy
        output_redundancy_csv: CSV file to save image redundancy info (optional)
    """
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    
    # Calculate image redundancy
    if image_id_column and image_id_column in df.columns:
        image_counts = df[image_id_column].value_counts().to_dict()
        print(f"Calculated image redundancy from column: {image_id_column}")
        print(f"Unique images: {len(image_counts)}")
        print(f"Total rows: {len(df)}")
        print(f"Average redundancy: {len(df) / len(image_counts):.2f}")
        
        # Save redundancy to separate CSV if specified
        if output_redundancy_csv:
            redundancy_df = pd.DataFrame([
                {'image_id': img_id, 'redundancy': count}
                for img_id, count in image_counts.items()
            ])
            redundancy_df = redundancy_df.sort_values('redundancy', ascending=False)
            redundancy_df.to_csv(output_redundancy_csv, index=False)
            print(f"Image redundancy saved to {output_redundancy_csv}")
    else:
        image_counts = {}
        print(f"Warning: image_id_column '{image_id_column}' not found")
    
    # Convert text_columns to list if it's a single string
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['row_index', 'image_tokens', 'text_tokens', 'total_tokens', 
                        'total_pixels', 'image_width', 'image_height'])
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract image binary
                image_data = row[image_column]
                image_binary = extract_image_binary_from_pope_data(image_data)
                
                # Get image dimensions
                image = Image.open(io.BytesIO(image_binary))
                width, height = image.size
                
                # Combine all text columns
                text_parts = []
                for col_name in text_columns:
                    text_value = row[col_name]
                    if pd.notna(text_value):  # Check if not NaN
                        text_parts.append(str(text_value))
                
                combined_text = text_separator.join(text_parts)
                
                # Get token counts and pixel count
                N, M, total_pixels = getTokenCount(model, vision_tower, tokenizer, image_binary, combined_text)
                
                # Write to CSV
                writer.writerow([idx, N, M, N + M, total_pixels, width, height])
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} rows")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                writer.writerow([idx, 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])
    
    print(f"Token counts saved to {output_csv}")

def process_json_file_with_spark(model, vision_tower, tokenizer, json_path, 
                                 image_column, text_columns, 
                                 image_id_column=None,
                                 image_base_path="/home/haikai/images_train/",
                                 output_csv="token_counts.csv",
                                 output_redundancy_csv=None,
                                 text_separator=" ",
                                 limit=None):
    """
    Process JSON file using Spark and save token counts to CSV.
    
    Args:
        image_id_column: Column name to use for calculating image redundancy
        output_redundancy_csv: CSV file to save image redundancy info (optional)
    """
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("TokenCountAnalysis") \
        .getOrCreate()
    
    # Read JSON file
    df_spark = spark.read \
        .option("multiLine", "true") \
        .option("encoding", "UTF-8") \
        .json(json_path)
    
    if limit:
        df_spark = df_spark.limit(limit)
    
    df_spark = df_spark.cache()
    
    # Convert to Pandas
    df = df_spark.toPandas()
    
    print(f"Total rows to process: {len(df)}")
    
    # Calculate image redundancy
    if image_id_column and image_id_column in df.columns:
        image_counts = df[image_id_column].value_counts().to_dict()
        print(f"Calculated image redundancy from column: {image_id_column}")
        print(f"Unique images: {len(image_counts)}")
        print(f"Total rows: {len(df)}")
        print(f"Average redundancy: {len(df) / len(image_counts):.2f}")
        
        # Save redundancy to separate CSV if specified
        if output_redundancy_csv:
            redundancy_df = pd.DataFrame([
                {'image_id': img_id, 'redundancy': count}
                for img_id, count in image_counts.items()
            ])
            redundancy_df = redundancy_df.sort_values('redundancy', ascending=False)
            redundancy_df.to_csv(output_redundancy_csv, index=False)
            print(f"Image redundancy saved to {output_redundancy_csv}")
    else:
        image_counts = {}
        print(f"Warning: image_id_column '{image_id_column}' not found")
    
    # Convert text_columns to list if it's a single string
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['row_index', 'image_tokens', 'text_tokens', 'total_tokens', 
                        'total_pixels', 'image_width', 'image_height'])
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract image path and read binary
                image_path = row[image_column]
                image_binary = extract_image_binary_from_scivqa_data(image_path, image_base_path)
                
                # Get image dimensions
                image = Image.open(io.BytesIO(image_binary))
                width, height = image.size
                
                # Combine all text columns (handling lists and arrays)
                text_parts = []
                for col_name in text_columns:
                    text_value = row[col_name]
                    # Convert to string (handles lists, arrays, etc.)
                    text_str = convert_to_string(text_value)
                    if text_str:  # Only add non-empty strings
                        text_parts.append(text_str)
                
                combined_text = text_separator.join(text_parts)
                
                # Get token counts and pixel count
                N, M, total_pixels = getTokenCount(model, vision_tower, tokenizer, image_binary, combined_text)
                
                # Write to CSV
                writer.writerow([idx, N, M, N + M, total_pixels, width, height])
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} rows")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
                writer.writerow([idx, 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])
    
    print(f"Token counts saved to {output_csv}")
    spark.stop()

# Load models once
vision_tower, model, tokenizer = load_vision_models(device='cuda')

# ===== Example 1: POPE Dataset =====
# POPE_PATH = "/home/haikai/haikai/entropyTest/POPE.parquet"
# process_parquet_file(
#     model=model,
#     vision_tower=vision_tower,
#     tokenizer=tokenizer,
#     parquet_path=POPE_PATH,
#     image_column="image",
#     text_columns="question",
#     image_id_column="image_source",  # POPE uses image_source
#     output_csv="POPE.csv",
#     output_redundancy_csv="POPE_redundancy.csv",  # Separate redundancy file
#     text_separator=" "
# )

# # ===== Example 2: VQAtext Dataset =====
VQATEXT_PATH = "/home/haikai/LLM-Multimodal/VQAtext/validation-00000-of-00003.parquet"
process_parquet_file(
    model=model,
    vision_tower=vision_tower,
    tokenizer=tokenizer,
    parquet_path=VQATEXT_PATH,
    image_column="image",
    text_columns="question",
    image_id_column="image_id",  # VQAtext uses image_id
    output_csv="VQAtext.csv",
    output_redundancy_csv="VQAtext_redundancy.csv",
    text_separator=" "
)

# ===== Example 3: VQAv2 Dataset =====
# VQAV2_PATH = "/home/haikai/LLM-Multimodal/VQAv2/validation-00000-of-00068.parquet"
# process_parquet_file(
#     model=model,
#     vision_tower=vision_tower,
#     tokenizer=tokenizer,
#     parquet_path=VQAV2_PATH,
#     image_column="image",
#     text_columns="question",
#     image_id_column="image_id",  # VQAv2 uses image_id
#     output_csv="VQAv2.csv",
#     output_redundancy_csv="VQAv2_redundancy.csv",
#     text_separator=" "
# )

# # ===== Example 4: SciVQA Dataset =====
# JSON_PATH = "/home/haikai/train_2025-07-03_09-06.json"
# process_json_file_with_spark(
#     model=model,
#     vision_tower=vision_tower,
#     tokenizer=tokenizer,
#     json_path=JSON_PATH,
#     image_column="image_file",
#     text_columns=["caption", "figure_type", "qa_pair_type", "question", "answer_options"],
#     image_id_column="image_file",  # SciVQA uses image_file
#     image_base_path="/home/haikai/images_train/",
#     output_csv="scivqa_token_counts.csv",
#     output_redundancy_csv="scivqa_redundancy.csv",
#     text_separator=" ",
#     limit=None
# )