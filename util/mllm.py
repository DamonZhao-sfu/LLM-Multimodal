import torch
import torch.nn.functional as F
from PIL import Image
from util.cdencoder import CLIPVisionTower
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
import time
import torch
import numpy as np
from PIL import Image
import requests
import json
import io
import base64
import os
import csv
import pandas as pd
import json
import time
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from util.prompt import DEFAULT_SYSTEM_PROMPT
from util.utils import _generate_prompt

def getLocalModal():
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        attn_implementation="eager"
    )
    return model


vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
class MockArgs:
    def __init__(self):
        self.mm_vision_select_layer = -2
        self.mm_vision_select_feature = 'patch'

mock_args = MockArgs()
vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
vision_tower = vision_tower.to("cuda")

def encode_image_embedding_to_base64(image_embedding):
    """
    Encode image embedding tensor to base64 string
    
    Args:
        image_embedding: PyTorch tensor containing image embeddings
        
    Returns:
        base64 encoded string of the tensor
    """
    buffer = io.BytesIO()
    torch.save(image_embedding, buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    base64_image_embedding = base64.b64encode(binary_data).decode('utf-8')
    return base64_image_embedding


def call_vllm_api_with_embeds(image_embedding, question="What's in this image?", model="llava-hf/llava-1.5-7b-hf", api_url="http://localhost:8005"):
    # Encode image embedding
    base64_image_embedding = encode_image_embedding_to_base64(image_embedding)
    
    # Prepare the request payload
    embeds = {
        "type": "image_embeds",
        "image_embeds": base64_image_embedding
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    embeds,
                    {
                        "type": "text",
                        "text": question,
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "guided_choice": ["yes", "no"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {api_url}/v1/chat/completions...")
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return None


def getOriginalVisualToken(model, image_binary, texts, keep_ratio=0.25, lambda_val=0.1, recovery_ratio=0.1):
    # Load and preprocess image
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    image_stream = torch.cuda.Stream()
    text_stream = torch.cuda.Stream()
    
    model_device = vision_tower.device
    
    # Process image features
    with torch.cuda.stream(image_stream):
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
      
    torch.cuda.synchronize()
    
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features).detach().cpu()
    return image_features


def getPrunedVisualTokenVisPruner_optimized(model, image_binary, texts, keep_ratio=0.125, 
                                          important_ratio=0.6, recovery_ratio=0.1, text_guidance_weight=0.5):
    """
    Highly optimized version of VisPruner with multiple speedup techniques:
    1. Minimized GPU-CPU transfers
    2. In-place operations where possible  
    3. Vectorized operations
    4. Memory-efficient tensor operations
    5. Early termination optimizations
    6. Approximate similarity computation for large token sets
    """
    
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    
    model_device = vision_tower.device
    dtype = vision_tower.dtype
    
    # Process image features and get attention from visual encoder
    with torch.no_grad():
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Extract [CLS] attention more efficiently
        attentions = image_forward_outs.attentions
        print("len of attention " + str(len(attentions)))
        print("attentions[-1]" + str(attentions[-1]))


        if len(attentions) > 1:
            # Use penultimate layer, average across heads in one operation
            cls_attention = attentions[-2].squeeze(0).mean(dim=0)[0, 1:]  # [num_patches]
        else:
            cls_attention = attentions[-1].squeeze(0).mean(dim=0)[0, 1:]
        
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(dtype)  # Keep on GPU
    
    B, N, C = image_features.shape
    
    # Ensure consistent dtypes - convert image_features to float16 and projector to same device/dtype
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(device=model_device, dtype=torch.float16)
    projected_features = model.multi_modal_projector(image_features)  # [B, N, hidden_dim]

    # Pre-calculate token counts to avoid repeated calculations
    num_tokens_to_keep = min(int(keep_ratio * N), N)
    num_important_tokens = int(num_tokens_to_keep * important_ratio)
    num_diverse_tokens = num_tokens_to_keep - num_important_tokens
    
    # Early exit if keeping all tokens
    if num_tokens_to_keep >= N:
        image_features = model.multi_modal_projector(image_features)
        return image_features.detach().cpu()
    
    # Step 1: Select important tokens (vectorized topk)
    _, important_indices = torch.topk(cls_attention, num_important_tokens, dim=-1)
    
    # Create boolean mask for remaining indices (more memory efficient)
    all_mask = torch.ones(N, dtype=torch.bool, device=model_device)
    all_mask[important_indices] = False
    remaining_indices = torch.nonzero(all_mask, as_tuple=True)[0]
    
    begin = time.time()
    # Step 2: Optimized diverse token selection
    diverse_indices = torch.empty(0, dtype=torch.long, device=model_device)
    
    if num_diverse_tokens > 0 and len(remaining_indices) > 0:
        if len(remaining_indices) <= num_diverse_tokens:
            diverse_indices = remaining_indices
        else:
            # For diversity selection, also consider text guidance
            if texts is not None and text_guidance_weight > 0:
                # Weight the remaining features by their text relevance
                remaining_text_scores = text_visual_norm[remaining_indices] if 'text_visual_norm' in locals() else torch.ones(len(remaining_indices), device=model_device)
                
                # Apply weighted sampling probability
                sampling_weights = 0.7 + 0.3 * remaining_text_scores  # Ensure minimum probability
                sampling_probs = sampling_weights / sampling_weights.sum()
                
                # Use approximate sampling for large sets
                if len(remaining_indices) > 500:
                    # Sample based on text relevance + randomness
                    sample_size = min(num_diverse_tokens * 3, len(remaining_indices))
                    sampled_idx = torch.multinomial(sampling_probs, sample_size, replacement=False)
                    sampled_indices = remaining_indices[sampled_idx]
                    
                    remaining_features = projected_features[0, sampled_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = similarity_based_duplicate_removal_fast(
                        remaining_features, min(num_diverse_tokens, len(sampled_indices))
                    )
                    diverse_indices = sampled_indices[diverse_idx]
                else:
                    # Full computation with text guidance
                    remaining_features = projected_features[0, remaining_indices, :]
                    remaining_features = F.normalize(remaining_features, p=2, dim=-1)
                    
                    diverse_idx = text_guided_diversity_selection(
                        remaining_features, sampling_weights, num_diverse_tokens
                    )
                    diverse_indices = remaining_indices[diverse_idx]
                 
    # Combine and sort indices
    selected_indices = torch.cat([important_indices, diverse_indices])
    selected_indices = torch.sort(selected_indices)[0]
    
    # Extract selected features
    image_features_selected = projected_features[:, selected_indices, :]
    
    end = time.time()

    if texts is not None and recovery_ratio > 0:
        # Process text more efficiently
        text_embeds = process_text_efficiently(texts, vision_tower, model_device)
        
        if text_embeds is not None:
            # Get pruned indices using boolean indexing
            selected_mask = torch.zeros(N, dtype=torch.bool, device=model_device)
            selected_mask[selected_indices] = True
            pruned_indices = torch.nonzero(~selected_mask, as_tuple=True)[0]
            
            if len(pruned_indices) > 0:
                num_tokens_to_recover = min(int(recovery_ratio * N), len(pruned_indices))
                
                if num_tokens_to_recover > 0:
                    # Efficient attention computation
                    if text_embeds.shape[-1] != projected_features.shape[-1]:  # FIX: Compare with projected_features
                        # Use smaller projection layer
                        projection_layer = torch.nn.Linear(
                            text_embeds.shape[-1], projected_features.shape[-1],  # FIX: Use projected_features dimension
                            bias=False  # Remove bias for speed
                        ).to(model_device, dtype=torch.float16)
                        text_embeds = projection_layer(text_embeds)
                    
                    # Compute attention only for pruned tokens (memory efficient)
                    pruned_features = projected_features[:, pruned_indices, :]  # FIX: Use projected_features instead of image_features
                    attention_scores = torch.einsum('btc,bpc->btp', text_embeds, pruned_features)
                    attention_scores = F.softmax(attention_scores, dim=-1).mean(dim=(0, 1))
                    
                    # Recover tokens
                    _, recovery_idx = torch.topk(attention_scores, num_tokens_to_recover)
                    recovery_indices = pruned_indices[recovery_idx]
                    
                    # Concatenate efficiently - FIX: Use projected_features for recovery
                    image_features_recovered = projected_features[:, recovery_indices, :]
                    image_features_selected = torch.cat(
                        [image_features_selected, image_features_recovered], dim=1
                    )
    # Single GPU-CPU transfer at the end
    result = image_features_selected.detach().cpu()
    
    print(f"Final output shape: {result.shape}")
    print(f"Kept {result.shape[1]} out of {N} tokens ({result.shape[1]/N*100:.1f}%)")
    
    return result


def post_http_request_with_embeds(
    model: str,
    prompts: List[str],
    temperature: float = 1.0,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    guided_choice: List[str] = None,
    image_embeddings: List[torch.Tensor] = None,  # Changed: List of embedding tensors
) -> requests.Response:
    """
    Send POST request to chat completions endpoint with image embeddings.
    
    Args:
        model: Model name/identifier
        prompts: List of text prompts
        temperature: Sampling temperature
        api_url: API endpoint URL
        guided_choice: Optional guided choices
        image_embeddings: Optional list of pruned image embedding tensors
    """
    messages_list = []
    
    for i, prompt in enumerate(prompts):
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add image embeddings if provided
        if image_embeddings and i < len(image_embeddings) and image_embeddings[i] is not None:
            # Convert tensor to list for JSON serialization

            embedding_data = image_embeddings[i]
            embedding = encode_image_embedding_to_base64(embedding_data)
            content.append({
                "type": "image_embeds",
                "image_embeds": embedding
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

def execute_batch_v2_with_pruned_embeddings(
    model,
    modelname,
    fields: List[Dict[str, any]],
    query: str,
    typed_fields: List[Tuple[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
    keep_ratio: float = 0.5,
    recovery_ratio: float = 0.0
) -> List[str]:
    """
    Execute batch queries with pruned image embeddings.
    
    Args:
        model: The LLM model (used for pruning)
        fields: List of dictionaries containing field values
        query: The query template with typed placeholders
        typed_fields: List of (field_name, field_type) tuples
        system_prompt: System prompt
        guided_choice: Optional guided choices
        base_url: API base URL
        keep_ratio: Ratio of tokens to keep during pruning
        recovery_ratio: Ratio of tokens to recover
    """
    # Build user prompts and generate pruned embeddings
    print("execute_batch_v2_with_pruned_embeddings")

    user_prompts = []
    all_pruned_embeddings = []
    
    for field_dict in fields:
        # Replace text placeholders in the query
        user_prompt = query
        pruned_embeddings_for_this_prompt = []
        
        for field_name, field_type in typed_fields:
            placeholder = f"{{{field_type}:{field_name}}}"
            
            if field_type == "text":
                value = field_dict.get(field_name, "")
                user_prompt = user_prompt.replace(placeholder, str(value))
            
            elif field_type == "image":
                user_prompt = user_prompt.replace(placeholder, "[image]")
                image_data = field_dict.get(field_name)
                
                if image_data is not None:
                    # Extract image binary
                    image_binary = extract_image_binary_from_pope_data(image_data)
                    
                    # Generate pruned embeddings
                    embed_start = time.time()
                    reduced_tokens = getOriginalVisualToken(
                        model,
                        image_binary,
                        user_prompt
                    )
                    # reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
                    #     model,
                    #     image_binary,
                    #     user_prompt,  # Use the current prompt as question
                    #     keep_ratio=keep_ratio,
                    #     important_ratio=0.6,
                    #     recovery_ratio=recovery_ratio
                    # )
                    embed_end = time.time()
                    
                    #print(f"Embedding generation time: {embed_end - embed_start:.3f}s")
                    #print(f"Pruned tokens: {reduced_tokens.shape[1]}")
                    
                    # Convert to float16 for efficiency
                    pruned_embeddings_for_this_prompt.append(reduced_tokens.to(torch.float16))
        
        user_prompts.append(user_prompt)
        all_pruned_embeddings.append(
            pruned_embeddings_for_this_prompt[0] if pruned_embeddings_for_this_prompt else None
        )
    
    # Generate full prompts with system prompt
    prompts = [_generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) 
               for user_prompt in user_prompts]
    
    outputs = []
    if base_url:
        # For each prompt, send a separate HTTP POST request with embeddings
        for i, prompt in enumerate(prompts):

            api_start = time.time()
            response = post_http_request_with_embeds(
                modelname,
                [prompt],
                temperature=0,
                api_url=(base_url + "/chat/completions"),
                guided_choice=guided_choice,
                image_embeddings=[all_pruned_embeddings[i]] if all_pruned_embeddings[i] is not None else None
            )
            api_end = time.time()
                        
            request_output = json.loads(response.content)
            choices = request_output.get('choices', [])
            
            if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                outputs.append(choices[0]['message']['content'])
            else:
                outputs.append(None)
        
        return outputs


# Helper function to extract image binary (you'll need to implement this based on your data format)
def extract_image_binary_from_pope_data(image_data):
    """
    Extract image binary from your data format.
    Implement this based on how your image data is structured.
    """
    if isinstance(image_data, (list, tuple)):
        return image_data[0]
    return image_data