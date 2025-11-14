import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from util.cdencoder import CLIPVisionTower
from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
import time
import torch
from PIL import Image
import requests
import json
import io
import base64
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from util.prompt import DEFAULT_SYSTEM_PROMPT
from util.utils import _generate_prompt
import gc


def load_vision_models(device='cuda'):
    """Load vision tower and projection model"""
    print(f"[Model Loading] Loading vision tower on {device}...")
    
    # Set CUDA device
    #torch.cuda.set_device(0)
    vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
    MODEL_PATH = "/data/models/llava-1.5-7b-hf"

    # Load vision tower
    #vision_tower_name = "/scratch/hpc-prf-haqc/haikai/hf-cache/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    class MockArgs:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
    
    mock_args = MockArgs()
    vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
    vision_tower = vision_tower.to('cuda')
    vision_tower.vision_tower.config._attn_implementation = "eager"
    vision_tower.vision_tower.config.output_attentions = True
    # Load LLaVA model for projection layer
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map='cuda',
        attn_implementation="eager"
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    tokenizer = processor.tokenizer
    
    print(f"[Model Loading] Vision tower loaded successfully on {device}")
    
    return vision_tower, model, tokenizer


def load_vision_models_only(device='cuda'):
    vision_tower_name = "/data/models/clip-vit-p14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
    class MockArgs:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
    
    mock_args = MockArgs()
    vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
    vision_tower = vision_tower.to('cuda')
    # vision_tower.vision_tower.config._attn_implementation = "eager"
    # vision_tower.vision_tower.config.output_attentions = True
    
    return vision_tower

def cleanup_vision_models(vision_tower, model):
    """Clean up GPU memory used by vision models"""
    print(f"[Memory Cleanup] Releasing GPU memory...")
    
    # Move models to CPU first
    if vision_tower is not None:
        vision_tower = vision_tower.cpu()
        del vision_tower
    
    if model is not None:
        model = model.cpu()
        del model
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print(f"[Memory Cleanup] GPU memory released")


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


def getOriginalVisualToken(model, vision_tower, image_binary):
    
    preprocess_start = time.time()   
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    preprocess_end = time.time()   
    preprocess_time = preprocess_end - preprocess_start

    image_stream = torch.cuda.Stream()
    
    model_device = vision_tower.device
    encode_begin = time.time()
    with torch.cuda.stream(image_stream):
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
      
    torch.cuda.synchronize()
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features).detach().cpu() 
    encode_end = time.time()
    encode_time = encode_end - encode_begin

    return image_features,preprocess_time,encode_time


def process_text_efficiently(texts, vision_tower, model_device):
    try:
        with torch.no_grad():
            text_inputs = vision_tower.text_tokenizer(text=texts, return_tensors="pt")
            
            # More efficient padding computation
            input_length = text_inputs.input_ids.shape[1]
            max_pos = vision_tower.max_position_embeddings
            
            if input_length <= max_pos:
                # No segmentation needed
                padding_needed = max_pos - input_length
                if padding_needed > 0:
                    pad_tensor = torch.zeros((1, padding_needed), dtype=text_inputs.input_ids.dtype)
                    text_inputs = {
                        k: torch.cat([v, pad_tensor], dim=1).to(model_device)
                        for k, v in text_inputs.items()
                    }
                else:
                    text_inputs = {k: v.to(model_device) for k, v in text_inputs.items()}
            else:
                # Efficient segmentation
                text_segment = (input_length - 1) // max_pos + 1
                total_length = max_pos * text_segment
                padding_needed = total_length - input_length
                
                text_inputs = {
                    k: torch.cat([v, v.new_zeros((v.shape[0], padding_needed))], dim=1)
                    .reshape(-1, max_pos).to(model_device)
                    for k, v in text_inputs.items()
                }
            
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
            
            # Efficient reshaping
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(0)
            elif text_embeds.dim() > 3:
                batch_size = 1
                seq_len = text_embeds.numel() // (batch_size * text_embeds.size(-1))
                text_embeds = text_embeds.view(batch_size, seq_len, -1)
            
            return text_embeds.to(torch.float16)
            
    except Exception as e:
        print(f"Text processing failed: {e}")
        return None

def text_guided_diversity_selection(features, text_weights, num_tokens):
    """
    Select diverse tokens with text guidance weighting
    """
    selected_indices = []
    available_mask = torch.ones(len(features), dtype=torch.bool, device=features.device)
    
    # Start with highest text-weighted token
    weighted_scores = text_weights.clone()
    
    for _ in range(num_tokens):
        if available_mask.sum() == 0:
            break
            
        # Select token with highest current score
        available_scores = weighted_scores.clone()
        available_scores[~available_mask] = -float('inf')
        selected_idx = available_scores.argmax()
        
        selected_indices.append(selected_idx)
        available_mask[selected_idx] = False
        
        if available_mask.sum() == 0:
            break
        
        # Update scores based on similarity to selected token
        selected_feature = features[selected_idx:selected_idx+1]
        similarities = torch.mm(features, selected_feature.t()).squeeze(-1)
        
        # Reduce scores for similar tokens (diversity penalty)
        penalty = 0.8 * similarities * available_mask.float()
        weighted_scores = weighted_scores - penalty
    
    return torch.tensor(selected_indices, device=features.device)

def similarity_based_duplicate_removal_fast(features, num_to_keep):
    """
    Optimized similarity-based token selection with approximate methods for speed.
    """
    n_tokens, dim = features.shape
    
    if n_tokens <= num_to_keep:
        return torch.arange(n_tokens, device=features.device)
    
    # For very large token sets, use clustering-based approximation
    if n_tokens > 1000:
        return cluster_based_selection(features, num_to_keep)
    
    # For medium-sized sets, use optimized greedy selection
    selected_idx = []
    selected_idx.append(0)  # Start with first token
    
    remaining_mask = torch.ones(n_tokens, dtype=torch.bool, device=features.device)
    remaining_mask[0] = False
    
    # Precompute all pairwise similarities (batch operation)
    similarity_matrix = torch.mm(features, features.t())
    
    for _ in range(num_to_keep - 1):
        remaining_indices = torch.nonzero(remaining_mask, as_tuple=True)[0]
        if len(remaining_indices) == 0:
            break
        
        # Find token with minimum maximum similarity to selected tokens
        selected_similarities = similarity_matrix[remaining_indices][:, selected_idx]
        max_similarities = selected_similarities.max(dim=1)[0]
        min_idx = max_similarities.argmin()
        
        next_token_idx = remaining_indices[min_idx].item()
        selected_idx.append(next_token_idx)
        remaining_mask[next_token_idx] = False
    
    return torch.tensor(selected_idx, device=features.device)


def getPrunedVisualTokenVisPruner_optimized(model, vision_tower, image_binary, texts, keep_ratio=0.125, 
                                          important_ratio=0.6, recovery_ratio=0.1, text_guidance_weight=0.5):
    preprocess_start = time.time()   
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    preprocess_end = time.time()   
    preprocess_time = preprocess_end - preprocess_start
    
    encode_begin = time.time()
    model_device = vision_tower.device
    dtype = vision_tower.dtype
    with torch.no_grad():
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=model_device, dtype=dtype),
            output_hidden_states=True,
            output_attentions=True
        )
        # Extract [CLS] attention more efficiently
        attentions = image_forward_outs.attentions
        if len(attentions) > 1:
            # Use penultimate layer, average across heads in one operation
            cls_attention = attentions[-2].squeeze(0).mean(dim=0)[0, 1:]  # [num_patches]
        else:
            cls_attention = attentions[-1].squeeze(0).mean(dim=0)[0, 1:]
        
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(dtype)  # Keep on GPU
    
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(device=model_device, dtype=torch.float16)
    projected_features = model.multi_modal_projector(image_features)  # [B, N, hidden_dim]
    encode_end = time.time()
    encode_time = encode_end - encode_begin

    prune_begin = time.time()

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
    result = image_features_selected.detach().cpu()
    prune_end = time.time()
    prune_time = prune_end-prune_begin

    return result, preprocess_time, encode_time, prune_time

def post_http_request_with_embeds(
    model: str,
    prompts: List[str],
    temperature: float = 1.0,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    guided_choice: List[str] = None,
    image_embeddings: List[torch.Tensor] = None,
    answer_schema: Optional[Dict[str, Any]] = None,  # Optional JSON schema parameter
) -> requests.Response:
    messages_list = []
    
    for i, prompt in enumerate(prompts):
        content = []
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        if image_embeddings and i < len(image_embeddings) and image_embeddings[i] is not None:
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

    pload = {
        "model": model,
        "messages": messages_list,
        "temperature": temperature,
    }
    
    # Add guided_json only if answer_schema is provided
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


# def post_http_request_with_embeds(
#     model: str,
#     prompts: List[str],
#     temperature: float = 1.0,
#     api_url: str = "http://localhost:8000/v1/chat/completions",
#     guided_choice: List[str] = None,
#     image_embeddings: List[torch.Tensor] = None,  # Changed: List of embedding tensors
# ) -> requests.Response:
#     messages_list = []
    
#     for i, prompt in enumerate(prompts):
#         content = []
        
#         content.append({
#             "type": "text",
#             "text": prompt
#         })
        
#         if image_embeddings and i < len(image_embeddings) and image_embeddings[i] is not None:
#             embedding_data = image_embeddings[i]
#             embedding = encode_image_embedding_to_base64(embedding_data)
#             content.append({
#                 "type": "image_embeds",
#                 "image_embeds": embedding
#             })
        
#         messages_list.append({
#             "role": "user",
#             "content": content
#         })
    
#     answer_schema = {
#         "type": "object",
#         "properties": {
#             "Answer": {
#                 "type": "string"
#             }
#         },
#         "required": ["Answer"]
#     }

#     pload = {
#         "model": model,
#         "messages": messages_list,
#         "temperature": temperature,
#         "guided_json": answer_schema,  # vLLM's guided_json parameter

#     }
#     if guided_choice is not None and len(guided_choice) > 0:
#         pload["guided_choice"] = guided_choice


#     headers = {"Content-Type": "application/json"}
#     req = requests.Request('POST', api_url, headers=headers, data=json.dumps(pload))
#     prepared = req.prepare()

#     with requests.Session() as session:
#         response = session.send(prepared)

#     return response

def execute_batch_v2_with_pruned_embeddings(
    modelname,
    fields: List[Dict[str, any]],
    query: str,
    keep_ratio: float,
    typed_fields: List[Tuple[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    guided_choice: List[str] = None,
    base_url: str = "http://localhost:8000/v1",
) -> List[str]:
    """
    Execute batch queries with pruned image embeddings.
    Models are loaded at the start and cleaned up at the end.
    """
    # Load models at the beginning
    vision_tower, model = load_vision_models(device='cuda:0')
    
    try:
        # Build user prompts and generate pruned embeddings
        user_prompts = []
        all_pruned_embeddings = []
        
        for field_dict in fields:
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
                        image_binary = extract_image_binary_from_pope_data(image_data)
                        
                        if keep_ratio == 1:
                            reduced_tokens = getOriginalVisualToken(
                                model,
                                vision_tower,
                                image_binary
                            )
                        else:
                            reduced_tokens = getPrunedVisualTokenVisPruner_optimized(
                                model,
                                vision_tower,
                                image_binary,
                                user_prompt,
                                keep_ratio=keep_ratio
                            )
                        
                        pruned_embeddings_for_this_prompt.append(reduced_tokens.to(torch.float16))
            
            user_prompts.append(user_prompt)
            all_pruned_embeddings.append(
                pruned_embeddings_for_this_prompt[0] if pruned_embeddings_for_this_prompt else None
            )
        
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
        pass
        # Always clean up models, even if an error occurred
        #(vision_tower, model)

# Helper function to extract image binary (you'll need to implement this based on your data format)
def extract_image_binary_from_pope_data(image_data):
    """
    Extract image binary from your data format.
    Implement this based on how your image data is structured.
    """
    if isinstance(image_data, (list, tuple)):
        return image_data[0]
    return image_data