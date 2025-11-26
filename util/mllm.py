import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from util.cdencoder import CLIPVisionTower
from transformers import LlavaNextProcessor
#from util.pruMerge import LlavaNextForConditionalGeneration
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import LlavaNextForConditionalGeneration
from transformers import AutoProcessor, LlavaForConditionalGeneration
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
import numpy as np

def load_vision_models_llava_onevision(device='cuda'):
    """Load vision tower and projection model"""
    print(f"[Model Loading] Loading vision tower on {device}...")
    
    # Set CUDA device
    #torch.cuda.set_device(0)
    vision_tower_name = "/scratch/hpc-prf-haqc/haikai/hf-cache/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
    MODEL_PATH = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    class MockArgs:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
    
    mock_args = MockArgs()
    vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
    vision_tower = vision_tower.to('cuda')
    vision_tower.vision_tower.config._attn_implementation = "eager"
    vision_tower.vision_tower.config.output_attentions = True
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map='cuda',
        attn_implementation="eager"
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    tokenizer = processor.tokenizer
    
    print(f"[Model Loading] Vision tower loaded successfully on {device}")
    
    return vision_tower, model, processor



def load_vision_models_llava_next(device='cuda'):
    """Load vision tower and projection model"""
    print(f"[Model Loading] Loading vision tower on {device}...")
    
    # Set CUDA device
    #torch.cuda.set_device(0)
    vision_tower_name = "/scratch/hpc-prf-haqc/haikai/hf-cache/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
    MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

    class MockArgs:
        def __init__(self):
            self.mm_vision_select_layer = -2
            self.mm_vision_select_feature = 'patch'
    
    mock_args = MockArgs()
    vision_tower = CLIPVisionTower(vision_tower_name, mock_args, delay_load=False)
    vision_tower = vision_tower.to('cuda')
    vision_tower.vision_tower.config._attn_implementation = "eager"
    vision_tower.vision_tower.config.output_attentions = True
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map='cuda',
        attn_implementation="eager"
    )

    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)

    tokenizer = processor.tokenizer
    
    print(f"[Model Loading] Vision tower loaded successfully on {device}")
    
    return vision_tower, model, processor

def get_model_class(model_name):
    if "Qwen2.5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    elif "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    elif "llava-1.5" in model_name:
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration
    elif "llava-v1.6" in model_name:
        from transformers import LlavaNextForConditionalGeneration

        return LlavaNextForConditionalGeneration
    else:
        error_msg = f"Unsupported model class for: {model_name}"
        raise ValueError(error_msg)

def load_vision_models(device='cuda'):
    """Load vision tower and projection model"""
    print(f"[Model Loading] Loading vision tower on {device}...")
    
    # Set CUDA device
    #torch.cuda.set_device(0)
    vision_tower_name = "/scratch/hpc-prf-haqc/haikai/hf-cache/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"  # Default CLIP model
    MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

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

def getOriginalVisualToken_Next(model, processor, image_binary):
    preprocess_start = time.time()
    image = Image.open(io.BytesIO(image_binary))
    
    # 1. Use the specific LlavaNextProcessor (handles AnyRes grid splitting)
    inputs = processor(text="", images=image, return_tensors="pt")
    
    # Inputs now contains 'pixel_values' AND 'image_sizes'
    pixel_values = inputs["pixel_values"].to(model.device, dtype=model.dtype)
    image_sizes = inputs["image_sizes"].to(model.device)
    
    preprocess_end = time.time()
    preprocess_time = preprocess_end - preprocess_start
    
    encode_begin = time.time()
    
    # 2. Pass through the Vision Tower
    # We can rely on the model's internal helper to handle the grid/newline logic
    # model.vision_tower is usually the CLIP tower, but we need the model's method
    # to handle the multi-patch logic.
    if pixel_values.ndim == 5:
        batch_size, num_patches, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_patches, c, h, w)
    
    with torch.no_grad():
        # Get the vision features from the backbone
        image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_features = image_outputs.hidden_states[model.config.vision_feature_layer]
        
        # 3. Project features
        image_features = model.multi_modal_projector(selected_image_features)
        
        # 4. CRITICAL: Apply the AnyRes packing logic (Newline insertion)
        # This function reshapes the grid and adds the newline tokens
        # Note: Different HF versions might name this slightly differently.
        # This is based on standard LlavaNext implementation.
        image_features = model.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy
        )

    encode_end = time.time()
    encode_time = encode_end - encode_begin

    # Squeeze is usually not needed here as image_features is (N_tokens, Dim) 
    # but if batch size is 1, ensure it fits your API expectation.
    return image_features, preprocess_time, encode_time

def getOriginalVisualTokenLlavaNext(model, processor, image_binary, **kwargs):

    image = Image.open(io.BytesIO(image_binary))
    inputs = processor(text="", images=image, return_tensors="pt").to(model.device)
    
    pixel_values = inputs.pixel_values
    image_num_patches = None
    #if "LlavaNextForConditionalGeneration" in model.config.architectures and pixel_values.dim() == 5:
    from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
    print("LlavaNextForConditionalGeneration")
    image_num_patches = [
        image_size_to_num_patches(
            image_size=imsize,
            grid_pinpoints=model.config.image_grid_pinpoints,
            patch_size=model.config.vision_config.image_size,
        )
        for imsize in inputs.image_sizes
    ]
    # stacked if input is (batch_size, num_patches, num_channels, height, width)
    _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
    pixel_values = torch.cat(_pixel_values_list, dim=0)

    if hasattr(model, "vision_tower"):  # llava or internvl
        vision_feature_layer = kwargs.get("vision_feature_layer", -1)
        image_features = model.vision_tower(pixel_values=pixel_values.to(device='cuda'), output_hidden_states=True)  # .last_hidden_state
        image_features = image_features.hidden_states[vision_feature_layer]

        if kwargs.get("vision_feature_select_strategy") == "default":
            image_features = image_features[:, 1:]  # remove CLS token
        if image_num_patches is None:  # LlavaForConditionalGeneration
            image_num_patches = image_features.shape[0]
    elif hasattr(model, "visual") or hasattr(model, "vision_tower"):  # qwen
        image_features = model.visual.patch_embed(pixel_values)
        image_num_patches = (kwargs["image_grid_thw"].prod(-1)).tolist()
    else:
        error_msg = "Unsupported visual model"
        raise NotImplementedError(error_msg)

    image_features = torch.split(image_features, image_num_patches, dim=0)

    #if "LlavaNextForConditionalGeneration" in model.config.architectures:
    print("LlavaNextForConditionalGeneration")
    embed_std = 1 / np.sqrt(model.config.text_config.hidden_size)
    image_newline = torch.Tensor(
        torch.randn(image_features[0].shape[-1], dtype=image_features[0].dtype) * embed_std
    ).to(model.device)
    image_features, _ = model.pack_image_features(
        image_features,
        inputs.image_sizes,
        vision_feature_select_strategy=kwargs.get("vision_feature_select_strategy"),
        image_newline=image_newline,
    )
    # elif (
    #     "Qwen2_5_VLForConditionalGeneration" in model.config.architectures
    #     or "Qwen2VLForConditionalGeneration" in model.config.architectures
    # ):
    #     spatial_merge_size = model.visual.config.spatial_merge_size
    #     pooled_image_features = []
    #     for img_feat, (t, h, w) in zip(image_features, kwargs["image_grid_thw"]):
    #         num_patches, d = img_feat.shape
    #         assert t == 1, "Only single-frame temporal dimension supported"
    #         assert h * w == num_patches, f"H*W != num_patches: {h}*{w} != {num_patches}"

    #         # Reshape to [1, D, H, W]
    #         x = img_feat.view(h, w, d).permute(2, 0, 1).unsqueeze(0)

    #         # Apply avg pooling
    #         x_pooled = F.avg_pool2d(x, kernel_size=spatial_merge_size, stride=spatial_merge_size)

    #         # Reshape back to [num_pooled_patches, D]
    #         pooled = x_pooled.squeeze(0).permute(1, 2, 0).reshape(-1, d)
    #         pooled_image_features.append(pooled)
    #     image_features = pooled_image_features

    if image_features[0].dim() < 3:
        image_features = [feat_i.unsqueeze(0) for feat_i in image_features]
    return image_features.detach().cpu()


def getOriginalVisualToken(model, vision_tower, image_binary):
    
    preprocess_start = time.time()   
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    preprocess_end = time.time()   
    preprocess_time = preprocess_end - preprocess_start
    
    encode_begin = time.time()
    model_device = vision_tower.device

    image_forward_outs = vision_tower.vision_tower(
        images.to(device=model_device, dtype=vision_tower.dtype),
        output_hidden_states=True,
        output_attentions=True
    )
    image_outputs = vision_tower.feature_select(image_forward_outs)
    image_features = image_outputs.to(images.dtype)
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
        
        if image_embeddings:
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