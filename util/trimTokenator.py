
import time
import torch
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from datetime import datetime
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
from util.cdencoder import CLIPVisionTower  # Assuming first file is saved as clipEncoder.py

@torch.no_grad()
def trimTokenatorPruning(model, vision_tower, tokenizer, image_binary, texts, keep_ratio=0.25, 
                         stage1_ratio=0.8):
    """
    TrimTokenator: Two-stage visual token pruning for multimodal models.
    
    Stage 1: Cross-modal alignment - retain tokens with highest mutual information with text
    Stage 2: Greedy intra-modal diversity - maximize expected pairwise distances
    
    Args:
        model: The multimodal model
        vision_tower: Vision encoder
        image_binary: Input image as binary data
        texts: Text prompt/instruction (can be None for uniform relevance)
        keep_ratio: Final ratio of tokens to keep (default: 0.25 = 25%)
        stage1_ratio: Ratio for stage 1 pruning (default: 0.8 = 80%)
    
    Returns:
        Selected visual tokens after two-stage pruning
    """
    
    # Parse image
    preprocess_start = time.time()   
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    preprocess_end = time.time()   
    preprocess_time = preprocess_end - preprocess_start

    encode_begin = time.time()
    model_device = vision_tower.device

    # Extract visual features
    image_forward_outs = vision_tower.vision_tower(
        images.to(device=model_device, dtype=vision_tower.dtype), 
        output_hidden_states=True, output_attentions=True
    )
    image_outputs = vision_tower.feature_select(image_forward_outs)
    image_features = image_outputs.to(images.dtype)
    B, N, C = image_features.shape
    

    # Project visual features
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    encode_end = time.time()
    encode_time = encode_end - encode_begin
    
    # ============================================================================
    # STAGE 1: Cross-Modal Alignment (Mutual Information via L2 norm)
    # ============================================================================
    prune_begin = time.time()
    # Calculate N1 (tokens after stage 1) and N2 (final tokens)
    N1 = int(stage1_ratio * N)
    N2 = int(keep_ratio * N)
    
    if texts is not None:
        # Encode text

        text_inputs = tokenizer(
            text=texts, 
            return_tensors="pt", 
            padding=True
            ).to(device=model_device)
        
        text_embeds = model.get_input_embeddings()(text_inputs.input_ids) # <-- FIX: 4096-dim
        text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)
        M = text_embeds.shape[1]  # Number of text tokens
        
        # Compute alignment scores: α_i = -1/M * Σ ||x_v_i - x_t_j||²
        # Shape: [B, N, M]
        pairwise_distances = torch.cdist(
            image_features, 
            text_embeds, 
            p=2.0
        ) ** 2
        
        # Average L2 distance over text tokens (negative for "higher is better")
        alignment_scores = -pairwise_distances.mean(dim=2)  # [B, N]
        
    else:
        # No text: uniform relevance (all tokens equally important)
        alignment_scores = torch.ones(B, N, device=model_device)
    
    # Select top N1 tokens based on alignment scores
    _, top_indices_stage1 = torch.topk(alignment_scores, k=N1, dim=1)  # [B, N1]
    
    # Gather selected tokens for stage 2
    batch_indices = torch.arange(B, device=model_device).unsqueeze(1).expand(-1, N1)
    X_v1 = image_features[batch_indices, top_indices_stage1]  # [B, N1, C]
    
    # ============================================================================
    # STAGE 2: Greedy Intra-Modal Diversity Maximization (RepMax)
    # ============================================================================
    
    # Normalize tokens for cosine similarity computation
    X_v1_norm = X_v1 / (X_v1.norm(dim=-1, keepdim=True) + 1e-8)
    
    C = torch.matmul(
        X_v1_norm, 
        X_v1_norm.transpose(1, 2)
    )

    # Initialize selection
    selected_indices = []
    remaining_mask = torch.ones(B, N1, dtype=torch.bool, device=model_device)
    
    # Greedy algorithm for N2 iterations
    for t in range(N2):
        if t == 0:
            # Initial: select token with lowest average similarity to all others
            avg_similarity = C.sum(dim=2) / N1  # [B, N1]
            avg_similarity[~remaining_mask] = float('inf')  # Mask already selected
            initial_idx = avg_similarity.argmin(dim=1)  # [B]
            selected_indices.append(initial_idx)
            
            # Update cumulative similarity vector
            batch_range = torch.arange(B, device=model_device)
            sigma = C[batch_range, initial_idx, :]  # [B, N1]
            
            # Mark as selected
            remaining_mask[batch_range, initial_idx] = False
            
        else:
            # Compute average similarity to selected set
            avg_sim_to_selected = sigma / t  # [B, N1]
            avg_sim_to_selected[~remaining_mask] = float('inf')  # Mask selected tokens
            
            # Select token with minimum average similarity
            next_idx = avg_sim_to_selected.argmin(dim=1)  # [B]
            selected_indices.append(next_idx)
            
            # Update cumulative similarity
            batch_range = torch.arange(B, device=model_device)
            sigma = sigma + C[batch_range, next_idx, :]  # [B, N1]
            
            # Mark as selected
            remaining_mask[batch_range, next_idx] = False
    
    # Stack selected indices
    selected_stage2 = torch.stack(selected_indices, dim=1)  # [B, N2]
    
    # Gather final tokens from stage 1 selection
    batch_indices_final = torch.arange(B, device=model_device).unsqueeze(1).expand(-1, N2)
    X_v2 = X_v1[batch_indices_final, selected_stage2]  # [B, N2, C]
    
    # Map back to original indices
    final_original_indices = top_indices_stage1[batch_indices_final, selected_stage2]
    
    # Sort indices to maintain spatial order (optional, improves interpretability)
    final_original_indices_sorted, sort_order = torch.sort(final_original_indices, dim=1)
    X_v2_sorted = X_v2[batch_indices_final, sort_order]
    # Move to CPU and detach
    result = X_v2_sorted.squeeze(0).detach().cpu()
    prune_end = time.time()
    prune_time = prune_end-prune_begin
    return result, preprocess_time, encode_time, prune_time
