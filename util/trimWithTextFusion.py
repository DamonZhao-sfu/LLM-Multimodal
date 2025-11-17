import time
import torch
import numpy as np
from PIL import Image
import io

@torch.no_grad()
def trimTokenatorPruning_v2(model, vision_tower, tokenizer, image_binary, texts, 
                            keep_ratio=0.25, stage1_ratio=0.8, 
                            stage2_method='text_guided_diversity'):
    """
    Enhanced TrimTokenator with improved Stage 2 that leverages text embeddings.
    
    Stage 1: Cross-modal alignment - retain tokens with highest mutual information with text
    Stage 2: Text-guided diversity selection - multiple algorithm options
    
    Args:
        model: The multimodal model
        vision_tower: Vision encoder
        tokenizer: Text tokenizer
        image_binary: Input image as binary data
        texts: Text prompt/instruction (can be a string or list of strings)
        keep_ratio: Final ratio of tokens to keep (default: 0.25 = 25%)
        stage1_ratio: Ratio for stage 1 pruning (default: 0.8 = 80%)
        stage2_method: Algorithm for stage 2 selection:
            - 'text_guided_diversity': Balance diversity and text relevance (RECOMMENDED)
            - 'dpp': Determinantal Point Process
            - 'fps': Farthest Point Sampling with text guidance
            - 'original': Original diversity-only algorithm
    
    Returns:
        result: Selected visual tokens [1, N2, C] (batch dim is 1)
        preprocess_time: Image preprocessing time
        encode_time: Vision encoding time
        prune_time: Token pruning time
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
        output_hidden_states=True
    )
    image_outputs = vision_tower.feature_select(image_forward_outs)
    image_features = image_outputs.to(images.dtype)
    B, N, C = image_features.shape
    
    # --- START OF FIX ---
    # This function assumes a single image (B=1).
    # We must ensure the text embeddings also have a batch size of 1.
    if B > 1:
        # This implementation is designed for single-image batching (B=1).
        # We will only process the first image if a batch is provided.
        print(f"Warning: Multiple images detected (batch size {B}). Processing only the first image.")
        image_features = image_features[0:1]
        B = 1
    # --- END OF FIX ---


    # Project visual features
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    encode_end = time.time()
    encode_time = encode_end - encode_begin
    
    # ============================================================================
    # STAGE 1: Cross-Modal Alignment
    # ============================================================================
    prune_begin = time.time()
    N1 = int(stage1_ratio * N)
    N2 = int(keep_ratio * N)
    
    # --- START OF FIX ---
    # Ensure 'texts' is a single string, not a batch (list) of strings.
    # If it's a list, join it into one conceptual prompt.
    if isinstance(texts, list):
        processed_text = " ".join(texts)
    else:
        processed_text = texts
    # --- END OF FIX ---

    # Encode text
    text_inputs = tokenizer(
        text=processed_text,  # Use the processed text
        return_tensors="pt", 
        padding=True
    ).to(device=model_device)
    
    text_embeds = model.get_input_embeddings()(text_inputs.input_ids)
    text_embeds = text_embeds.to(device=model_device, dtype=torch.float16)
    # Now, text_embeds will always have shape [1, M, C]
    
    M = text_embeds.shape[1]
    
    # Compute alignment scores
    # image_features shape is [1, N, C]
    # text_embeds shape is [1, M, C]
    pairwise_distances = torch.cdist(image_features, text_embeds, p=2.0) ** 2
    # pairwise_distances shape will be [1, N, M]
    alignment_scores = -pairwise_distances.mean(dim=2)  # Shape: [1, N]
    
    # Select top N1 tokens
    _, top_indices_stage1 = torch.topk(alignment_scores, k=N1, dim=1) # Shape: [1, N1]
    
    # Gather selected tokens
    # B is 1
    batch_indices = torch.arange(B, device=model_device).unsqueeze(1).expand(-1, N1) # Shape: [1, N1]
    X_v1 = image_features[batch_indices, top_indices_stage1]  # Shape: [1, N1, C]
    
    # ============================================================================
    # STAGE 2: Enhanced Text-Guided Selection
    # ============================================================================
    
    if stage2_method == 'text_guided_diversity':
        selected_stage2 = stage2_text_guided_diversity(
            X_v1, text_embeds, N2, model_device
        )
    elif stage2_method == 'dpp':
        selected_stage2 = stage2_dpp_sampling(
            X_v1, text_embeds, N2, model_device
        )
    elif stage2_method == 'fps':
        selected_stage2 = stage2_fps_text_guided(
            X_v1, text_embeds, N2, model_device
        )
    elif stage2_method == 'original':
        selected_stage2 = stage2_original_diversity(
            X_v1, N2, model_device
        )
    else:
        raise ValueError(f"Unknown stage2_method: {stage2_method}")
    
    # All stage 2 functions now receive inputs with B=1
    # selected_stage2 will have shape [1, N2]
    
    # Gather final tokens
    batch_indices_final = torch.arange(B, device=model_device).unsqueeze(1).expand(-1, N2) # Shape: [1, N2]
    X_v2 = X_v1[batch_indices_final, selected_stage2]  # Shape: [1, N2, C]
    
    # Map back to original indices and sort
    final_original_indices = top_indices_stage1[batch_indices_final, selected_stage2]
    final_original_indices_sorted, sort_order = torch.sort(final_original_indices, dim=1)
    X_v2_sorted = X_v2[batch_indices_final, sort_order] # Shape: [1, N2, C]
    
    #print(f"Stage 2 method: {stage2_method}, Output shape: {X_v2_sorted.shape}")
    
    result = X_v2_sorted.detach().cpu()
    prune_end = time.time()
    prune_time = prune_end - prune_begin
    
    return result, preprocess_time, encode_time, prune_time


# ============================================================================
# Stage 2 Algorithm Implementations (Unchanged)
# ============================================================================

def stage2_text_guided_diversity(X_v1, text_embeds, N2, device):
    """
    Text-Guided Diversity Selection (RECOMMENDED).
    
    Balances:
    - Diversity: Select tokens different from already-selected ones
    - Relevance: Prefer tokens aligned with text query
    
    Score = λ * diversity + (1-λ) * text_relevance
    """
    B, N1, C = X_v1.shape # B will be 1
    lambda_diversity = 0.6  # Tune this: higher = more diversity, lower = more relevance
    
    # Normalize for cosine similarity
    X_v1_norm = X_v1 / (X_v1.norm(dim=-1, keepdim=True) + 1e-8)
    text_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Token-to-token similarity [B, N1, N1]
    C = torch.matmul(X_v1_norm, X_v1_norm.transpose(1, 2))
    
    # Token-to-text relevance [B, N1]
    text_relevance = torch.matmul(X_v1_norm, text_norm.transpose(1, 2)).mean(dim=2)
    
    selected_indices = []
    remaining_mask = torch.ones(B, N1, dtype=torch.bool, device=device)
    
    for t in range(N2):
        if t == 0:
            # Start with most text-relevant token
            text_relevance_masked = text_relevance.clone()
            text_relevance_masked[~remaining_mask] = float('-inf')
            initial_idx = text_relevance_masked.argmax(dim=1)
            selected_indices.append(initial_idx)
            
            batch_range = torch.arange(B, device=device)
            sigma = C[batch_range, initial_idx, :]
            remaining_mask[batch_range, initial_idx] = False
        else:
            # Diversity: negative average similarity to selected
            diversity_score = -(sigma / t)
            
            # Combined score
            combined_score = lambda_diversity * diversity_score + \
                             (1 - lambda_diversity) * text_relevance
            
            combined_score[~remaining_mask] = float('-inf')
            next_idx = combined_score.argmax(dim=1)
            selected_indices.append(next_idx)
            
            batch_range = torch.arange(B, device=device)
            sigma = sigma + C[batch_range, next_idx, :]
            remaining_mask[batch_range, next_idx] = False
    
    return torch.stack(selected_indices, dim=1)


def stage2_dpp_sampling(X_v1, text_embeds, N2, device):
    """
    Determinantal Point Process (DPP) based selection.
    
    Naturally balances diversity and quality through:
    L = diag(q) @ S @ diag(q)
    
    where q = text relevance, S = similarity kernel
    """
    B, N1, C = X_v1.shape
    
    # Store original dtype and convert to float32 for numerical stability
    original_dtype = X_v1.dtype
    X_v1_fp32 = X_v1.float()
    text_embeds_fp32 = text_embeds.float()
    
    X_v1_norm = X_v1_fp32 / (X_v1_fp32.norm(dim=-1, keepdim=True) + 1e-8)
    text_norm = text_embeds_fp32 / (text_embeds_fp32.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Quality scores (text relevance)
    q = torch.matmul(X_v1_norm, text_norm.transpose(1, 2)).mean(dim=2)
    q = torch.clamp(q, min=0) + 0.1  # Ensure positive
    
    # Similarity kernel
    S = torch.matmul(X_v1_norm, X_v1_norm.transpose(1, 2))
    
    # DPP kernel
    q_sqrt = q.sqrt().unsqueeze(-1)
    L = S * (q_sqrt @ q_sqrt.transpose(1, 2))
    
    selected_indices = []
    remaining_mask = torch.ones(B, N1, dtype=torch.bool, device=device)
    batch_range = torch.arange(B, device=device)
    
    # Pre-allocate for efficiency
    L_selected_inv = None
    
    for t in range(N2):
        if t == 0:
            scores = q.clone()
            scores[~remaining_mask] = float('-inf')
            initial_idx = scores.argmax(dim=1)
            selected_indices.append(initial_idx)
            remaining_mask[batch_range, initial_idx] = False
            
            # Initialize inverse (1x1 matrix)
            L_selected_inv = 1.0 / L[batch_range, initial_idx, initial_idx].unsqueeze(-1).unsqueeze(-1)
        else:
            # Efficiently calculate marginal gain
            selected_so_far = torch.stack(selected_indices, dim=1)  # [B, t]
            
            # Get L[i, selected_indices] for all i
            # Shape: [B, N1, t]
            L_i_selected = L[batch_range.unsqueeze(1).unsqueeze(2), 
                            torch.arange(N1, device=device).unsqueeze(0).unsqueeze(-1), 
                            selected_so_far.unsqueeze(1)]
            
            # Compute score: L[i,i] - L[i,S] @ L[S,S]^{-1} @ L[S,i]
            L_inv_dot_L_i_selected_t = torch.matmul(L_selected_inv, L_i_selected.transpose(1, 2))
            term = torch.matmul(L_i_selected, L_inv_dot_L_i_selected_t)
            
            diag_L = torch.diagonal(L, dim1=1, dim2=2)  # [B, N1]
            diag_term = torch.diagonal(term, dim1=1, dim2=2)  # [B, N1]
            
            scores = diag_L - diag_term
            scores[~remaining_mask] = float('-inf')
            
            next_idx = scores.argmax(dim=1)
            selected_indices.append(next_idx)
            remaining_mask[batch_range, next_idx] = False
            
            # Update L_selected_inv using matrix inversion
            # For numerical stability, recompute from scratch
            selected_now = torch.stack(selected_indices, dim=1)  # [B, t+1]
            L_selected = L[batch_range.unsqueeze(1), selected_now.unsqueeze(2), 
                          selected_now.unsqueeze(1)]
            
            # Add small jitter for numerical stability
            L_selected = L_selected + torch.eye(t + 1, device=device, dtype=torch.float32).unsqueeze(0) * 1e-6
            
            try:
                L_selected_inv = torch.linalg.inv(L_selected)
            except:
                # Fallback with more jitter
                L_selected_inv = torch.linalg.inv(
                    L_selected + torch.eye(t + 1, device=device, dtype=torch.float32).unsqueeze(0) * 1e-4
                )
    
    return torch.stack(selected_indices, dim=1)


def stage2_fps_text_guided(X_v1, text_embeds, N2, device):
    """
    Farthest Point Sampling with Text Guidance.
    
    - Start with most text-relevant token
    - Iteratively select farthest token (weighted by text relevance)
    """
    B, N1, C = X_v1.shape # B will be 1
    
    X_v1_norm = X_v1 / (X_v1.norm(dim=-1, keepdim=True) + 1e-8)
    text_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Text relevance weights
    relevance_weights = torch.matmul(X_v1_norm, text_norm.transpose(1, 2)).mean(dim=2)
    relevance_weights = torch.sigmoid(relevance_weights * 2) # Shape [B, N1]
    
    # Distance matrix (using cosine distance)
    C = torch.matmul(X_v1_norm, X_v1_norm.transpose(1, 2))
    D = 1 - C # Shape [B, N1, N1]
    
    selected_indices = []
    # min_dist_to_selected[b, i] = min distance from token i to any selected token
    min_dist_to_selected = torch.full((B, N1), float('inf'), device=device)
    batch_range = torch.arange(B, device=device)

    for t in range(N2):
        if t == 0:
            scores = relevance_weights.clone()
            initial_idx = scores.argmax(dim=1) # Shape [B]
            selected_indices.append(initial_idx)
            
            min_dist_to_selected = D[batch_range, initial_idx, :]
        else:
            # Score = current_min_distance * text_relevance
            weighted_dist = min_dist_to_selected * relevance_weights
            
            # Mask out already selected tokens
            selected_so_far = torch.stack(selected_indices, dim=1) # Shape [B, t]
            weighted_dist[batch_range.unsqueeze(1), selected_so_far] = float('-inf')
            
            next_idx = weighted_dist.argmax(dim=1) # Shape [B]
            selected_indices.append(next_idx)
            
            # Update min distances
            new_dist = D[batch_range, next_idx, :]
            min_dist_to_selected = torch.minimum(min_dist_to_selected, new_dist)
    
    return torch.stack(selected_indices, dim=1)


def stage2_original_diversity(X_v1, N2, device):
    """
    Original diversity-only algorithm (for comparison).
    """
    B, N1, C = X_v1.shape # B will be 1
    
    X_v1_norm = X_v1 / (X_v1.norm(dim=-1, keepdim=True) + 1e-8)
    C = torch.matmul(X_v1_norm, X_v1_norm.transpose(1, 2))
    
    selected_indices = []
    remaining_mask = torch.ones(B, N1, dtype=torch.bool, device=device)
    batch_range = torch.arange(B, device=device)

    for t in range(N2):
        if t == 0:
            # Start with the token that is least similar to all others
            avg_similarity = C.sum(dim=2) / N1
            avg_similarity[~remaining_mask] = float('inf')
            initial_idx = avg_similarity.argmin(dim=1)
            selected_indices.append(initial_idx)
            
            sigma = C[batch_range, initial_idx, :]
            remaining_mask[batch_range, initial_idx] = False
        else:
            # Pick the token least similar to the selected set
            avg_sim_to_selected = sigma / t
            avg_sim_to_selected[~remaining_mask] = float('inf')
            
            next_idx = avg_sim_to_selected.argmin(dim=1)
            selected_indices.append(next_idx)
            
            sigma = sigma + C[batch_range, next_idx, :]
            remaining_mask[batch_range, next_idx] = False
    
    return torch.stack(selected_indices, dim=1)