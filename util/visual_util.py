# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This logic is largely copied from the https://github.com/Theia-4869/CDPruner/tree/main
import numpy as np
import torch
import torch.nn.functional as F


def get_visual_similarity(image_features):
    """
    Compute the cosine similarity matrix among image features.
    """
    image_features = image_features.float()  # (B, N, D)
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)
    return similarity


def get_relevance_score(image_embeds, text_embeds):
    """
    Compute the relevance score between image and text embeddings.
    """
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (M, C)

    relevance = torch.matmul(image_embeds, text_embeds.t())  # (B, N, M)
    relevance = (-relevance).mean(dim=-1)  # (B, N)
    relevance = (relevance - relevance.min(dim=1, keepdim=True)[0]) / (
        relevance.max(dim=1, keepdim=True)[0] - relevance.min(dim=1, keepdim=True)[0] + 1e-6
    )
    return relevance


def build_conditional_kernel_matrix(relevance, similarity, theta=0.5):
    """
    Build the conditional DPP kernel matrix based on relevance and visual similarity.
    """
    if theta != 1:
        alpha = theta / (2 * (1 - theta))
        relevance = torch.exp(alpha * relevance)  # (B, N)

    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)
    return kernel


def conditional_dpp_map(kernel, num_keep_tokens):
    """
    Perform conditional DPP MAP inference to select a subset of tokens.
    """
    device = kernel.device

    # kernel diagonal (di2s[b, i] = kernel[b, i, i] = relevance[b, i] ** 2 * (L[i,i]=1))
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()

    # orthogonal directions corresponding to selected tokens (L~=CC^T)
    B, N = di2s.shape
    cis = torch.zeros((num_keep_tokens, B, N), device=device)  # (num_keep_tokens, B, N)

    keep_indices = torch.empty((num_keep_tokens, B), dtype=torch.long, device=device)
    batch_idx = torch.arange(B)
    for i in range(num_keep_tokens):
        j = torch.argmax(di2s, dim=-1)  # Select the index with highest remaining score
        keep_indices[i] = j

        # compute the orthogonalized row vector for token j
        if i == 0:
            eis = kernel[batch_idx, j] / torch.sqrt(kernel[batch_idx, j, j].unsqueeze(1) + 1e-5)  # (B, N)
        else:
            proj = torch.einsum("tb,tbn->bn", cis[:i, batch_idx, j], cis[:i])  # (B, N)
            eis = (kernel[batch_idx, j] - proj) / torch.sqrt(di2s[batch_idx, j].unsqueeze(-1) + 1e-5)

        cis[i, :, :] = eis
        di2s -= eis**2
        di2s[batch_idx, j] = -float("inf")

    keep_indices = torch.sort(keep_indices.t()).values
    return keep_indices


def get_model_kwargs(model, inputs):
    """
    Get the model keyword arguments from the model and inputs.
    """
    kwargs = {}
    if hasattr(model.config, "vision_feature_select_strategy"):
        kwargs["vision_feature_select_strategy"] = model.config.vision_feature_select_strategy
    if hasattr(model.config, "vision_feature_layer"):
        kwargs["vision_feature_layer"] = model.config.vision_feature_layer
    if hasattr(inputs, "image_sizes"):
        kwargs["image_sizes"] = inputs.image_sizes
    if hasattr(inputs, "image_grid_thw"):
        kwargs["image_grid_thw"] = inputs.image_grid_thw
    return kwargs


def get_image_features(model, inputs, **kwargs):
    """
    Extract image features from the model.
    """
    pixel_values = inputs.pixel_values
    image_num_patches = None
    if "LlavaNextForConditionalGeneration" in model.config.architectures and pixel_values.dim() == 5:
        from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

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
        image_features = model.vision_tower(pixel_values=pixel_values, output_hidden_states=True)  # .last_hidden_state
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

    if "LlavaNextForConditionalGeneration" in model.config.architectures:
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
    elif (
        "Qwen2_5_VLForConditionalGeneration" in model.config.architectures
        or "Qwen2VLForConditionalGeneration" in model.config.architectures
    ):
        spatial_merge_size = model.visual.config.spatial_merge_size
        pooled_image_features = []
        for img_feat, (t, h, w) in zip(image_features, kwargs["image_grid_thw"]):
            num_patches, d = img_feat.shape
            assert t == 1, "Only single-frame temporal dimension supported"
            assert h * w == num_patches, f"H*W != num_patches: {h}*{w} != {num_patches}"

            # Reshape to [1, D, H, W]
            x = img_feat.view(h, w, d).permute(2, 0, 1).unsqueeze(0)

            # Apply avg pooling
            x_pooled = F.avg_pool2d(x, kernel_size=spatial_merge_size, stride=spatial_merge_size)

            # Reshape back to [num_pooled_patches, D]
            pooled = x_pooled.squeeze(0).permute(1, 2, 0).reshape(-1, d)
            pooled_image_features.append(pooled)
        image_features = pooled_image_features

    if image_features[0].dim() < 3:
        image_features = [feat_i.unsqueeze(0) for feat_i in image_features]
    return image_features


def get_cdpruner_mask(image_embeds, image_features, text_embeds, special_image_mask, num_keep_tokens, theta):
    """
    Generate a mask to retain image tokens based on fast MAP inference using Conditional DPP for token selection.
    """
    keep_indices = []
    offset = 0
    # Compute keep_indices for each image embedding
    for emb_i, feat_i in zip(image_embeds, image_features):
        rel_i = get_relevance_score(emb_i.unsqueeze(0), text_embeds)
        sim_i = get_visual_similarity(feat_i)
        kernel_i = build_conditional_kernel_matrix(rel_i, sim_i, theta)
        keep_i = conditional_dpp_map(kernel_i, num_keep_tokens)[0] + offset
        keep_indices.append(keep_i)
        offset += emb_i.shape[0]

    keep_indices = torch.cat(keep_indices, dim=0)

    # Get the positions of the selected image tokens
    image_token_positions = torch.nonzero(special_image_mask[0], as_tuple=False).squeeze(1)
    kept_positions = image_token_positions[keep_indices]

    # Build mask to keep: original text + selected image tokens
    kept_mask = ~special_image_mask
    kept_mask[0, kept_positions] = True

    return kept_mask

@torch.no_grad()
def get_inputs_embeds(model, inputs, num_keep_tokens=None, theta=0.5):
    # --- FIX START: Determine device and move inputs ---
    # 1. Get the device the model is currently on
    device = model.get_input_embeddings().weight.device
    
    # 2. Move the input_ids to that device BEFORE using them
    input_ids = inputs.input_ids.to(device)
    
    # 3. Also move pixel_values (you will need this a few lines down)
    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
        inputs.pixel_values = inputs.pixel_values.to(device)
    # ---------------------------------------------------

    kwargs = get_model_kwargs(model, inputs)
    
    # Use the moved input_ids here
    inputs_embeds = model.get_input_embeddings()(input_ids) 
    
    B, _, emb_dim = inputs_embeds.shape
    assert B == 1

    # Ensure image_token_id is on the correct device
    image_token_id = torch.tensor(model.config.image_token_id, dtype=torch.long, device=device)
    special_image_emb = model.get_input_embeddings()(image_token_id)
    special_image_mask = (inputs_embeds == special_image_emb).all(-1)

    image_embeds = model.get_image_features(
        pixel_values=inputs.pixel_values, # This is now safe because we moved it above
        **kwargs,
    )

    flat_image_embeds = torch.cat(image_embeds, dim=0).to(device, inputs_embeds.dtype)
    exp_special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds_with_images = inputs_embeds.masked_scatter(exp_special_image_mask, flat_image_embeds)

    if num_keep_tokens is None:
        return inputs_embeds_with_images

    text_embeds = inputs_embeds[~special_image_mask].view(-1, emb_dim)
    image_features = [feat.to(device) for feat in get_image_features(model, inputs, **kwargs)]
    kept_mask = get_cdpruner_mask(image_embeds, image_features, text_embeds, special_image_mask, num_keep_tokens, theta)

    kept_mask = kept_mask.unsqueeze(-1).expand_as(inputs_embeds)
    pruned_embeds = inputs_embeds_with_images[kept_mask].view(B, -1, emb_dim)
    
    return pruned_embeds.to(device)


def get_trimtokenator_mask(image_embeds, text_embeds, special_image_mask, keep_ratio=0.25, stage1_ratio=0.8):
    """
    Generate a mask to retain image tokens based on TrimTokenator two-stage pruning.
    
    Stage 1: Cross-modal alignment - retain tokens with highest mutual information with text
    Stage 2: Greedy intra-modal diversity - maximize expected pairwise distances
    """
    keep_indices = []
    offset = 0
    
    # Process each image embedding
    for image_emb in image_embeds:
        device = image_emb.device
        
        # Handle different input shapes
        if image_emb.dim() == 2:
            # Shape is [N, C], add batch dimension
            image_emb = image_emb.unsqueeze(0)  # [1, N, C]
        
        B, N, C = image_emb.shape
        
        # Calculate N1 (tokens after stage 1) and N2 (final tokens)
        N1 = int(stage1_ratio * N)
        N2 = int(keep_ratio * N)
        
        # Safety check: ensure N1 and N2 are valid
        N1 = max(1, min(N1, N))
        N2 = max(1, min(N2, N1))
        
        # ============================================================================
        # STAGE 1: Cross-Modal Alignment (Mutual Information via L2 norm)
        # ============================================================================
        if text_embeds is not None and text_embeds.numel() > 0:
            M = text_embeds.shape[0]  # Number of text tokens
            
            # Compute pairwise distances
            pairwise_distances = torch.cdist(
                image_emb,
                text_embeds.unsqueeze(0),
                p=2.0
            ) ** 2  # [B, N, M]
            
            # Average L2 distance over text tokens (negative for "higher is better")
            alignment_scores = -pairwise_distances.mean(dim=2)  # [B, N]
        else:
            # No text: uniform relevance (all tokens equally important)
            alignment_scores = torch.ones(B, N, device=device)
        
        # Select top N1 tokens based on alignment scores
        _, top_indices_stage1 = torch.topk(alignment_scores, k=N1, dim=1)  # [B, N1]
        
        # Gather selected tokens for stage 2
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N1)
        X_v1 = image_emb[batch_indices, top_indices_stage1]  # [B, N1, C]
        
        # ============================================================================
        # STAGE 2: Greedy Intra-Modal Diversity Maximization (RepMax)
        # ============================================================================
        
        # Normalize tokens for cosine similarity computation
        X_v1_norm = X_v1 / (X_v1.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(X_v1_norm, X_v1_norm.transpose(1, 2))  # [B, N1, N1]
        
        # Initialize selection
        selected_indices = []
        remaining_mask = torch.ones(B, N1, dtype=torch.bool, device=device)
        
        # Greedy algorithm for N2 iterations
        for t in range(N2):
            if t == 0:
                # Initial: select token with lowest average similarity to all others
                avg_similarity = similarity_matrix.sum(dim=2) / N1  # [B, N1]
                avg_similarity[~remaining_mask] = float('inf')  # Mask already selected
                initial_idx = avg_similarity.argmin(dim=1)  # [B]
                selected_indices.append(initial_idx)
                
                # Update cumulative similarity vector
                batch_range = torch.arange(B, device=device)
                sigma = similarity_matrix[batch_range, initial_idx, :]  # [B, N1]
                
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
                batch_range = torch.arange(B, device=device)
                sigma = sigma + similarity_matrix[batch_range, next_idx, :]  # [B, N1]
                
                # Mark as selected
                remaining_mask[batch_range, next_idx] = False
        
        # Stack selected indices
        selected_stage2 = torch.stack(selected_indices, dim=1)  # [B, N2]
        
        # Map back to original indices
        batch_indices_final = torch.arange(B, device=device).unsqueeze(1).expand(-1, N2)
        final_original_indices = top_indices_stage1[batch_indices_final, selected_stage2]
        
        # Sort indices to maintain spatial order
        final_original_indices_sorted, _ = torch.sort(final_original_indices, dim=1)
        
        # Add offset and collect indices
        keep_indices.append(final_original_indices_sorted.squeeze(0) + offset)
        offset += N
    
    # Concatenate all keep indices
    keep_indices = torch.cat(keep_indices, dim=0)
    
    # Get the positions of the selected image tokens
    image_token_positions = torch.nonzero(special_image_mask[0], as_tuple=False).squeeze(1)
    kept_positions = image_token_positions[keep_indices]
    
    # Build mask to keep: original text + selected image tokens
    kept_mask = ~special_image_mask
    kept_mask[0, kept_positions] = True
    
    return kept_mask


@torch.no_grad()
def get_inputs_embeds_trim(model, inputs, keep_ratio=0.25, stage1_ratio=0.8):
    """
    Get input embeddings with TrimTokenator pruning applied to image tokens.
    
    Args:
        model: The multimodal model
        inputs: Input data containing input_ids and pixel_values
        keep_ratio: Final ratio of image tokens to keep (default: 0.25 = 25%)
        stage1_ratio: Ratio for stage 1 pruning (default: 0.8 = 80%)
    
    Returns:
        Pruned input embeddings as 2D tensor [seq_len, emb_dim]
    """
    # Determine device and move inputs
    device = model.get_input_embeddings().weight.device
    
    # Move the input_ids to that device
    input_ids = inputs.input_ids.to(device)
    
    # Move pixel_values
    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
        inputs.pixel_values = inputs.pixel_values.to(device)
    
    # Get model-specific kwargs
    kwargs = {}
    if hasattr(inputs, 'image_sizes'):
        kwargs['image_sizes'] = inputs.image_sizes
    if hasattr(inputs, 'image_grid_thw'):
        kwargs['image_grid_thw'] = inputs.image_grid_thw
    
    # Get input embeddings
    inputs_embeds = model.get_input_embeddings()(input_ids)
    
    B, _, emb_dim = inputs_embeds.shape
    assert B == 1, "Batch size must be 1"
    
    # Identify image token positions
    image_token_id = torch.tensor(model.config.image_token_id, dtype=torch.long, device=device)
    special_image_emb = model.get_input_embeddings()(image_token_id)
    special_image_mask = (inputs_embeds == special_image_emb).all(-1)
    
    # Get image embeddings
    image_embeds = model.get_image_features(
        pixel_values=inputs.pixel_values,
        **kwargs,
    )
    
    # Ensure image_embeds are 3D [1, N, C] or list of [N, C]
    processed_image_embeds = []
    for img_emb in image_embeds:
        if img_emb.dim() == 2:
            # [N, C] -> add batch dimension
            processed_image_embeds.append(img_emb.unsqueeze(0))
        elif img_emb.dim() == 3:
            processed_image_embeds.append(img_emb)
        else:
            raise ValueError(f"Unexpected image embedding shape: {img_emb.shape}")
    
    # Flatten for insertion: concatenate all image tokens
    flat_image_embeds = torch.cat([emb.reshape(-1, emb_dim) for emb in processed_image_embeds], dim=0)
    flat_image_embeds = flat_image_embeds.to(device, inputs_embeds.dtype)
    
    exp_special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds_with_images = inputs_embeds.masked_scatter(exp_special_image_mask, flat_image_embeds)
    
    # If no pruning requested, return as 2D tensor
    if keep_ratio is None or keep_ratio >= 1.0:
        return inputs_embeds_with_images.squeeze(0)  # [seq_len, emb_dim]
    
    # Extract text embeddings for alignment computation
    text_embeds = inputs_embeds[~special_image_mask].view(-1, emb_dim)
    
    # Apply TrimTokenator pruning (pass the processed embeddings)
    kept_mask = get_trimtokenator_mask(
        processed_image_embeds, 
        text_embeds, 
        special_image_mask, 
        keep_ratio=keep_ratio,
        stage1_ratio=stage1_ratio
    )
    
    # Apply mask to keep selected tokens
    kept_mask = kept_mask.unsqueeze(-1).expand_as(inputs_embeds)
    pruned_embeds = inputs_embeds_with_images[kept_mask].view(B, -1, emb_dim)
    
    # Return as 2D tensor [seq_len, emb_dim]
    return pruned_embeds.to(device)