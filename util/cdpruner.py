
import time
import torch
import numpy as np
from PIL import Image
import requests
import json
import io
import base64
import os
import torch
import torch.nn as nn
from datetime import datetime
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
from util.cdencoder import CLIPVisionTower  # Assuming first file is saved as clipEncoder.py

def getCDPrunedVisualToken(vision_tower, image_binary, texts, keep_ratio=0.25):

    kept_token = int(keep_ratio * 576)

    image = Image.open(io.BytesIO(image_binary))       
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    image_stream = torch.cuda.Stream()
    text_stream = torch.cuda.Stream()
    
    model_device = vision_tower.device

    with torch.cuda.stream(image_stream):
        image_forward_outs = vision_tower.vision_tower(images.to(device=model_device, dtype=vision_tower.dtype), output_hidden_states=True)
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
    
    if texts is not None:
        with torch.cuda.stream(text_stream):
            text_inputs = vision_tower.text_tokenizer(text=texts, return_tensors="pt")
            text_segment = (text_inputs.input_ids.shape[1] - 1) // vision_tower.max_position_embeddings + 1
            text_padding = vision_tower.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
            text_inputs = {
                k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
                                dim=1).reshape(-1, vision_tower.max_position_embeddings).to(device=vision_tower.device)
                for k, v in text_inputs.items()
            }
            # Keep text_embeds on GPU
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
    
    torch.cuda.synchronize()

    if texts is not None:
        image_embeds = vision_tower.vision_tower.vision_model.post_layernorm(image_outputs)
        image_embeds = vision_tower.vision_tower.visual_projection(image_embeds.float())

    B, N, C = image_features.shape
    
    # Move image_features to the same device as the model BEFORE applying projection
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    
    # Ensure all tensors are on the same device as the model
    image_embeds = image_embeds.to(device=model_device)
    text_embeds = text_embeds.to(device=model_device)
    
    mm_projector = nn.Linear(1024, 4096).to(device=model_device, dtype=torch.float16)

    # Apply projection - now both tensors are on the same device
    image_features = mm_projector(image_features)

    # [CDPruner] Calculate cosine similarity - ALL ON SAME DEVICE
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, D)
    image_normalized = image_normalized.float()  # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)

    # [CDPruner] Calculate query relevance - ALL ON SAME DEVICE
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (M, C)
    relevance = torch.matmul(image_embeds, text_embeds.t())  # (B, N, M)
    relevance = (-relevance).mean(dim=-1)  # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())  # (B, N)

    # [CDPruner] Construct kernel matrix - ALL ON SAME DEVICE
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)
    
    # [CDPruner] Fast MAP inference of conditional DPP - ALL ON SAME DEVICE
    cis = torch.zeros(int(kept_token), B, N, device=model_device, dtype=torch.float32)    #cis = torch.zeros((kept_token, B, N), device=model_device)  # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()  # (B, N)
    select_idx = torch.empty((int(kept_token), B), dtype=torch.long, device=model_device)  # (T, B)
    
    for i in range(kept_token):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    
    select_idx = torch.sort(select_idx.t()).values  # (B, T)
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=model_device)
    index_masks.scatter_(1, select_idx, True)
    
    # Apply mask and move final result to CPU only at the very end
    image_features_selected = image_features[index_masks].unsqueeze(0).detach().cpu()
    
    return image_features_selected