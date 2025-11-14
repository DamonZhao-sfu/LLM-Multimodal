
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

@torch.no_grad() 
def getCDPrunedVisualToken(model, vision_tower, image_binary, texts, keep_ratio=0.25):

    kept_token = int(keep_ratio * 576)
    preprocess_start = time.time()   
    image = Image.open(io.BytesIO(image_binary))
    inputs = vision_tower.image_processor(image, return_tensors="pt")
    images = inputs["pixel_values"]
    preprocess_end = time.time()   
    preprocess_time = preprocess_end - preprocess_start

    encode_begin = time.time()
    image_stream = torch.cuda.Stream()
    text_stream = torch.cuda.Stream()
    model_device = vision_tower.device
    with torch.cuda.stream(image_stream):
        image_forward_outs = vision_tower.vision_tower(images.to(device=model_device, dtype=vision_tower.dtype), output_hidden_states=True)
        image_outputs = vision_tower.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)
    
    image_embeds = vision_tower.vision_tower.vision_model.post_layernorm(image_outputs)
    image_embeds = vision_tower.vision_tower.visual_projection(image_embeds.float())
    B, N, C = image_features.shape
    image_features = image_features.to(device=model_device, dtype=torch.float16)
    image_embeds = image_embeds.to(device=model_device)
    model.multi_modal_projector = model.multi_modal_projector.to(model_device)
    image_features = model.multi_modal_projector(image_features)
    encode_end = time.time()
    encode_time = encode_end - encode_begin

    prune_begin = time.time()
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
            text_embeds = vision_tower.text_tower(**text_inputs).text_embeds
            text_embeds = text_embeds.to(device=model_device)

    # Calculate cosine similarity
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
    image_normalized = image_normalized.float()
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))

    # Calculate query relevance
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    relevance = torch.matmul(image_embeds, text_embeds.t())
    relevance = (-relevance).mean(dim=-1)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())

    # Construct kernel matrix
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    
    # Fast MAP inference of conditional DPP
    cis = torch.zeros(int(kept_token), B, N, device=model_device, dtype=torch.float32)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    select_idx = torch.empty((int(kept_token), B), dtype=torch.long, device=model_device)
    
    for i in range(kept_token):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    
    select_idx = torch.sort(select_idx.t()).values
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=model_device)
    index_masks.scatter_(1, select_idx, True)
    
    # Move result to CPU before cleanup
    image_features_selected = image_features[index_masks].unsqueeze(0).detach().cpu()
    prune_end = time.time()
    prune_time = prune_end-prune_begin
    return image_features_selected, preprocess_time, encode_time, prune_time