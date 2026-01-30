"""
Feature Extraction Join (FE-Join)

A generic multimodal join operator that:
1. Left Table: Contains image column
2. Right Table: Contains text column
3. Join: Groups similar images, extracts descriptions, performs semantic text-text join

Key Optimization:
- Groups similar images to reduce VLM calls (100 images -> ~20 groups)
- Each group shares a representative description
- Final join is text-text (cheaper than repeated VLM calls)

Usage:
    -- SQL Interface
    SELECT * FROM left_table l
    JOIN right_table r
    ON FEATURE_JOIN(l.image, r.text, 'Extract product features and match with description')

    -- Or with extracted features
    SELECT l.*, r.*, EXTRACT_FEATURES(l.image) as features
    FROM left_table l
    JOIN right_table r
    ON SEMANTIC_MATCH(EXTRACT_FEATURES(l.image), r.text) > 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import json
import time
import re
import requests
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FeatureExtractionConfig:
    """Configuration for Feature Extraction Join."""
    # Embedding model
    embedding_model: str = "openai/clip-vit-large-patch14-336"

    # Image grouping/clustering
    enable_grouping: bool = True
    grouping_method: str = "kmeans"  # "kmeans", "agglomerative", "threshold"
    num_groups: int = 20  # Target number of groups (for kmeans)
    similarity_threshold: float = 0.85  # For threshold-based grouping
    min_group_size: int = 1
    max_group_size: int = 50

    # Feature extraction (VLM)
    vlm_model: str = "llava-hf/llava-1.5-7b-hf"
    vlm_api_url: str = "http://localhost:8000/v1"
    extraction_prompt: str = "Describe this image in detail, focusing on key visual features, objects, colors, and any text visible."

    # Text-text join
    text_similarity_threshold: float = 0.5
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Batch processing
    batch_size: int = 32

    # Cost tracking
    track_costs: bool = True


@dataclass
class JoinResult:
    """Result of a Feature Extraction Join."""
    left_id: Any
    right_id: Any
    left_image_path: Optional[str] = None
    extracted_description: str = ""
    right_text: str = ""
    similarity_score: float = 0.0
    group_id: int = -1
    matched: bool = False


@dataclass
class JoinStatistics:
    """Statistics for Feature Extraction Join."""
    total_left_rows: int = 0
    total_right_rows: int = 0
    num_image_groups: int = 0
    vlm_calls: int = 0
    text_comparisons: int = 0
    matched_pairs: int = 0

    embedding_time: float = 0.0
    grouping_time: float = 0.0
    extraction_time: float = 0.0
    join_time: float = 0.0
    total_time: float = 0.0

    # Cost estimates
    vlm_cost_saved: float = 0.0  # Percentage saved vs naive approach

    def compute_savings(self):
        """Compute cost savings vs naive VLM-per-image approach."""
        naive_vlm_calls = self.total_left_rows
        actual_vlm_calls = self.vlm_calls
        if naive_vlm_calls > 0:
            self.vlm_cost_saved = 1.0 - (actual_vlm_calls / naive_vlm_calls)


# ============================================================================
# Image Embedding and Grouping
# ============================================================================

class ImageEmbedder:
    """
    Embeds images using CLIP for similarity computation and grouping.
    """

    def __init__(self, config: FeatureExtractionConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.vision_tower = None
        self._initialized = False

    def load_model(self):
        """Load CLIP model for embeddings."""
        if self._initialized:
            return

        try:
            from util.cdencoder import CLIPVisionTower

            class MockArgs:
                def __init__(self):
                    self.mm_vision_select_layer = -2
                    self.mm_vision_select_feature = 'patch'

            mock_args = MockArgs()
            self.vision_tower = CLIPVisionTower(
                self.config.embedding_model,
                mock_args,
                delay_load=False
            )
            self.vision_tower = self.vision_tower.to(self.device)
            self._initialized = True

        except Exception as e:
            print(f"Failed to load vision tower: {e}")
            print("Falling back to transformers CLIP...")

            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            self._initialized = True
            self.vision_tower = None

    def embed_image(self, image_data: Union[bytes, str, Image.Image]) -> torch.Tensor:
        """Embed a single image."""
        self.load_model()

        # Load image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, Image.Image):
            image = image_data.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        with torch.no_grad():
            if self.vision_tower is not None:
                # Use custom vision tower
                inputs = self.vision_tower.image_processor(image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=self.vision_tower.dtype)

                outputs = self.vision_tower.vision_tower(
                    pixel_values,
                    output_hidden_states=True
                )
                # Use CLS token as image embedding
                embedding = outputs.last_hidden_state[:, 0, :]
                embedding = F.normalize(embedding, p=2, dim=-1)
            else:
                # Use transformers CLIP
                inputs = self.clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.clip_model.get_image_features(**inputs)
                embedding = F.normalize(outputs, p=2, dim=-1)

        return embedding.cpu()

    def embed_images_batch(
        self,
        image_data_list: List[Union[bytes, str]]
    ) -> torch.Tensor:
        """Embed multiple images in batch."""
        self.load_model()

        images = []
        valid_indices = []

        for i, image_data in enumerate(image_data_list):
            try:
                if isinstance(image_data, bytes):
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                elif isinstance(image_data, str):
                    image = Image.open(image_data).convert('RGB')
                elif isinstance(image_data, Image.Image):
                    image = image_data.convert('RGB')
                else:
                    continue
                images.append(image)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading image {i}: {e}")
                continue

        if not images:
            return torch.tensor([])

        # Process in batches
        all_embeddings = []
        batch_size = self.config.batch_size

        with torch.no_grad():
            for batch_start in range(0, len(images), batch_size):
                batch_images = images[batch_start:batch_start + batch_size]

                if self.vision_tower is not None:
                    inputs = self.vision_tower.image_processor(batch_images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.device, dtype=self.vision_tower.dtype)

                    outputs = self.vision_tower.vision_tower(
                        pixel_values,
                        output_hidden_states=True
                    )
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                else:
                    inputs = self.clip_processor(images=batch_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    batch_embeddings = self.clip_model.get_image_features(**inputs)
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)

                all_embeddings.append(batch_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0), valid_indices


class ImageGrouper:
    """
    Groups similar images together to reduce VLM calls.

    Instead of calling VLM on each image, we:
    1. Embed all images
    2. Cluster/group similar images
    3. Call VLM once per group (using representative image)
    4. Assign group description to all images in group
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def group_images(
        self,
        embeddings: torch.Tensor,
        image_ids: List[Any]
    ) -> Dict[int, List[int]]:
        """
        Group images based on embedding similarity.

        Returns:
            Dictionary mapping group_id -> list of image indices
        """
        n_images = embeddings.shape[0]

        if n_images == 0:
            return {}

        if n_images == 1:
            return {0: [0]}

        embeddings_np = embeddings.numpy()

        if self.config.grouping_method == "kmeans":
            return self._group_kmeans(embeddings_np, n_images)
        elif self.config.grouping_method == "agglomerative":
            return self._group_agglomerative(embeddings_np, n_images)
        elif self.config.grouping_method == "threshold":
            return self._group_threshold(embeddings_np, n_images)
        else:
            # Default: each image is its own group
            return {i: [i] for i in range(n_images)}

    def _group_kmeans(
        self,
        embeddings: np.ndarray,
        n_images: int
    ) -> Dict[int, List[int]]:
        """Group using K-means clustering."""
        # Determine number of clusters
        n_clusters = min(self.config.num_groups, n_images)
        n_clusters = max(1, n_clusters)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[int(label)].append(i)

        return dict(groups)

    def _group_agglomerative(
        self,
        embeddings: np.ndarray,
        n_images: int
    ) -> Dict[int, List[int]]:
        """Group using agglomerative clustering."""
        n_clusters = min(self.config.num_groups, n_images)
        n_clusters = max(1, n_clusters)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[int(label)].append(i)

        return dict(groups)

    def _group_threshold(
        self,
        embeddings: np.ndarray,
        n_images: int
    ) -> Dict[int, List[int]]:
        """Group using similarity threshold."""
        similarity_matrix = cosine_similarity(embeddings)

        groups = {}
        assigned = set()
        group_id = 0

        for i in range(n_images):
            if i in assigned:
                continue

            # Find all images similar to this one
            similar = np.where(similarity_matrix[i] >= self.config.similarity_threshold)[0]
            group_members = [j for j in similar if j not in assigned]

            # Limit group size
            group_members = group_members[:self.config.max_group_size]

            if group_members:
                groups[group_id] = group_members
                assigned.update(group_members)
                group_id += 1

        return groups

    def get_representative_image(
        self,
        group_indices: List[int],
        embeddings: torch.Tensor
    ) -> int:
        """
        Get the most representative image in a group (closest to centroid).
        """
        if len(group_indices) == 1:
            return group_indices[0]

        group_embeddings = embeddings[group_indices]
        centroid = group_embeddings.mean(dim=0, keepdim=True)

        # Find image closest to centroid
        similarities = F.cosine_similarity(group_embeddings, centroid)
        best_idx = similarities.argmax().item()

        return group_indices[best_idx]


# ============================================================================
# Feature Extraction (VLM)
# ============================================================================

class FeatureExtractor:
    """
    Extracts text descriptions from images using VLM.

    Optimized to work on group representatives rather than all images.
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def extract_description(
        self,
        image_data: Union[bytes, str],
        custom_prompt: Optional[str] = None
    ) -> str:
        """Extract description from a single image using VLM."""
        prompt = custom_prompt or self.config.extraction_prompt

        # Encode image to base64
        if isinstance(image_data, str):
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = image_data

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Build request
        payload = {
            "model": self.config.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                f"{self.config.vlm_api_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"VLM API error: {response.status_code}")
                return ""

        except Exception as e:
            print(f"VLM extraction error: {e}")
            return ""

    def extract_descriptions_batch(
        self,
        image_data_list: List[Union[bytes, str]],
        custom_prompt: Optional[str] = None
    ) -> List[str]:
        """Extract descriptions for multiple images."""
        descriptions = []
        for image_data in image_data_list:
            desc = self.extract_description(image_data, custom_prompt)
            descriptions.append(desc)
        return descriptions


# ============================================================================
# Text-Text Semantic Join
# ============================================================================

class TextMatcher:
    """
    Performs semantic matching between extracted descriptions and right table text.
    """

    def __init__(self, config: FeatureExtractionConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def load_model(self):
        """Load text embedding model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.text_embedding_model)
            self.model = self.model.to(self.device)
            self._initialized = True
        except ImportError:
            print("sentence-transformers not available, using CLIP text encoder")
            self._use_clip_text = True
            self._initialized = True

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed texts into vector space."""
        self.load_model()

        if hasattr(self, '_use_clip_text') and self._use_clip_text:
            # Use CLIP text encoder as fallback
            from transformers import CLIPProcessor, CLIPModel
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            with torch.no_grad():
                inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = clip.get_text_features(**inputs)
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            return embeddings.cpu()
        else:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings.cpu()

    def compute_similarity_matrix(
        self,
        left_texts: List[str],
        right_texts: List[str]
    ) -> np.ndarray:
        """Compute pairwise similarity between left and right texts."""
        left_embeddings = self.embed_texts(left_texts)
        right_embeddings = self.embed_texts(right_texts)

        # Compute cosine similarity matrix
        similarity = torch.mm(left_embeddings, right_embeddings.t())
        return similarity.numpy()

    def find_matches(
        self,
        left_texts: List[str],
        right_texts: List[str],
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find matching pairs between left and right texts.

        Returns:
            List of (left_idx, right_idx, similarity_score) tuples
        """
        threshold = threshold or self.config.text_similarity_threshold

        similarity_matrix = self.compute_similarity_matrix(left_texts, right_texts)

        matches = []
        for i in range(len(left_texts)):
            for j in range(len(right_texts)):
                if similarity_matrix[i, j] >= threshold:
                    matches.append((i, j, float(similarity_matrix[i, j])))

        # Sort by similarity score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches


# ============================================================================
# Feature Extraction Join Operator
# ============================================================================

class FeatureExtractionJoin:
    """
    Main Feature Extraction Join operator.

    Workflow:
    1. Embed all images from left table
    2. Group similar images together
    3. Extract description for each group (VLM on representative image)
    4. Perform semantic text-text join with right table

    Usage:
        config = FeatureExtractionConfig(num_groups=20)
        fe_join = FeatureExtractionJoin(config)

        results, stats = fe_join.execute(
            left_images=image_list,
            left_ids=id_list,
            right_texts=text_list,
            right_ids=right_id_list,
            join_prompt="Match product images with their descriptions"
        )
    """

    def __init__(self, config: FeatureExtractionConfig = None, device: str = 'cuda'):
        self.config = config or FeatureExtractionConfig()
        self.device = device

        self.embedder = ImageEmbedder(self.config, device)
        self.grouper = ImageGrouper(self.config)
        self.extractor = FeatureExtractor(self.config)
        self.matcher = TextMatcher(self.config, device)

        self.stats = JoinStatistics()

    def execute(
        self,
        left_images: List[Union[bytes, str]],
        left_ids: List[Any],
        right_texts: List[str],
        right_ids: List[Any],
        join_prompt: Optional[str] = None,
        extraction_prompt: Optional[str] = None
    ) -> Tuple[List[JoinResult], JoinStatistics]:
        """
        Execute the Feature Extraction Join.

        Args:
            left_images: List of image data (bytes or paths) from left table
            left_ids: List of IDs for left table rows
            right_texts: List of text from right table
            right_ids: List of IDs for right table rows
            join_prompt: Custom prompt for join semantics (optional)
            extraction_prompt: Custom prompt for feature extraction (optional)

        Returns:
            Tuple of (list of JoinResult, statistics)
        """
        start_time = time.time()
        self.stats = JoinStatistics()
        self.stats.total_left_rows = len(left_images)
        self.stats.total_right_rows = len(right_texts)

        print(f"Feature Extraction Join: {len(left_images)} images x {len(right_texts)} texts")

        # Step 1: Embed all images
        print("Step 1: Embedding images...")
        embed_start = time.time()
        embeddings, valid_indices = self.embedder.embed_images_batch(left_images)
        self.stats.embedding_time = time.time() - embed_start

        if len(valid_indices) == 0:
            print("No valid images to process")
            return [], self.stats

        # Filter to valid images only
        valid_images = [left_images[i] for i in valid_indices]
        valid_ids = [left_ids[i] for i in valid_indices]

        # Step 2: Group similar images
        print("Step 2: Grouping similar images...")
        group_start = time.time()

        if self.config.enable_grouping:
            groups = self.grouper.group_images(embeddings, valid_ids)
        else:
            groups = {i: [i] for i in range(len(valid_images))}

        self.stats.grouping_time = time.time() - group_start
        self.stats.num_image_groups = len(groups)

        print(f"  Created {len(groups)} groups from {len(valid_images)} images")

        # Step 3: Extract descriptions for each group
        print("Step 3: Extracting descriptions (VLM calls)...")
        extract_start = time.time()

        # Map: group_id -> extracted description
        group_descriptions = {}
        # Map: image_index -> group_id
        image_to_group = {}
        # Map: image_index -> description
        image_descriptions = {}

        for group_id, member_indices in groups.items():
            # Get representative image
            rep_idx = self.grouper.get_representative_image(member_indices, embeddings)
            rep_image = valid_images[rep_idx]

            # Extract description using VLM
            description = self.extractor.extract_description(
                rep_image,
                extraction_prompt or self.config.extraction_prompt
            )
            self.stats.vlm_calls += 1

            group_descriptions[group_id] = description

            # Assign description to all group members
            for idx in member_indices:
                image_to_group[idx] = group_id
                image_descriptions[idx] = description

        self.stats.extraction_time = time.time() - extract_start

        print(f"  Made {self.stats.vlm_calls} VLM calls (saved {len(valid_images) - self.stats.vlm_calls})")

        # Step 4: Perform text-text semantic join
        print("Step 4: Semantic text-text join...")
        join_start = time.time()

        # Get unique descriptions (one per group)
        unique_descriptions = list(group_descriptions.values())
        group_id_list = list(group_descriptions.keys())

        # Find matches between descriptions and right texts
        matches = self.matcher.find_matches(
            unique_descriptions,
            right_texts,
            self.config.text_similarity_threshold
        )

        self.stats.text_comparisons = len(unique_descriptions) * len(right_texts)

        # Build results
        results = []
        for desc_idx, right_idx, similarity in matches:
            group_id = group_id_list[desc_idx]
            description = unique_descriptions[desc_idx]

            # Get all images in this group
            group_member_indices = groups[group_id]

            for img_idx in group_member_indices:
                result = JoinResult(
                    left_id=valid_ids[img_idx],
                    right_id=right_ids[right_idx],
                    left_image_path=valid_images[img_idx] if isinstance(valid_images[img_idx], str) else None,
                    extracted_description=description,
                    right_text=right_texts[right_idx],
                    similarity_score=similarity,
                    group_id=group_id,
                    matched=True
                )
                results.append(result)

        self.stats.join_time = time.time() - join_start
        self.stats.matched_pairs = len(results)
        self.stats.total_time = time.time() - start_time
        self.stats.compute_savings()

        print(f"\n--- Join Statistics ---")
        print(f"Total images: {self.stats.total_left_rows}")
        print(f"Total texts: {self.stats.total_right_rows}")
        print(f"Image groups: {self.stats.num_image_groups}")
        print(f"VLM calls: {self.stats.vlm_calls}")
        print(f"VLM cost saved: {self.stats.vlm_cost_saved:.1%}")
        print(f"Matched pairs: {self.stats.matched_pairs}")
        print(f"Total time: {self.stats.total_time:.2f}s")

        return results, self.stats

    def get_extracted_features(
        self,
        images: List[Union[bytes, str]],
        extraction_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Extract features/descriptions from images without joining.

        Useful for:
        - Pre-computing features
        - Debugging
        - Using extracted features in custom SQL
        """
        embeddings, valid_indices = self.embedder.embed_images_batch(images)
        valid_images = [images[i] for i in valid_indices]

        if self.config.enable_grouping:
            groups = self.grouper.group_images(embeddings, list(range(len(valid_images))))
        else:
            groups = {i: [i] for i in range(len(valid_images))}

        # Extract per group
        descriptions = [""] * len(images)

        for group_id, member_indices in groups.items():
            rep_idx = self.grouper.get_representative_image(member_indices, embeddings)
            rep_image = valid_images[rep_idx]

            desc = self.extractor.extract_description(
                rep_image,
                extraction_prompt or self.config.extraction_prompt
            )

            for idx in member_indices:
                original_idx = valid_indices[idx]
                descriptions[original_idx] = desc

        return descriptions


# ============================================================================
# Spark SQL UDF Integration
# ============================================================================

def create_feature_extraction_udf(
    config: FeatureExtractionConfig = None,
    extraction_prompt: Optional[str] = None
):
    """
    Create a Spark UDF for extracting features from images.

    The prompt can include {image} placeholder which will be replaced during extraction.

    Usage:
        spark.udf.register("EXTRACT_FEATURES", create_feature_extraction_udf(
            config,
            extraction_prompt="Describe the product in {image} focusing on category, color, and material"
        ))
        spark.sql("SELECT id, EXTRACT_FEATURES(image) as features FROM images")
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    fe_join = FeatureExtractionJoin(config or FeatureExtractionConfig())

    @pandas_udf(StringType())
    def extract_features_udf(images: pd.Series) -> pd.Series:
        """Extract text descriptions from images."""
        image_list = images.tolist()
        descriptions = fe_join.get_extracted_features(
            image_list,
            extraction_prompt
        )
        return pd.Series(descriptions)

    return extract_features_udf


def create_semantic_match_udf(
    config: FeatureExtractionConfig = None
):
    """
    Create a Spark UDF for semantic matching between texts.

    Usage:
        spark.udf.register("SEMANTIC_MATCH", create_semantic_match_udf(config))
        spark.sql('''
            SELECT l.*, r.*
            FROM left_table l, right_table r
            WHERE SEMANTIC_MATCH(l.extracted_features, r.description) > 0.5
        ''')
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import FloatType

    matcher = TextMatcher(config or FeatureExtractionConfig())

    @pandas_udf(FloatType())
    def semantic_match_udf(left_text: pd.Series, right_text: pd.Series) -> pd.Series:
        """Compute semantic similarity between text pairs."""
        scores = []

        for lt, rt in zip(left_text, right_text):
            if not lt or not rt:
                scores.append(0.0)
                continue

            sim_matrix = matcher.compute_similarity_matrix([lt], [rt])
            scores.append(float(sim_matrix[0, 0]))

        return pd.Series(scores)

    return semantic_match_udf


def create_feature_join_udf(
    config: FeatureExtractionConfig = None,
    prompt: str = "Extract features from {image} and determine if it matches the description: {text}",
    threshold: float = 0.5
):
    """
    Create a Spark UDF for Feature Extraction Join with a prompt that includes both join keys.

    The prompt MUST include both {image} and {text} placeholders:
    - {image}: Placeholder for the image from left table
    - {text}: Placeholder for the text from right table

    The prompt guides both:
    1. How to extract features from the image
    2. How to match with the text description

    Usage:
        spark.udf.register("FEATURE_JOIN", create_feature_join_udf(
            config,
            prompt="Given {image}, extract product features and check if they match: {text}",
            threshold=0.5
        ))

        spark.sql('''
            SELECT l.*, r.*
            FROM products_images l, product_descriptions r
            WHERE FEATURE_JOIN(l.image, r.description)
        ''')

    Example prompts:
        - "Extract visual features from {image} and match with product description: {text}"
        - "Analyze {image} to identify objects and verify against: {text}"
        - "Does the property shown in {image} match the listing: {text}?"
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BooleanType

    config = config or FeatureExtractionConfig()
    fe_join = FeatureExtractionJoin(config)

    # Validate prompt has both placeholders
    if "{image}" not in prompt or "{text}" not in prompt:
        raise ValueError("Prompt must include both {image} and {text} placeholders")

    # Parse prompt to separate extraction and matching parts
    # The part before {text} guides extraction, the full prompt guides matching
    extraction_prompt = prompt.split("{text}")[0].replace("{image}", "this image").strip()
    if extraction_prompt.endswith(":"):
        extraction_prompt = extraction_prompt[:-1]

    # Cache for extracted descriptions
    _description_cache = {}

    @pandas_udf(BooleanType())
    def feature_join_udf(images: pd.Series, texts: pd.Series) -> pd.Series:
        """Check if image matches text based on extracted features and the join prompt."""
        results = []

        # Batch extract descriptions for new images
        new_images = []
        new_indices = []

        for i, img in enumerate(images):
            img_key = str(img)[:100] if img else None
            if img_key and img_key not in _description_cache:
                new_images.append(img)
                new_indices.append(i)

        if new_images:
            descriptions = fe_join.get_extracted_features(new_images, extraction_prompt)
            for idx, desc in zip(new_indices, descriptions):
                img_key = str(images[idx])[:100]
                _description_cache[img_key] = desc

        # Compute matches
        matcher = TextMatcher(config)

        for img, txt in zip(images, texts):
            if not img or not txt:
                results.append(False)
                continue

            img_key = str(img)[:100]
            description = _description_cache.get(img_key, "")

            if not description:
                results.append(False)
                continue

            # Compute similarity
            sim_matrix = matcher.compute_similarity_matrix([description], [txt])
            similarity = float(sim_matrix[0, 0])

            results.append(similarity >= threshold)

        return pd.Series(results)

    return feature_join_udf


def create_feature_join_udf_with_prompt(
    config: FeatureExtractionConfig = None,
    threshold: float = 0.5
):
    """
    Create a Spark UDF for Feature Extraction Join where the prompt is passed as a SQL parameter.

    This allows the prompt to be specified at query time rather than at UDF registration.
    The prompt MUST include both {image} and {text} placeholders.

    Usage:
        spark.udf.register("FEATURE_JOIN_PROMPT", create_feature_join_udf_with_prompt(config))

        spark.sql('''
            SELECT l.*, r.*
            FROM products_images l, product_descriptions r
            WHERE FEATURE_JOIN_PROMPT(
                l.image,
                r.description,
                'Extract features from {image} and match with: {text}'
            )
        ''')

    SQL Signature:
        FEATURE_JOIN_PROMPT(image_col, text_col, prompt_string) -> boolean
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BooleanType

    config = config or FeatureExtractionConfig()

    # Global cache shared across batches
    _description_cache = {}
    _prompt_cache = {}  # Cache extraction prompts derived from join prompts

    @pandas_udf(BooleanType())
    def feature_join_with_prompt_udf(
        images: pd.Series,
        texts: pd.Series,
        prompts: pd.Series
    ) -> pd.Series:
        """
        Feature join with prompt as parameter.

        Args:
            images: Image data (bytes or paths)
            texts: Text descriptions to match
            prompts: Join prompt with {image} and {text} placeholders
        """
        results = []

        # Get unique prompts in this batch (usually just one)
        unique_prompts = prompts.unique()

        for join_prompt in unique_prompts:
            if not join_prompt or "{image}" not in join_prompt or "{text}" not in join_prompt:
                continue

            # Derive extraction prompt from join prompt
            if join_prompt not in _prompt_cache:
                extraction_prompt = join_prompt.split("{text}")[0].replace("{image}", "this image").strip()
                if extraction_prompt.endswith(":"):
                    extraction_prompt = extraction_prompt[:-1]
                _prompt_cache[join_prompt] = extraction_prompt

        # Process each row
        fe_join = FeatureExtractionJoin(config)
        matcher = TextMatcher(config)

        for img, txt, prompt_str in zip(images, texts, prompts):
            if not img or not txt or not prompt_str:
                results.append(False)
                continue

            if "{image}" not in prompt_str or "{text}" not in prompt_str:
                results.append(False)
                continue

            # Get or compute description
            cache_key = (str(img)[:100], prompt_str)

            if cache_key not in _description_cache:
                extraction_prompt = _prompt_cache.get(prompt_str, "Describe this image")
                descriptions = fe_join.get_extracted_features([img], extraction_prompt)
                _description_cache[cache_key] = descriptions[0] if descriptions else ""

            description = _description_cache[cache_key]

            if not description:
                results.append(False)
                continue

            # Compute similarity
            sim_matrix = matcher.compute_similarity_matrix([description], [txt])
            similarity = float(sim_matrix[0, 0])

            results.append(similarity >= threshold)

        return pd.Series(results)

    return feature_join_with_prompt_udf


def create_llm_feature_join_udf(
    config: FeatureExtractionConfig = None
):
    """
    Create a Spark UDF that uses VLM for direct image-text matching with a custom prompt.

    This UDF sends both the image and text to the VLM in a single call,
    using the prompt to guide the matching decision.

    The prompt should include:
    - {image}: Placeholder for the image
    - {text}: Placeholder for the text to match

    Usage:
        spark.udf.register("LLM_FEATURE_JOIN", create_llm_feature_join_udf(config))

        spark.sql('''
            SELECT l.*, r.*
            FROM products_images l, product_descriptions r
            WHERE LLM_FEATURE_JOIN(
                l.image,
                r.description,
                'Does {image} show the product described as: {text}? Answer Yes or No.'
            ) = 'Yes'
        ''')

    SQL Signature:
        LLM_FEATURE_JOIN(image_col, text_col, prompt_string) -> string (VLM response)
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    config = config or FeatureExtractionConfig()
    extractor = FeatureExtractor(config)

    @pandas_udf(StringType())
    def llm_feature_join_udf(
        images: pd.Series,
        texts: pd.Series,
        prompts: pd.Series
    ) -> pd.Series:
        """
        Direct VLM-based image-text matching.

        The VLM receives both the image and the text, guided by the prompt.
        """
        results = []

        for img, txt, prompt_str in zip(images, texts, prompts):
            if not img or not txt or not prompt_str:
                results.append("")
                continue

            # Build the actual prompt by replacing placeholders
            actual_prompt = prompt_str.replace("{text}", str(txt))
            # {image} is handled by sending the image to VLM

            # Remove {image} placeholder from text prompt (image is sent separately)
            actual_prompt = actual_prompt.replace("{image}", "the image")

            try:
                response = extractor.extract_description(img, actual_prompt)
                results.append(response)
            except Exception as e:
                print(f"VLM error: {e}")
                results.append("")

        return pd.Series(results)

    return llm_feature_join_udf


# ============================================================================
# High-Level API for Spark DataFrames
# ============================================================================

def feature_extraction_join(
    left_df,
    right_df,
    left_image_col: str,
    right_text_col: str,
    left_id_col: str = None,
    right_id_col: str = None,
    config: FeatureExtractionConfig = None,
    join_prompt: str = None,
    spark: 'SparkSession' = None
):
    """
    Perform Feature Extraction Join on Spark DataFrames.

    Args:
        left_df: DataFrame with image column
        right_df: DataFrame with text column
        left_image_col: Column name for images in left_df
        right_text_col: Column name for text in right_df
        left_id_col: Optional ID column in left_df
        right_id_col: Optional ID column in right_df
        config: FeatureExtractionConfig
        join_prompt: Custom prompt for extraction
        spark: SparkSession

    Returns:
        Joined DataFrame

    Example:
        result_df = feature_extraction_join(
            images_df,
            descriptions_df,
            left_image_col="image",
            right_text_col="description",
            config=FeatureExtractionConfig(num_groups=20)
        )
    """
    config = config or FeatureExtractionConfig()

    # Register UDFs
    extract_udf = create_feature_extraction_udf(config, join_prompt)
    match_udf = create_semantic_match_udf(config)

    if spark:
        spark.udf.register("EXTRACT_FEATURES", extract_udf)
        spark.udf.register("SEMANTIC_MATCH", match_udf)

        # Create temp views
        left_df.createOrReplaceTempView("_fe_left")
        right_df.createOrReplaceTempView("_fe_right")

        # Build SQL
        left_id = left_id_col or "monotonically_increasing_id() as left_id"
        right_id = right_id_col or "monotonically_increasing_id() as right_id"

        sql = f"""
            WITH left_with_features AS (
                SELECT *, EXTRACT_FEATURES({left_image_col}) as _extracted_features
                FROM _fe_left
            )
            SELECT l.*, r.*
            FROM left_with_features l
            CROSS JOIN _fe_right r
            WHERE SEMANTIC_MATCH(l._extracted_features, r.{right_text_col}) >= {config.text_similarity_threshold}
        """

        return spark.sql(sql)
    else:
        # Direct pandas operation
        from pyspark.sql.functions import col

        # Add extracted features
        left_with_features = left_df.withColumn(
            "_extracted_features",
            extract_udf(col(left_image_col))
        )

        # Cross join and filter
        crossed = left_with_features.crossJoin(right_df)

        return crossed.filter(
            match_udf(
                col("_extracted_features"),
                col(right_text_col)
            ) >= config.text_similarity_threshold
        )
