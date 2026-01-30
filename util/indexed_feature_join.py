"""
Feature Extraction Join with Vector Indexing (FE-Join V2)

Improved time complexity using Approximate Nearest Neighbor (ANN) search.

Problem with naive approach:
- N images × M texts = O(N*M) comparisons
- Even with grouping, text matching is still O(groups * M)

Solution: Vector Indexing
1. Embed all images into vector space (N embeddings)
2. Embed all texts into same vector space (M embeddings)
3. Build ANN index on text embeddings (O(M log M))
4. For each image, query top-k nearest texts (O(N * log M))
5. Only verify top-k candidates with VLM (O(N * k) where k << M)

Time Complexity:
- Naive: O(N * M)
- Indexed: O(N * log(M) + N * k) where k is candidates per image

Example: N=1000 images, M=10000 texts, k=10
- Naive: 10,000,000 comparisons
- Indexed: ~10,000 ANN queries + 10,000 VLM verifications
"""

import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
import json
import time
import requests
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IndexedJoinConfig:
    """Configuration for Vector-Indexed Feature Extraction Join."""
    # Embedding model (shared for images and texts)
    embedding_model: str = "openai/clip-vit-large-patch14-336"

    # ANN Index settings
    index_type: str = "faiss"  # "faiss", "annoy", "brute_force"
    num_candidates: int = 10   # k: number of candidates per image
    use_gpu_index: bool = True

    # VLM verification
    vlm_model: str = "llava-hf/llava-1.5-7b-hf"
    vlm_api_url: str = "http://localhost:8000/v1"
    enable_vlm_verification: bool = True

    # Thresholds
    similarity_threshold: float = 0.3  # Minimum similarity for candidates
    verification_threshold: float = 0.7  # For VLM verification

    # Batch processing
    batch_size: int = 32


@dataclass
class IndexedJoinResult:
    """Result of a vector-indexed join."""
    left_id: Any
    right_id: Any
    left_image: Optional[str] = None
    right_text: str = ""
    embedding_similarity: float = 0.0
    vlm_verified: bool = False
    vlm_response: str = ""


@dataclass
class IndexedJoinStats:
    """Statistics for indexed join."""
    total_images: int = 0
    total_texts: int = 0
    ann_queries: int = 0
    vlm_calls: int = 0
    matched_pairs: int = 0

    embedding_time: float = 0.0
    indexing_time: float = 0.0
    search_time: float = 0.0
    verification_time: float = 0.0
    total_time: float = 0.0

    @property
    def naive_comparisons(self) -> int:
        return self.total_images * self.total_texts

    @property
    def actual_comparisons(self) -> int:
        return self.vlm_calls

    @property
    def speedup(self) -> float:
        if self.actual_comparisons == 0:
            return float('inf')
        return self.naive_comparisons / self.actual_comparisons


# ============================================================================
# Unified Embedder (Images + Texts in same space)
# ============================================================================

class UnifiedEmbedder:
    """
    Embeds both images and texts into the same vector space using CLIP.
    This allows direct similarity comparison between images and texts.
    """

    def __init__(self, config: IndexedJoinConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None
        self._initialized = False

    def load_model(self):
        """Load CLIP model for unified embeddings."""
        if self._initialized:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel

            model_name = "openai/clip-vit-large-patch14"
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._initialized = True

        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise

    def embed_images(self, images: List[Union[bytes, str]]) -> np.ndarray:
        """Embed images into vector space."""
        self.load_model()

        all_embeddings = []

        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]
            batch_images = []

            for img_data in batch:
                try:
                    if isinstance(img_data, bytes):
                        img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    elif isinstance(img_data, str):
                        img = Image.open(img_data).convert('RGB')
                    else:
                        continue
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    # Add zero embedding for failed images
                    continue

            if not batch_images:
                continue

            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = self.model.get_image_features(**inputs)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu().numpy())

        if not all_embeddings:
            return np.array([])

        return np.vstack(all_embeddings)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts into the same vector space as images."""
        self.load_model()

        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            with torch.no_grad():
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = self.model.get_text_features(**inputs)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# ============================================================================
# Vector Index for Fast ANN Search
# ============================================================================

class VectorIndex:
    """
    Vector index for fast approximate nearest neighbor search.

    Supports:
    - FAISS (recommended for large datasets)
    - Brute force (for small datasets or debugging)
    """

    def __init__(self, config: IndexedJoinConfig):
        self.config = config
        self.index = None
        self.embeddings = None
        self.ids = None

    def build(self, embeddings: np.ndarray, ids: List[Any]):
        """Build the index from embeddings."""
        self.embeddings = embeddings.astype('float32')
        self.ids = ids
        dim = embeddings.shape[1]

        if self.config.index_type == "faiss":
            self._build_faiss_index(dim)
        else:
            # Brute force - just store embeddings
            pass

    def _build_faiss_index(self, dim: int):
        """Build FAISS index."""
        try:
            import faiss

            # Use IVF index for large datasets, flat for small
            n_vectors = self.embeddings.shape[0]

            if n_vectors < 1000:
                # Small dataset - use flat index
                self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
            else:
                # Larger dataset - use IVF
                n_clusters = min(int(np.sqrt(n_vectors)), 100)
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
                self.index.train(self.embeddings)

            self.index.add(self.embeddings)

            # Move to GPU if available and requested
            if self.config.use_gpu_index:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except:
                    pass  # Fall back to CPU

        except ImportError:
            print("FAISS not available, falling back to brute force")
            self.config.index_type = "brute_force"

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Returns:
            Tuple of (distances, indices) arrays of shape (n_queries, k)
        """
        query_embeddings = query_embeddings.astype('float32')

        if self.config.index_type == "faiss" and self.index is not None:
            distances, indices = self.index.search(query_embeddings, k)
            return distances, indices
        else:
            # Brute force search
            similarities = np.dot(query_embeddings, self.embeddings.T)

            # Get top-k for each query
            k = min(k, self.embeddings.shape[0])
            indices = np.argsort(-similarities, axis=1)[:, :k]
            distances = np.take_along_axis(similarities, indices, axis=1)

            return distances, indices

    def get_id(self, index: int) -> Any:
        """Get original ID from index."""
        return self.ids[index]


# ============================================================================
# VLM Verifier
# ============================================================================

class VLMVerifier:
    """
    Verifies image-text matches using VLM.
    Only called on top-k candidates from ANN search.
    """

    def __init__(self, config: IndexedJoinConfig):
        self.config = config

    def verify(
        self,
        image_data: Union[bytes, str],
        text: str,
        prompt: str
    ) -> Tuple[bool, str]:
        """
        Verify if image matches text using VLM.

        Args:
            image_data: Image bytes or path
            text: Text to match
            prompt: Prompt template with {image} and {text} placeholders

        Returns:
            Tuple of (is_match, vlm_response)
        """
        # Build prompt
        actual_prompt = prompt.replace("{text}", text).replace("{image}", "the image")

        # Encode image
        if isinstance(image_data, str):
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = image_data

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        payload = {
            "model": self.config.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                        {"type": "text", "text": actual_prompt}
                    ]
                }
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                f"{self.config.vlm_api_url}/chat/completions",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                is_match = result.lower().strip().startswith('yes')
                return is_match, result

        except Exception as e:
            print(f"VLM error: {e}")

        return False, ""


# ============================================================================
# Indexed Feature Extraction Join
# ============================================================================

class IndexedFeatureExtractionJoin:
    """
    Feature Extraction Join with Vector Indexing.

    Time Complexity: O(N * log(M) + N * k)
    - N: number of images
    - M: number of texts
    - k: candidates per image (default 10)

    Workflow:
    1. Embed all images (N embeddings)
    2. Embed all texts (M embeddings)
    3. Build ANN index on text embeddings
    4. For each image, find top-k text candidates via ANN
    5. Optionally verify top candidates with VLM

    Usage:
        config = IndexedJoinConfig(num_candidates=10)
        join = IndexedFeatureExtractionJoin(config)

        results, stats = join.execute(
            images=image_list,
            image_ids=image_id_list,
            texts=text_list,
            text_ids=text_id_list,
            prompt="Does {image} match: {text}? Answer Yes or No."
        )
    """

    def __init__(self, config: IndexedJoinConfig = None, device: str = 'cuda'):
        self.config = config or IndexedJoinConfig()
        self.device = device

        self.embedder = UnifiedEmbedder(self.config, device)
        self.index = VectorIndex(self.config)
        self.verifier = VLMVerifier(self.config)

        self.stats = IndexedJoinStats()

    def execute(
        self,
        images: List[Union[bytes, str]],
        image_ids: List[Any],
        texts: List[str],
        text_ids: List[Any],
        prompt: str = "Does {image} show what is described as: {text}? Answer Yes or No."
    ) -> Tuple[List[IndexedJoinResult], IndexedJoinStats]:
        """
        Execute the indexed feature extraction join.

        Args:
            images: List of image data (bytes or paths)
            image_ids: IDs for images
            texts: List of texts to match
            text_ids: IDs for texts
            prompt: VLM prompt with {image} and {text} placeholders

        Returns:
            Tuple of (results, statistics)
        """
        start_time = time.time()
        self.stats = IndexedJoinStats()
        self.stats.total_images = len(images)
        self.stats.total_texts = len(texts)

        print(f"Indexed Join: {len(images)} images × {len(texts)} texts")
        print(f"Naive comparisons would be: {len(images) * len(texts):,}")

        # Step 1: Embed images
        print("Step 1: Embedding images...")
        embed_start = time.time()
        image_embeddings = self.embedder.embed_images(images)

        # Step 2: Embed texts
        print("Step 2: Embedding texts...")
        text_embeddings = self.embedder.embed_texts(texts)
        self.stats.embedding_time = time.time() - embed_start

        # Step 3: Build index on texts
        print("Step 3: Building ANN index on texts...")
        index_start = time.time()
        self.index.build(text_embeddings, text_ids)
        self.stats.indexing_time = time.time() - index_start

        # Step 4: Search for each image
        print(f"Step 4: Finding top-{self.config.num_candidates} candidates per image...")
        search_start = time.time()

        k = min(self.config.num_candidates, len(texts))
        distances, indices = self.index.search(image_embeddings, k)
        self.stats.ann_queries = len(images)
        self.stats.search_time = time.time() - search_start

        # Step 5: Collect candidates above threshold
        print("Step 5: Filtering candidates by similarity threshold...")
        candidates = []

        for img_idx, (dists, text_indices) in enumerate(zip(distances, indices)):
            for dist, text_idx in zip(dists, text_indices):
                if dist >= self.config.similarity_threshold:
                    candidates.append({
                        'image_idx': img_idx,
                        'text_idx': int(text_idx),
                        'similarity': float(dist)
                    })

        print(f"  Found {len(candidates)} candidates above threshold {self.config.similarity_threshold}")

        # Step 6: Optional VLM verification
        results = []

        if self.config.enable_vlm_verification and candidates:
            print(f"Step 6: VLM verification on {len(candidates)} candidates...")
            verify_start = time.time()

            for cand in candidates:
                img_idx = cand['image_idx']
                text_idx = cand['text_idx']

                is_match, vlm_response = self.verifier.verify(
                    images[img_idx],
                    texts[text_idx],
                    prompt
                )
                self.stats.vlm_calls += 1

                if is_match:
                    results.append(IndexedJoinResult(
                        left_id=image_ids[img_idx],
                        right_id=text_ids[text_idx],
                        left_image=images[img_idx] if isinstance(images[img_idx], str) else None,
                        right_text=texts[text_idx],
                        embedding_similarity=cand['similarity'],
                        vlm_verified=True,
                        vlm_response=vlm_response
                    ))

            self.stats.verification_time = time.time() - verify_start
        else:
            # No VLM verification - return all candidates above threshold
            for cand in candidates:
                img_idx = cand['image_idx']
                text_idx = cand['text_idx']

                results.append(IndexedJoinResult(
                    left_id=image_ids[img_idx],
                    right_id=text_ids[text_idx],
                    left_image=images[img_idx] if isinstance(images[img_idx], str) else None,
                    right_text=texts[text_idx],
                    embedding_similarity=cand['similarity'],
                    vlm_verified=False
                ))

        self.stats.matched_pairs = len(results)
        self.stats.total_time = time.time() - start_time

        # Print statistics
        print(f"\n--- Statistics ---")
        print(f"Naive comparisons: {self.stats.naive_comparisons:,}")
        print(f"Actual VLM calls: {self.stats.vlm_calls}")
        print(f"Speedup: {self.stats.speedup:.1f}x")
        print(f"Matched pairs: {self.stats.matched_pairs}")
        print(f"Total time: {self.stats.total_time:.2f}s")

        return results, self.stats

    def execute_without_vlm(
        self,
        images: List[Union[bytes, str]],
        image_ids: List[Any],
        texts: List[str],
        text_ids: List[Any],
        top_k: int = 1
    ) -> List[IndexedJoinResult]:
        """
        Fast join using only embedding similarity (no VLM).

        Returns top-k text matches for each image based on embedding similarity.
        """
        # Embed
        image_embeddings = self.embedder.embed_images(images)
        text_embeddings = self.embedder.embed_texts(texts)

        # Build index
        self.index.build(text_embeddings, text_ids)

        # Search
        k = min(top_k, len(texts))
        distances, indices = self.index.search(image_embeddings, k)

        # Build results
        results = []
        for img_idx, (dists, text_indices) in enumerate(zip(distances, indices)):
            for dist, text_idx in zip(dists, text_indices):
                results.append(IndexedJoinResult(
                    left_id=image_ids[img_idx],
                    right_id=text_ids[int(text_idx)],
                    left_image=images[img_idx] if isinstance(images[img_idx], str) else None,
                    right_text=texts[int(text_idx)],
                    embedding_similarity=float(dist),
                    vlm_verified=False
                ))

        return results


# ============================================================================
# Spark SQL UDF with Indexing
# ============================================================================

def create_indexed_feature_join_udf(
    config: IndexedJoinConfig = None,
    prompt: str = "Does {image} match: {text}? Answer Yes or No."
):
    """
    Create a Spark UDF for indexed feature extraction join.

    This UDF pre-builds an index on the text column for fast lookups.

    Usage:
        # First, register the right table texts for indexing
        texts_df = spark.sql("SELECT id, text FROM right_table").collect()
        text_list = [row['text'] for row in texts_df]
        text_ids = [row['id'] for row in texts_df]

        # Create UDF with indexed texts
        join_udf = create_indexed_feature_join_udf_with_index(
            config,
            text_list,
            text_ids,
            prompt="Match {image} with {text}"
        )
        spark.udf.register("INDEXED_JOIN", join_udf)

        # Use in SQL
        spark.sql('''
            SELECT l.*, INDEXED_JOIN(l.image) as matched_text_id
            FROM left_table l
        ''')
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType

    config = config or IndexedJoinConfig()

    # This will be populated when texts are registered
    _text_index = None
    _text_ids = None
    _texts = None
    _embedder = None

    def register_texts(texts: List[str], text_ids: List[Any]):
        """Register texts and build index."""
        nonlocal _text_index, _text_ids, _texts, _embedder

        _embedder = UnifiedEmbedder(config)
        _texts = texts
        _text_ids = text_ids

        # Embed and index texts
        text_embeddings = _embedder.embed_texts(texts)
        _text_index = VectorIndex(config)
        _text_index.build(text_embeddings, text_ids)

    # Return type: array of (text_id, similarity) pairs
    result_schema = ArrayType(StructType([
        StructField("text_id", StringType(), True),
        StructField("similarity", FloatType(), True)
    ]))

    @pandas_udf(result_schema)
    def indexed_join_udf(images: pd.Series) -> pd.Series:
        """Find matching texts for each image using index."""
        if _text_index is None:
            raise ValueError("Texts not registered. Call register_texts() first.")

        results = []

        # Embed images
        image_list = images.tolist()
        image_embeddings = _embedder.embed_images(image_list)

        # Search index
        k = config.num_candidates
        distances, indices = _text_index.search(image_embeddings, k)

        for dists, text_indices in zip(distances, indices):
            matches = []
            for dist, text_idx in zip(dists, text_indices):
                if dist >= config.similarity_threshold:
                    matches.append({
                        "text_id": str(_text_ids[int(text_idx)]),
                        "similarity": float(dist)
                    })
            results.append(matches)

        return pd.Series(results)

    # Attach register function to UDF
    indexed_join_udf.register_texts = register_texts

    return indexed_join_udf


def create_indexed_feature_join_predicate_udf(
    texts: List[str],
    text_ids: List[Any],
    config: IndexedJoinConfig = None,
    prompt: str = "Does {image} match: {text}? Answer Yes or No.",
    threshold: float = 0.5
):
    """
    Create a Spark UDF that acts as a join predicate with pre-indexed texts.

    Usage:
        # Pre-index the right table
        texts = ["desc1", "desc2", ...]
        text_ids = ["id1", "id2", ...]

        join_udf = create_indexed_feature_join_predicate_udf(
            texts, text_ids, config,
            prompt="Does {image} match: {text}?"
        )
        spark.udf.register("INDEXED_MATCH", join_udf)

        # Use in SQL - much faster than cross join!
        spark.sql('''
            SELECT l.*, r.*
            FROM left_table l, right_table r
            WHERE INDEXED_MATCH(l.image, r.id, r.text)
        ''')
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BooleanType

    config = config or IndexedJoinConfig()

    # Build index once at registration
    embedder = UnifiedEmbedder(config)
    text_embeddings = embedder.embed_texts(texts)

    # Create mapping from text_id to index
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    text_id_to_embedding = {tid: text_embeddings[i] for i, tid in enumerate(text_ids)}

    # Cache for image embeddings
    _image_cache = {}

    @pandas_udf(BooleanType())
    def indexed_match_udf(
        images: pd.Series,
        right_text_ids: pd.Series,
        right_texts: pd.Series
    ) -> pd.Series:
        """
        Check if image matches specific text using cached embeddings.

        This is efficient because:
        1. Image embeddings are cached
        2. Text embeddings are pre-computed at registration
        3. Only computes similarity for the specific pair being checked
        """
        results = []

        for img, text_id, text in zip(images, right_text_ids, right_texts):
            if not img or not text_id:
                results.append(False)
                continue

            # Get or compute image embedding
            img_key = str(img)[:100]
            if img_key not in _image_cache:
                img_emb = embedder.embed_images([img])
                if len(img_emb) > 0:
                    _image_cache[img_key] = img_emb[0]
                else:
                    results.append(False)
                    continue

            img_embedding = _image_cache[img_key]

            # Get text embedding from pre-computed cache
            if text_id in text_id_to_embedding:
                text_embedding = text_id_to_embedding[text_id]
            else:
                # Text not in index, compute on the fly
                text_emb = embedder.embed_texts([text])
                text_embedding = text_emb[0]

            # Compute similarity
            similarity = np.dot(img_embedding, text_embedding)

            results.append(similarity >= threshold)

        return pd.Series(results)

    return indexed_match_udf
