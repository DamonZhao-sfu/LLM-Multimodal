"""
Hierarchical Visual-Feature Proxy Join (HVFP-Join)

A novel multimodal join operator that uses a cascade of visual feature extractors
to efficiently filter candidates before invoking expensive VLM calls.

Three-Phase Architecture:
- Phase 1: Vector-Space Pruning (Embedding Similarity Filter)
- Phase 2: Symbolic Feature Proxy (Object Detection, OCR, Predicate Decomposition)
- Phase 3: Semantic Verification (VLM Reasoning)

Cost Model:
    Standard: Cost_std = N * C_vlm
    HVFP-Join: Cost_ours = N * C_emb + (N * s1) * C_det + (N * s1 * s2) * C_vlm

    Where s1, s2 are selectivities of Phase 1 and Phase 2 respectively.
    Goal: Cost_ours ~ 0.15 * Cost_std when C_det << C_vlm and s2 ~ 0.1
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
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd


# ============================================================================
# Data Structures
# ============================================================================

class JoinPhase(Enum):
    """Enumeration of HVFP-Join phases."""
    PHASE1_VECTOR_PRUNING = 1
    PHASE2_SYMBOLIC_PROXY = 2
    PHASE3_SEMANTIC_VERIFICATION = 3


@dataclass
class HVFPConfig:
    """Configuration for HVFP-Join operator."""
    # Phase 1: Vector-Space Pruning
    embedding_model: str = "openai/clip-vit-large-patch14-336"
    similarity_threshold: float = 0.25  # tau_coarse
    top_k_candidates: int = 100  # For k-NN filtering

    # Phase 2: Symbolic Feature Proxy
    enable_object_detection: bool = True
    enable_ocr: bool = True
    enable_predicate_decomposition: bool = True
    detector_model: str = "yolo-world"  # or "grounding-dino"
    detector_confidence_threshold: float = 0.3
    ocr_engine: str = "paddleocr"  # or "easyocr"
    decomposition_model: str = "llama-3-8b"  # Small LLM for predicate decomposition

    # Phase 3: Semantic Verification
    vlm_model: str = "llava-hf/llava-1.5-7b-hf"
    vlm_api_url: str = "http://localhost:8000/v1"
    vlm_temperature: float = 0.0

    # Cost tracking
    track_costs: bool = True

    # Batch processing
    batch_size: int = 32


@dataclass
class JoinCandidate:
    """Represents a candidate row for the join operation."""
    row_id: int
    image_data: Optional[bytes] = None
    image_path: Optional[str] = None
    text_data: Dict[str, Any] = field(default_factory=dict)

    # Phase results
    phase1_score: float = 0.0
    phase1_passed: bool = False

    phase2_detected_objects: List[str] = field(default_factory=list)
    phase2_detected_text: List[str] = field(default_factory=list)
    phase2_passed: bool = False

    phase3_vlm_response: Optional[str] = None
    phase3_passed: bool = False

    # Final result
    join_result: bool = False


@dataclass
class JoinStatistics:
    """Statistics for HVFP-Join execution."""
    total_candidates: int = 0
    phase1_passed: int = 0
    phase2_passed: int = 0
    phase3_passed: int = 0

    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0
    total_time: float = 0.0

    # Cost estimates (in arbitrary units)
    embedding_cost: float = 0.0
    detection_cost: float = 0.0
    vlm_cost: float = 0.0
    total_cost: float = 0.0

    @property
    def phase1_selectivity(self) -> float:
        return self.phase1_passed / max(self.total_candidates, 1)

    @property
    def phase2_selectivity(self) -> float:
        return self.phase2_passed / max(self.phase1_passed, 1)

    @property
    def overall_selectivity(self) -> float:
        return self.phase3_passed / max(self.total_candidates, 1)

    @property
    def cost_reduction(self) -> float:
        """Estimated cost reduction compared to standard VLM-only approach."""
        standard_cost = self.total_candidates * 1.0  # Normalized VLM cost = 1.0
        return 1.0 - (self.total_cost / max(standard_cost, 1))


# ============================================================================
# Phase 1: Vector-Space Pruning (The "Gist" Filter)
# ============================================================================

class VectorSpacePruner:
    """
    Phase 1: Vector-Space Pruning using CLIP/SigLIP embeddings.

    Converts join predicate and images into a shared embedding space
    and performs similarity-based filtering.
    """

    def __init__(self, config: HVFPConfig, vision_tower=None, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.vision_tower = vision_tower
        self._text_embeddings_cache = {}

    def load_model(self):
        """Load the CLIP vision tower for embeddings."""
        if self.vision_tower is not None:
            return

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

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text query into embedding space."""
        if text in self._text_embeddings_cache:
            return self._text_embeddings_cache[text]

        self.load_model()

        with torch.no_grad():
            text_inputs = self.vision_tower.text_tokenizer(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_embeds = self.vision_tower.text_tower(**text_inputs).text_embeds
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        self._text_embeddings_cache[text] = text_embeds
        return text_embeds

    def encode_image(self, image_data: Union[bytes, str, Image.Image]) -> torch.Tensor:
        """Encode image into embedding space."""
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

        # Process image
        inputs = self.vision_tower.image_processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.vision_tower.dtype)

        with torch.no_grad():
            image_forward_outs = self.vision_tower.vision_tower(
                pixel_values,
                output_hidden_states=True
            )
            # Get CLS token embedding
            image_embeds = image_forward_outs.last_hidden_state[:, 0, :]

            # Project to shared space
            image_embeds = self.vision_tower.vision_tower.visual_projection(image_embeds)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        return image_embeds

    def compute_similarity(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor
    ) -> float:
        """Compute cosine similarity between text and image embeddings."""
        similarity = torch.matmul(text_embedding, image_embedding.t())
        return similarity.item()

    def filter_candidates(
        self,
        predicate: str,
        candidates: List[JoinCandidate],
        mode: str = "threshold"  # "threshold" or "topk"
    ) -> List[JoinCandidate]:
        """
        Filter candidates using vector similarity.

        Args:
            predicate: The join predicate text
            candidates: List of candidate rows
            mode: Filtering mode - "threshold" or "topk"

        Returns:
            Filtered list of candidates that pass Phase 1
        """
        text_embedding = self.encode_text(predicate)

        for candidate in candidates:
            try:
                image_data = candidate.image_data or candidate.image_path
                if image_data is None:
                    candidate.phase1_score = 0.0
                    candidate.phase1_passed = False
                    continue

                image_embedding = self.encode_image(image_data)
                similarity = self.compute_similarity(text_embedding, image_embedding)

                candidate.phase1_score = similarity

                if mode == "threshold":
                    candidate.phase1_passed = similarity >= self.config.similarity_threshold

            except Exception as e:
                print(f"Phase 1 error for candidate {candidate.row_id}: {e}")
                candidate.phase1_score = 0.0
                candidate.phase1_passed = False

        # For top-k mode, select top candidates
        if mode == "topk":
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.phase1_score,
                reverse=True
            )
            for i, candidate in enumerate(sorted_candidates):
                candidate.phase1_passed = i < self.config.top_k_candidates

        return [c for c in candidates if c.phase1_passed]

    def batch_encode_images(
        self,
        image_data_list: List[Union[bytes, str]]
    ) -> torch.Tensor:
        """Batch encode multiple images for efficiency."""
        self.load_model()

        images = []
        for image_data in image_data_list:
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            elif isinstance(image_data, str):
                image = Image.open(image_data).convert('RGB')
            else:
                continue
            images.append(image)

        if not images:
            return torch.tensor([])

        inputs = self.vision_tower.image_processor(images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.vision_tower.dtype)

        with torch.no_grad():
            image_forward_outs = self.vision_tower.vision_tower(
                pixel_values,
                output_hidden_states=True
            )
            image_embeds = image_forward_outs.last_hidden_state[:, 0, :]
            image_embeds = self.vision_tower.vision_tower.visual_projection(image_embeds)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        return image_embeds


# ============================================================================
# Phase 2: Symbolic Feature Proxy (The "Fact" Filter)
# ============================================================================

class PredicateDecomposer:
    """
    Decomposes complex join predicates into verifiable visual facts.

    Example: "luxury outdoor living with pool" ->
             ["swimming_pool", "outdoor_furniture", "garden"]
    """

    def __init__(self, config: HVFPConfig):
        self.config = config
        self._decomposition_cache = {}

    def decompose(self, predicate: str) -> Dict[str, List[str]]:
        """
        Decompose predicate into visual facts.

        Returns:
            Dictionary with keys:
            - 'objects': List of objects to detect
            - 'attributes': List of attributes to verify
            - 'text_patterns': List of text patterns to find via OCR
            - 'spatial_relations': List of spatial relationships
        """
        if predicate in self._decomposition_cache:
            return self._decomposition_cache[predicate]

        # Rule-based decomposition for common real estate terms
        result = {
            'objects': [],
            'attributes': [],
            'text_patterns': [],
            'spatial_relations': []
        }

        predicate_lower = predicate.lower()

        # Object mapping for real estate domain
        object_keywords = {
            'pool': ['swimming_pool', 'pool'],
            'swimming pool': ['swimming_pool'],
            'fire pit': ['fire_pit', 'firepit'],
            'fireplace': ['fireplace', 'fire'],
            'kitchen': ['kitchen', 'stove', 'oven', 'refrigerator'],
            "chef's kitchen": ['kitchen', 'stove', 'oven', 'kitchen_island'],
            'stainless steel': ['stainless_steel_appliance', 'refrigerator', 'oven'],
            'marble': ['marble_countertop', 'countertop'],
            'granite': ['granite_countertop', 'countertop'],
            'hardwood': ['hardwood_floor', 'wood_floor'],
            'garage': ['garage', 'car'],
            'garden': ['garden', 'plants', 'flowers'],
            'patio': ['patio', 'outdoor_furniture'],
            'deck': ['deck', 'wooden_deck'],
            'balcony': ['balcony'],
            'bathroom': ['bathroom', 'toilet', 'bathtub', 'shower'],
            'bedroom': ['bedroom', 'bed'],
            'living room': ['living_room', 'sofa', 'couch'],
            'dining room': ['dining_room', 'dining_table'],
            'outdoor': ['outdoor', 'yard', 'garden', 'patio'],
            'luxury': ['chandelier', 'marble', 'high_ceiling'],
            'modern': ['modern_furniture', 'minimalist'],
            'vintage': ['vintage', 'antique'],
        }

        # Attribute keywords
        attribute_keywords = {
            'spacious': 'spacious',
            'bright': 'bright',
            'natural light': 'natural_light',
            'renovated': 'renovated',
            'updated': 'updated',
            'new': 'new_construction',
            'open concept': 'open_concept',
            'open floor plan': 'open_floor_plan',
            'high ceiling': 'high_ceiling',
            'vaulted ceiling': 'vaulted_ceiling',
        }

        # Text patterns for OCR
        text_patterns = {
            'price drop': ['price drop', 'reduced', '$ off', 'sale'],
            'for sale': ['for sale', 'listing'],
            'open house': ['open house'],
            'sold': ['sold', 'pending'],
            'new listing': ['new listing', 'just listed'],
        }

        # Extract objects
        for keyword, objects in object_keywords.items():
            if keyword in predicate_lower:
                result['objects'].extend(objects)

        # Extract attributes
        for keyword, attr in attribute_keywords.items():
            if keyword in predicate_lower:
                result['attributes'].append(attr)

        # Extract text patterns
        for keyword, patterns in text_patterns.items():
            if keyword in predicate_lower:
                result['text_patterns'].extend(patterns)

        # Remove duplicates
        result['objects'] = list(set(result['objects']))
        result['attributes'] = list(set(result['attributes']))
        result['text_patterns'] = list(set(result['text_patterns']))

        self._decomposition_cache[predicate] = result
        return result

    def decompose_with_llm(
        self,
        predicate: str,
        api_url: str = "http://localhost:8000/v1"
    ) -> Dict[str, List[str]]:
        """
        Use a small LLM to decompose complex predicates.

        This is more accurate but slower than rule-based decomposition.
        """
        prompt = f"""Decompose the following search query into visual elements that can be detected in images.

Query: "{predicate}"

Respond in JSON format with these keys:
- "objects": List of objects/items that should be visible (e.g., "swimming_pool", "fireplace")
- "attributes": List of visual attributes to check (e.g., "stainless_steel", "marble")
- "text_patterns": List of text that might appear in the image (e.g., "FOR SALE", "Price Drop")

Only include items that can be visually verified. Be specific and use snake_case.
JSON response:"""

        try:
            response = requests.post(
                f"{api_url}/completions",
                json={
                    "model": self.config.decomposition_model,
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.0
                },
                timeout=30
            )

            if response.status_code == 200:
                result_text = response.json()['choices'][0]['text']
                # Parse JSON from response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

        except Exception as e:
            print(f"LLM decomposition failed: {e}")

        # Fallback to rule-based
        return self.decompose(predicate)


class ObjectDetector:
    """
    Runs object detection on images using YOLO-World or Grounding DINO.

    This provides cheap, fast verification of specific objects in images.
    """

    def __init__(self, config: HVFPConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = None
        self._initialized = False

    def load_model(self):
        """Load the object detection model."""
        if self._initialized:
            return

        try:
            if self.config.detector_model == "yolo-world":
                # Try to load YOLO-World
                from ultralytics import YOLO
                self.model = YOLO('yolov8x-worldv2.pt')
                self.model_type = "yolo-world"
            elif self.config.detector_model == "grounding-dino":
                # Grounding DINO integration
                from groundingdino.util.inference import load_model as load_gdino
                self.model = load_gdino(
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "weights/groundingdino_swint_ogc.pth"
                )
                self.model_type = "grounding-dino"
            else:
                # Fallback to standard YOLO
                from ultralytics import YOLO
                self.model = YOLO('yolov8x.pt')
                self.model_type = "yolo"

            self._initialized = True

        except ImportError as e:
            print(f"Object detector not available: {e}")
            self.model = None
            self._initialized = True

    def detect(
        self,
        image_data: Union[bytes, str, Image.Image],
        target_classes: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Detect specific objects in an image.

        Args:
            image_data: Image to process
            target_classes: List of object classes to detect

        Returns:
            Dictionary mapping class names to list of detections with confidence
        """
        self.load_model()

        if self.model is None:
            # Return empty result if no model available
            return {cls: [] for cls in target_classes}

        # Load image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, str):
            image = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, Image.Image):
            image = image_data.convert('RGB')
        else:
            return {cls: [] for cls in target_classes}

        results = {cls: [] for cls in target_classes}

        try:
            if self.model_type == "yolo-world":
                # YOLO-World supports open vocabulary
                self.model.set_classes(target_classes)
                detections = self.model.predict(image, conf=self.config.detector_confidence_threshold)

                for det in detections[0].boxes:
                    cls_id = int(det.cls[0])
                    cls_name = target_classes[cls_id] if cls_id < len(target_classes) else "unknown"
                    confidence = float(det.conf[0])
                    bbox = det.xyxy[0].tolist()

                    if cls_name in results:
                        results[cls_name].append({
                            'confidence': confidence,
                            'bbox': bbox
                        })

            elif self.model_type == "yolo":
                # Standard YOLO with COCO classes
                detections = self.model.predict(image, conf=self.config.detector_confidence_threshold)

                # Map COCO classes to target classes
                coco_to_target = self._map_coco_to_targets(target_classes)

                for det in detections[0].boxes:
                    cls_name = self.model.names[int(det.cls[0])]
                    confidence = float(det.conf[0])

                    # Check if COCO class maps to any target
                    for target in target_classes:
                        if cls_name in coco_to_target.get(target, []):
                            results[target].append({
                                'confidence': confidence,
                                'bbox': det.xyxy[0].tolist()
                            })

        except Exception as e:
            print(f"Object detection error: {e}")

        return results

    def _map_coco_to_targets(self, target_classes: List[str]) -> Dict[str, List[str]]:
        """Map target class names to COCO class names."""
        mapping = {
            'swimming_pool': ['pool'],
            'pool': ['pool'],
            'car': ['car', 'truck'],
            'sofa': ['couch', 'sofa'],
            'couch': ['couch', 'sofa'],
            'bed': ['bed'],
            'dining_table': ['dining table'],
            'toilet': ['toilet'],
            'tv': ['tv', 'television'],
            'refrigerator': ['refrigerator'],
            'oven': ['oven'],
            'microwave': ['microwave'],
            'sink': ['sink'],
            'chair': ['chair'],
            'potted_plant': ['potted plant'],
            'plant': ['potted plant'],
        }

        result = {}
        for target in target_classes:
            target_lower = target.lower().replace('_', ' ')
            if target in mapping:
                result[target] = mapping[target]
            else:
                result[target] = [target_lower]

        return result

    def check_objects_present(
        self,
        image_data: Union[bytes, str, Image.Image],
        required_objects: List[str],
        mode: str = "any"  # "any" or "all"
    ) -> Tuple[bool, List[str]]:
        """
        Check if required objects are present in the image.

        Args:
            image_data: Image to check
            required_objects: Objects that should be present
            mode: "any" (at least one) or "all" (all required)

        Returns:
            Tuple of (passed, list of detected objects)
        """
        if not required_objects:
            return True, []

        detections = self.detect(image_data, required_objects)
        detected = [obj for obj, dets in detections.items() if len(dets) > 0]

        if mode == "any":
            passed = len(detected) > 0
        else:  # all
            passed = len(detected) == len(required_objects)

        return passed, detected


class OCREngine:
    """
    OCR engine for detecting text in images.

    Useful for detecting price drops, "For Sale" signs, brand names, etc.
    """

    def __init__(self, config: HVFPConfig):
        self.config = config
        self.reader = None
        self._initialized = False

    def load_model(self):
        """Load the OCR engine."""
        if self._initialized:
            return

        try:
            if self.config.ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR
                self.reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                self.engine_type = "paddleocr"
            else:
                import easyocr
                self.reader = easyocr.Reader(['en'])
                self.engine_type = "easyocr"

            self._initialized = True

        except ImportError as e:
            print(f"OCR engine not available: {e}")
            self.reader = None
            self._initialized = True

    def extract_text(
        self,
        image_data: Union[bytes, str, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Extract all text from an image.

        Returns:
            List of dictionaries with 'text', 'confidence', and 'bbox'
        """
        self.load_model()

        if self.reader is None:
            return []

        # Load image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, str):
            image = image_data  # Pass path directly
        elif isinstance(image_data, Image.Image):
            # Convert to numpy for OCR
            image = np.array(image_data.convert('RGB'))
        else:
            return []

        results = []

        try:
            if self.engine_type == "paddleocr":
                ocr_result = self.reader.ocr(image, cls=True)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        bbox, (text, confidence) = line
                        results.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            else:  # easyocr
                ocr_result = self.reader.readtext(image)
                for bbox, text, confidence in ocr_result:
                    results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })

        except Exception as e:
            print(f"OCR error: {e}")

        return results

    def find_patterns(
        self,
        image_data: Union[bytes, str, Image.Image],
        patterns: List[str],
        case_sensitive: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Find specific text patterns in an image.

        Args:
            image_data: Image to search
            patterns: Text patterns to find
            case_sensitive: Whether matching is case-sensitive

        Returns:
            Tuple of (any_found, list of found patterns)
        """
        if not patterns:
            return True, []

        text_results = self.extract_text(image_data)
        all_text = " ".join([r['text'] for r in text_results])

        if not case_sensitive:
            all_text = all_text.lower()
            patterns = [p.lower() for p in patterns]

        found = []
        for pattern in patterns:
            if pattern in all_text:
                found.append(pattern)

        return len(found) > 0, found


class SymbolicFeatureProxy:
    """
    Phase 2: Combines object detection, OCR, and predicate decomposition
    to filter candidates based on verifiable visual facts.
    """

    def __init__(self, config: HVFPConfig, device: str = 'cuda'):
        self.config = config
        self.decomposer = PredicateDecomposer(config)
        self.detector = ObjectDetector(config, device) if config.enable_object_detection else None
        self.ocr = OCREngine(config) if config.enable_ocr else None

    def filter_candidates(
        self,
        predicate: str,
        candidates: List[JoinCandidate],
        required_match_ratio: float = 0.5
    ) -> List[JoinCandidate]:
        """
        Filter candidates using symbolic feature verification.

        Args:
            predicate: The join predicate
            candidates: Candidates that passed Phase 1
            required_match_ratio: Fraction of decomposed elements that must match

        Returns:
            Candidates that pass Phase 2
        """
        # Decompose predicate into visual facts
        decomposed = self.decomposer.decompose(predicate)

        required_objects = decomposed.get('objects', [])
        text_patterns = decomposed.get('text_patterns', [])

        total_checks = len(required_objects) + len(text_patterns)

        for candidate in candidates:
            try:
                image_data = candidate.image_data or candidate.image_path
                if image_data is None:
                    candidate.phase2_passed = False
                    continue

                matches = 0

                # Check objects
                if self.detector and required_objects:
                    passed, detected = self.detector.check_objects_present(
                        image_data,
                        required_objects,
                        mode="any"
                    )
                    candidate.phase2_detected_objects = detected
                    matches += len(detected)

                # Check text patterns via OCR
                if self.ocr and text_patterns:
                    found, patterns = self.ocr.find_patterns(
                        image_data,
                        text_patterns
                    )
                    candidate.phase2_detected_text = patterns
                    matches += len(patterns)

                # Determine if candidate passes
                if total_checks > 0:
                    match_ratio = matches / total_checks
                    candidate.phase2_passed = match_ratio >= required_match_ratio
                else:
                    # No specific checks required, pass through
                    candidate.phase2_passed = True

            except Exception as e:
                print(f"Phase 2 error for candidate {candidate.row_id}: {e}")
                candidate.phase2_passed = False

        return [c for c in candidates if c.phase2_passed]


# ============================================================================
# Phase 3: Semantic Verification (The "Reasoning" Filter)
# ============================================================================

class SemanticVerifier:
    """
    Phase 3: Uses a SOTA VLM to perform semantic verification
    on candidates that passed earlier phases.
    """

    def __init__(self, config: HVFPConfig, vision_tower=None, model=None):
        self.config = config
        self.vision_tower = vision_tower
        self.model = model

    def load_models(self):
        """Load VLM models for verification."""
        if self.vision_tower is not None and self.model is not None:
            return

        from util.mllm import load_vision_models
        self.vision_tower, self.model, self.tokenizer = load_vision_models(device='cuda')

    def verify_single(
        self,
        candidate: JoinCandidate,
        predicate: str,
        context: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify a single candidate using VLM.

        Args:
            candidate: The candidate to verify
            predicate: The join predicate/question
            context: Additional context from Phase 2 detections

        Returns:
            Tuple of (passed, vlm_response)
        """
        # Build prompt with context
        if context:
            prompt = f"""Based on the image and the following context:
{context}

Answer this question with "Yes" or "No":
{predicate}"""
        else:
            prompt = f"""Look at the image and answer with "Yes" or "No":
{predicate}"""

        try:
            image_data = candidate.image_data or candidate.image_path
            if image_data is None:
                return False, "No image data"

            # Call VLM API
            response = self._call_vlm_api(image_data, prompt)

            # Parse response
            response_lower = response.lower().strip()
            passed = response_lower.startswith("yes") or "yes" in response_lower[:10]

            return passed, response

        except Exception as e:
            print(f"VLM verification error: {e}")
            return False, str(e)

    def _call_vlm_api(
        self,
        image_data: Union[bytes, str],
        prompt: str
    ) -> str:
        """Call the VLM API with image and prompt."""
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
            "max_tokens": 100,
            "temperature": self.config.vlm_temperature,
            "guided_choice": ["Yes", "No"]
        }

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
            raise Exception(f"VLM API error: {response.status_code} - {response.text}")

    def verify_candidates(
        self,
        candidates: List[JoinCandidate],
        predicate: str
    ) -> List[JoinCandidate]:
        """
        Verify all candidates using VLM.

        Args:
            candidates: Candidates that passed Phase 2
            predicate: The join predicate

        Returns:
            Candidates that pass Phase 3 verification
        """
        for candidate in candidates:
            # Build context from Phase 2 results
            context_parts = []
            if candidate.phase2_detected_objects:
                context_parts.append(
                    f"Detected objects: {', '.join(candidate.phase2_detected_objects)}"
                )
            if candidate.phase2_detected_text:
                context_parts.append(
                    f"Detected text: {', '.join(candidate.phase2_detected_text)}"
                )
            context = "\n".join(context_parts) if context_parts else None

            passed, response = self.verify_single(candidate, predicate, context)
            candidate.phase3_passed = passed
            candidate.phase3_vlm_response = response
            candidate.join_result = passed

        return [c for c in candidates if c.phase3_passed]


# ============================================================================
# HVFP-Join Main Operator
# ============================================================================

class HVFPJoin:
    """
    Main HVFP-Join operator implementing the three-phase cascade.

    Usage:
        config = HVFPConfig(similarity_threshold=0.3)
        hvfp = HVFPJoin(config)

        results = hvfp.execute(
            predicate="luxury outdoor living with swimming pool",
            candidates=candidate_list
        )
    """

    def __init__(self, config: HVFPConfig = None, device: str = 'cuda'):
        self.config = config or HVFPConfig()
        self.device = device

        # Initialize phases
        self.phase1 = VectorSpacePruner(self.config, device=device)
        self.phase2 = SymbolicFeatureProxy(self.config, device=device)
        self.phase3 = SemanticVerifier(self.config)

        # Statistics
        self.stats = JoinStatistics()

    def execute(
        self,
        predicate: str,
        candidates: List[JoinCandidate],
        skip_phase1: bool = False,
        skip_phase2: bool = False,
        skip_phase3: bool = False
    ) -> Tuple[List[JoinCandidate], JoinStatistics]:
        """
        Execute the HVFP-Join operation.

        Args:
            predicate: The join predicate
            candidates: List of candidate rows
            skip_phase1: Skip vector pruning (for ablation studies)
            skip_phase2: Skip symbolic proxy (for ablation studies)
            skip_phase3: Skip VLM verification (return Phase 2 results)

        Returns:
            Tuple of (filtered candidates, execution statistics)
        """
        self.stats = JoinStatistics()
        self.stats.total_candidates = len(candidates)

        start_time = time.time()
        current_candidates = candidates

        # Phase 1: Vector-Space Pruning
        if not skip_phase1:
            phase1_start = time.time()
            current_candidates = self.phase1.filter_candidates(
                predicate,
                current_candidates,
                mode="threshold"
            )
            self.stats.phase1_time = time.time() - phase1_start
            self.stats.phase1_passed = len(current_candidates)
            self.stats.embedding_cost = self.stats.total_candidates * 0.001  # Low cost
        else:
            self.stats.phase1_passed = len(current_candidates)
            for c in current_candidates:
                c.phase1_passed = True

        print(f"Phase 1: {self.stats.phase1_passed}/{self.stats.total_candidates} passed "
              f"(selectivity: {self.stats.phase1_selectivity:.2%})")

        # Phase 2: Symbolic Feature Proxy
        if not skip_phase2 and current_candidates:
            phase2_start = time.time()
            current_candidates = self.phase2.filter_candidates(
                predicate,
                current_candidates
            )
            self.stats.phase2_time = time.time() - phase2_start
            self.stats.phase2_passed = len(current_candidates)
            self.stats.detection_cost = self.stats.phase1_passed * 0.01  # Medium cost
        else:
            self.stats.phase2_passed = len(current_candidates)
            for c in current_candidates:
                c.phase2_passed = True

        print(f"Phase 2: {self.stats.phase2_passed}/{self.stats.phase1_passed} passed "
              f"(selectivity: {self.stats.phase2_selectivity:.2%})")

        # Phase 3: Semantic Verification
        if not skip_phase3 and current_candidates:
            phase3_start = time.time()
            current_candidates = self.phase3.verify_candidates(
                current_candidates,
                predicate
            )
            self.stats.phase3_time = time.time() - phase3_start
            self.stats.phase3_passed = len(current_candidates)
            self.stats.vlm_cost = self.stats.phase2_passed * 1.0  # High cost (normalized)
        else:
            self.stats.phase3_passed = len(current_candidates)
            for c in current_candidates:
                c.phase3_passed = True
                c.join_result = True

        print(f"Phase 3: {self.stats.phase3_passed}/{self.stats.phase2_passed} passed")

        # Calculate totals
        self.stats.total_time = time.time() - start_time
        self.stats.total_cost = (
            self.stats.embedding_cost +
            self.stats.detection_cost +
            self.stats.vlm_cost
        )

        print(f"\nTotal: {self.stats.phase3_passed}/{self.stats.total_candidates} "
              f"(overall selectivity: {self.stats.overall_selectivity:.2%})")
        print(f"Estimated cost reduction: {self.stats.cost_reduction:.2%}")
        print(f"Total time: {self.stats.total_time:.2f}s")

        return current_candidates, self.stats

    def execute_batch(
        self,
        predicate: str,
        image_data_list: List[Union[bytes, str]],
        text_data_list: Optional[List[Dict]] = None
    ) -> Tuple[List[int], JoinStatistics]:
        """
        Convenience method for batch execution.

        Args:
            predicate: The join predicate
            image_data_list: List of image data (bytes or paths)
            text_data_list: Optional list of text metadata per image

        Returns:
            Tuple of (list of passing indices, statistics)
        """
        # Create candidates
        candidates = []
        for i, image_data in enumerate(image_data_list):
            text_data = text_data_list[i] if text_data_list else {}

            if isinstance(image_data, str):
                candidate = JoinCandidate(
                    row_id=i,
                    image_path=image_data,
                    text_data=text_data
                )
            else:
                candidate = JoinCandidate(
                    row_id=i,
                    image_data=image_data,
                    text_data=text_data
                )
            candidates.append(candidate)

        # Execute join
        results, stats = self.execute(predicate, candidates)

        # Return passing indices
        passing_indices = [c.row_id for c in results]
        return passing_indices, stats


# ============================================================================
# Spark SQL UDF Integration
# ============================================================================

def create_hvfp_join_udf(
    config: HVFPConfig = None,
    predicate: str = None
):
    """
    Create a Spark UDF for HVFP-Join filtering.

    Usage in Spark SQL:
        spark.udf.register("HVFP_FILTER", create_hvfp_join_udf(
            config=config,
            predicate="luxury outdoor living with pool"
        ))

        SELECT * FROM listings WHERE HVFP_FILTER(image_col) = true
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BooleanType

    hvfp = HVFPJoin(config or HVFPConfig())

    @pandas_udf(BooleanType())
    def hvfp_filter_udf(images: pd.Series) -> pd.Series:
        """Filter images using HVFP-Join cascade."""
        candidates = []
        for i, image_data in enumerate(images):
            if image_data is not None:
                candidates.append(JoinCandidate(
                    row_id=i,
                    image_data=image_data if isinstance(image_data, bytes) else None,
                    image_path=image_data if isinstance(image_data, str) else None
                ))
            else:
                candidates.append(JoinCandidate(row_id=i))

        results, _ = hvfp.execute(predicate, candidates)
        passing_ids = {c.row_id for c in results}

        return pd.Series([i in passing_ids for i in range(len(images))])

    return hvfp_filter_udf


def create_hvfp_join_udf_with_score(
    config: HVFPConfig = None,
    predicate: str = None
):
    """
    Create a Spark UDF that returns similarity scores instead of boolean.

    Useful for ranking or soft filtering.
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import FloatType

    hvfp = HVFPJoin(config or HVFPConfig())

    @pandas_udf(FloatType())
    def hvfp_score_udf(images: pd.Series) -> pd.Series:
        """Return Phase 1 similarity scores for images."""
        scores = []

        text_embedding = hvfp.phase1.encode_text(predicate)

        for image_data in images:
            try:
                if image_data is not None:
                    image_embedding = hvfp.phase1.encode_image(image_data)
                    score = hvfp.phase1.compute_similarity(text_embedding, image_embedding)
                else:
                    score = 0.0
            except Exception:
                score = 0.0
            scores.append(score)

        return pd.Series(scores)

    return hvfp_score_udf


# ============================================================================
# High-Level API for Multimodal Join
# ============================================================================

def hvfp_multimodal_join(
    left_df,
    right_df,
    predicate: str,
    left_image_col: str,
    right_text_col: str = None,
    config: HVFPConfig = None,
    spark: 'SparkSession' = None
):
    """
    Perform a multimodal join using HVFP-Join optimization.

    This is a high-level API for joining a table with images against
    a predicate or another table.

    Args:
        left_df: DataFrame with images
        right_df: Optional DataFrame to join against
        predicate: Join predicate (e.g., "luxury outdoor living")
        left_image_col: Column name containing images in left_df
        right_text_col: Optional column in right_df containing text predicates
        config: HVFP configuration
        spark: SparkSession

    Returns:
        Filtered DataFrame with matching rows

    Example:
        result = hvfp_multimodal_join(
            listings_df,
            None,
            predicate="Find listings with swimming pool and fire pit",
            left_image_col="image",
            config=HVFPConfig(similarity_threshold=0.3)
        )
    """
    config = config or HVFPConfig()

    # Register UDFs
    filter_udf = create_hvfp_join_udf(config, predicate)

    if spark:
        spark.udf.register("HVFP_FILTER", filter_udf)

        # Use SQL for filtering
        left_df.createOrReplaceTempView("_hvfp_left")

        result = spark.sql(f"""
            SELECT * FROM _hvfp_left
            WHERE HVFP_FILTER({left_image_col}) = true
        """)

        return result
    else:
        # Direct pandas operation
        from pyspark.sql.functions import col

        return left_df.filter(filter_udf(col(left_image_col)))
