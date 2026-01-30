"""
LLM-Multimodal Utility Package

This package contains utility modules for multimodal LLM processing with Spark SQL.

Modules:
- hvfp_join: Hierarchical Visual-Feature Proxy Join (three-phase cascade)
- feature_extraction_join: Feature Extraction Join (image grouping + text-text join)
"""

from util.hvfp_join import (
    HVFPJoin,
    HVFPConfig,
    JoinCandidate,
    JoinStatistics as HVFPStatistics,
    JoinPhase,
    VectorSpacePruner,
    SymbolicFeatureProxy,
    SemanticVerifier,
    PredicateDecomposer,
    ObjectDetector,
    OCREngine,
    create_hvfp_join_udf,
    create_hvfp_join_udf_with_score,
    hvfp_multimodal_join,
)

from util.feature_extraction_join import (
    FeatureExtractionJoin,
    FeatureExtractionConfig,
    JoinResult,
    JoinStatistics as FEJoinStatistics,
    ImageEmbedder,
    ImageGrouper,
    FeatureExtractor,
    TextMatcher,
    create_feature_extraction_udf,
    create_semantic_match_udf,
    create_feature_join_udf,
    feature_extraction_join,
)

__all__ = [
    # HVFP-Join Core
    'HVFPJoin',
    'HVFPConfig',
    'JoinCandidate',
    'HVFPStatistics',
    'JoinPhase',

    # HVFP Phase Components
    'VectorSpacePruner',
    'SymbolicFeatureProxy',
    'SemanticVerifier',
    'PredicateDecomposer',
    'ObjectDetector',
    'OCREngine',

    # HVFP UDF Factories
    'create_hvfp_join_udf',
    'create_hvfp_join_udf_with_score',
    'hvfp_multimodal_join',

    # Feature Extraction Join Core
    'FeatureExtractionJoin',
    'FeatureExtractionConfig',
    'JoinResult',
    'FEJoinStatistics',

    # FE-Join Components
    'ImageEmbedder',
    'ImageGrouper',
    'FeatureExtractor',
    'TextMatcher',

    # FE-Join UDF Factories
    'create_feature_extraction_udf',
    'create_semantic_match_udf',
    'create_feature_join_udf',
    'feature_extraction_join',
]
