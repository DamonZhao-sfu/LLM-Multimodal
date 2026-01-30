"""
LLM-Multimodal Utility Package

This package contains utility modules for multimodal LLM processing with Spark SQL.
"""

from util.hvfp_join import (
    HVFPJoin,
    HVFPConfig,
    JoinCandidate,
    JoinStatistics,
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

__all__ = [
    # HVFP-Join Core
    'HVFPJoin',
    'HVFPConfig',
    'JoinCandidate',
    'JoinStatistics',
    'JoinPhase',

    # Phase Components
    'VectorSpacePruner',
    'SymbolicFeatureProxy',
    'SemanticVerifier',
    'PredicateDecomposer',
    'ObjectDetector',
    'OCREngine',

    # UDF Factories
    'create_hvfp_join_udf',
    'create_hvfp_join_udf_with_score',
    'hvfp_multimodal_join',
]
