"""
Demo: Hierarchical Visual-Feature Proxy Join (HVFP-Join)

This script demonstrates the HVFP-Join operator on the Real Estate dataset
from HuggingFace (Binaryy/multimodal-real-estate-search).

Sample Queries:
1. Visual Amenity Join - Object Detection Focus
2. Material Verification Join - Texture/Attribute Focus
3. State/Condition Join - OCR + Defect Focus

Usage:
    python demo_hvfp_join.py --query 1 --limit 100
    python demo_hvfp_join.py --query all --limit 500
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, udf, lit
from pyspark.sql.types import (
    StringType, BooleanType, FloatType, StructType,
    StructField, ArrayType, IntegerType
)

from util.hvfp_join import (
    HVFPJoin, HVFPConfig, JoinCandidate, JoinStatistics,
    VectorSpacePruner, SymbolicFeatureProxy, SemanticVerifier,
    PredicateDecomposer, create_hvfp_join_udf
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DemoConfig:
    """Configuration for the demo."""
    # Dataset
    dataset_name: str = "Binaryy/multimodal-real-estate-search"
    data_limit: int = 100

    # HVFP-Join settings
    similarity_threshold: float = 0.25
    detector_confidence: float = 0.3
    enable_object_detection: bool = True
    enable_ocr: bool = True

    # VLM settings
    vlm_api_url: str = "http://localhost:8000/v1"
    vlm_model: str = "llava-hf/llava-1.5-7b-hf"

    # Output
    output_dir: str = "./hvfp_results"
    verbose: bool = True


# ============================================================================
# Sample Queries for Real Estate Dataset
# ============================================================================

SAMPLE_QUERIES = {
    "query1": {
        "name": "Visual Amenity Join (Object Detection Focus)",
        "description": """
        Find listings where the text mentions 'luxury outdoor living'
        AND the image contains a Swimming Pool and a Fire Pit.

        Optimization Path:
        - Phase 1: Filter broadly for "outdoors"
        - Phase 2 (Proxy): Use YOLO-World to detect class="swimming pool"
          and class="fire pit". This filters out 90% of listings that just have a patio.
        - Phase 3 (VLM): Verify if the setting is "luxury" (subjective).
        """,
        "predicate": "luxury outdoor living with swimming pool and fire pit",
        "text_filter": "luxury outdoor living",
        "required_objects": ["swimming_pool", "fire_pit", "pool"],
        "phase2_mode": "any"
    },

    "query2": {
        "name": "Material Verification Join (Texture/Attribute Focus)",
        "description": """
        Identify listings described as 'chef's kitchen' that actually have
        Stainless Steel Appliances and Marble Countertops in the photos.

        Optimization Path:
        - Phase 2 (Proxy): Use a specialized classifier or cheap VLM
          to detect "stainless steel" vs. "white" appliances.
        - Phase 3 (VLM): Confirm the overall "chef's kitchen" layout/vibe.
        """,
        "predicate": "chef's kitchen with stainless steel appliances and marble countertops",
        "text_filter": "chef's kitchen",
        "required_objects": ["stainless_steel_appliance", "kitchen", "refrigerator", "oven"],
        "phase2_mode": "any"
    },

    "query3": {
        "name": "State/Condition Join (OCR + Defect Focus)",
        "description": """
        Find listings with 'Recent Price Drop' (detect text overlay on image)
        AND the house exterior looks 'Dilapidated' or 'Needs Work'.

        Optimization Path:
        - Phase 2 (Proxy): Run OCR to find "Price Drop" or "$$$ Off"
          text overlaid on the image.
        - Phase 3 (VLM): Assess the structural condition ("dilapidated")
          which is hard for simple detectors.
        """,
        "predicate": "property with price drop text overlay that needs renovation or repair",
        "text_filter": None,  # No text filter, relies on OCR
        "text_patterns": ["price drop", "reduced", "$ off", "sale", "discount"],
        "required_objects": [],
        "phase2_mode": "any"
    }
}


# ============================================================================
# Data Loading
# ============================================================================

def load_real_estate_dataset(
    config: DemoConfig,
    spark: SparkSession
) -> 'DataFrame':
    """
    Load the Real Estate dataset from HuggingFace.

    The dataset contains:
    - id: Listing ID
    - image: Image data
    - description: Text description of the listing
    - price: Listing price
    - bedrooms, bathrooms: Property features
    - location: Property location
    """
    try:
        from datasets import load_dataset

        print(f"Loading dataset: {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, split="train")

        # Convert to pandas and limit
        if config.data_limit:
            dataset = dataset.select(range(min(config.data_limit, len(dataset))))

        df = dataset.to_pandas()
        print(f"Loaded {len(df)} records")

        # Create Spark DataFrame
        spark_df = spark.createDataFrame(df)
        return spark_df

    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data(config, spark)


def generate_synthetic_data(
    config: DemoConfig,
    spark: SparkSession
) -> 'DataFrame':
    """
    Generate synthetic real estate data for demonstration.
    """
    np.random.seed(42)
    n_records = config.data_limit

    # Sample descriptions
    descriptions = [
        "Beautiful luxury home with outdoor living space featuring a stunning swimming pool and cozy fire pit area.",
        "Modern chef's kitchen with stainless steel appliances and elegant marble countertops.",
        "Spacious family home with updated appliances and granite countertops.",
        "Charming fixer-upper with great potential. Price recently reduced!",
        "Stunning property with pool and outdoor entertainment area.",
        "Contemporary kitchen featuring top-of-the-line appliances.",
        "Cozy home with beautiful garden and patio space.",
        "Renovated property with modern finishes throughout.",
        "Investment opportunity - needs some TLC. Major price drop!",
        "Elegant estate with resort-style backyard and pool.",
    ]

    # Generate records
    data = []
    for i in range(n_records):
        record = {
            "id": f"listing_{i:04d}",
            "description": np.random.choice(descriptions),
            "price": np.random.randint(200000, 2000000),
            "bedrooms": np.random.randint(2, 6),
            "bathrooms": np.random.randint(1, 4),
            "sqft": np.random.randint(1000, 5000),
            "location": np.random.choice([
                "Los Angeles, CA", "San Francisco, CA", "Seattle, WA",
                "Austin, TX", "Denver, CO", "Miami, FL"
            ]),
            "image_path": f"./sample_images/listing_{i:04d}.jpg"
        }
        data.append(record)

    df = pd.DataFrame(data)
    spark_df = spark.createDataFrame(df)

    print(f"Generated {n_records} synthetic records")
    return spark_df


# ============================================================================
# HVFP-Join Execution
# ============================================================================

def run_hvfp_join_query(
    query_config: Dict,
    df: 'DataFrame',
    hvfp_config: HVFPConfig,
    spark: SparkSession,
    verbose: bool = True
) -> Tuple['DataFrame', JoinStatistics]:
    """
    Execute an HVFP-Join query on the dataframe.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query_config['name']}")
    print(f"{'='*60}")
    print(f"Predicate: {query_config['predicate']}")
    print(f"Description: {query_config['description'][:200]}...")
    print()

    # Initialize HVFP-Join operator
    hvfp = HVFPJoin(hvfp_config)

    # Collect data for processing (in real scenario, this would be distributed)
    start_time = time.time()

    # Get image data and create candidates
    rows = df.collect()
    candidates = []

    for i, row in enumerate(rows):
        # Handle different image column names
        image_data = None
        if hasattr(row, 'image'):
            image_data = row.image
        elif hasattr(row, 'image_path'):
            image_data = row.image_path

        candidate = JoinCandidate(
            row_id=i,
            image_data=image_data if isinstance(image_data, bytes) else None,
            image_path=image_data if isinstance(image_data, str) else None,
            text_data={
                'description': getattr(row, 'description', ''),
                'price': getattr(row, 'price', 0),
                'id': getattr(row, 'id', str(i))
            }
        )
        candidates.append(candidate)

    # Apply text filter if specified
    if query_config.get('text_filter'):
        text_filter = query_config['text_filter'].lower()
        candidates = [
            c for c in candidates
            if text_filter in c.text_data.get('description', '').lower()
        ]
        if verbose:
            print(f"After text filter: {len(candidates)} candidates")

    # Execute HVFP-Join
    results, stats = hvfp.execute(
        predicate=query_config['predicate'],
        candidates=candidates
    )

    total_time = time.time() - start_time

    # Print results
    if verbose:
        print(f"\n--- Results ---")
        print(f"Matching listings: {len(results)}")

        for result in results[:5]:  # Show first 5
            print(f"\n  ID: {result.text_data.get('id', 'N/A')}")
            print(f"  Description: {result.text_data.get('description', 'N/A')[:100]}...")
            print(f"  Phase 1 Score: {result.phase1_score:.4f}")
            print(f"  Detected Objects: {result.phase2_detected_objects}")
            print(f"  Detected Text: {result.phase2_detected_text}")
            print(f"  VLM Response: {result.phase3_vlm_response}")

    # Print statistics
    print(f"\n--- Statistics ---")
    print(f"Total candidates: {stats.total_candidates}")
    print(f"Phase 1 passed: {stats.phase1_passed} (selectivity: {stats.phase1_selectivity:.2%})")
    print(f"Phase 2 passed: {stats.phase2_passed} (selectivity: {stats.phase2_selectivity:.2%})")
    print(f"Phase 3 passed: {stats.phase3_passed}")
    print(f"Overall selectivity: {stats.overall_selectivity:.2%}")
    print(f"Estimated cost reduction: {stats.cost_reduction:.2%}")
    print(f"Total time: {total_time:.2f}s")

    # Create result DataFrame
    result_ids = [r.row_id for r in results]
    result_df = df.filter(col("id").isin([
        candidates[i].text_data.get('id') for i in result_ids
    ]))

    return result_df, stats


def run_baseline_vlm_only(
    query_config: Dict,
    df: 'DataFrame',
    vlm_config: HVFPConfig,
    verbose: bool = True
) -> Tuple[int, float]:
    """
    Run baseline VLM-only approach for comparison.
    This calls VLM on every candidate without pre-filtering.
    """
    print(f"\n{'='*60}")
    print(f"BASELINE (VLM-Only): {query_config['name']}")
    print(f"{'='*60}")

    # In practice, this would call VLM on every row
    # We simulate the cost for demonstration

    n_candidates = df.count()
    vlm_cost_per_call = 1.0  # Normalized cost

    # Simulate baseline execution
    baseline_cost = n_candidates * vlm_cost_per_call
    baseline_time = n_candidates * 0.5  # Estimated 0.5s per VLM call

    if verbose:
        print(f"Total candidates: {n_candidates}")
        print(f"VLM calls required: {n_candidates}")
        print(f"Estimated cost: {baseline_cost:.2f}")
        print(f"Estimated time: {baseline_time:.2f}s")

    return n_candidates, baseline_cost


# ============================================================================
# SQL Interface Demo
# ============================================================================

def demo_sql_interface(
    spark: SparkSession,
    df: 'DataFrame',
    config: DemoConfig
):
    """
    Demonstrate HVFP-Join through Spark SQL interface.
    """
    print(f"\n{'='*60}")
    print("SQL Interface Demo")
    print(f"{'='*60}")

    # Create temp view
    df.createOrReplaceTempView("listings")

    # Create HVFP configuration
    hvfp_config = HVFPConfig(
        similarity_threshold=config.similarity_threshold,
        vlm_api_url=config.vlm_api_url,
        vlm_model=config.vlm_model
    )

    # Register UDF for filtering
    filter_udf = create_hvfp_join_udf(
        config=hvfp_config,
        predicate="luxury outdoor living with swimming pool"
    )
    spark.udf.register("HVFP_FILTER", filter_udf)

    # Example SQL queries
    sql_queries = [
        """
        -- Query 1: Filter using HVFP-Join
        SELECT id, description, price
        FROM listings
        WHERE HVFP_FILTER(image_path) = true
        LIMIT 10
        """,

        """
        -- Query 2: Combine HVFP-Join with text filters
        SELECT id, description, price
        FROM listings
        WHERE LOWER(description) LIKE '%pool%'
          AND HVFP_FILTER(image_path) = true
        ORDER BY price DESC
        LIMIT 10
        """,

        """
        -- Query 3: Aggregation with HVFP-Join
        SELECT location, COUNT(*) as luxury_outdoor_count, AVG(price) as avg_price
        FROM listings
        WHERE HVFP_FILTER(image_path) = true
        GROUP BY location
        ORDER BY luxury_outdoor_count DESC
        """
    ]

    for i, sql in enumerate(sql_queries, 1):
        print(f"\n--- SQL Query {i} ---")
        print(sql.strip())
        print()

        try:
            result = spark.sql(sql)
            result.show(truncate=50)
        except Exception as e:
            print(f"Query execution note: {e}")
            print("(This is expected if image data is not available)")


# ============================================================================
# Cost Model Validation
# ============================================================================

def validate_cost_model(
    stats_list: List[JoinStatistics],
    baseline_costs: List[float]
):
    """
    Validate the HVFP-Join cost model against baseline.

    Cost Model:
        Standard: Cost_std = N * C_vlm
        HVFP-Join: Cost_ours = N * C_emb + (N * s1) * C_det + (N * s1 * s2) * C_vlm
    """
    print(f"\n{'='*60}")
    print("Cost Model Validation")
    print(f"{'='*60}")

    # Cost assumptions (normalized)
    C_vlm = 1.0       # VLM cost (baseline)
    C_emb = 0.001     # Embedding cost
    C_det = 0.01      # Detection cost

    for i, (stats, baseline_cost) in enumerate(zip(stats_list, baseline_costs), 1):
        print(f"\n--- Query {i} ---")

        N = stats.total_candidates
        s1 = stats.phase1_selectivity
        s2 = stats.phase2_selectivity

        # Calculate costs
        standard_cost = N * C_vlm

        hvfp_cost = (
            N * C_emb +                    # Phase 1: All candidates
            (N * s1) * C_det +             # Phase 2: After vector pruning
            (N * s1 * s2) * C_vlm          # Phase 3: After symbolic proxy
        )

        reduction = 1 - (hvfp_cost / standard_cost)

        print(f"N (total candidates): {N}")
        print(f"s1 (Phase 1 selectivity): {s1:.2%}")
        print(f"s2 (Phase 2 selectivity): {s2:.2%}")
        print()
        print(f"Standard cost (N * C_vlm): {standard_cost:.2f}")
        print(f"HVFP-Join cost: {hvfp_cost:.4f}")
        print(f"  - Phase 1 (embedding): {N * C_emb:.4f}")
        print(f"  - Phase 2 (detection): {(N * s1) * C_det:.4f}")
        print(f"  - Phase 3 (VLM): {(N * s1 * s2) * C_vlm:.4f}")
        print()
        print(f"Cost reduction: {reduction:.2%}")
        print(f"Speedup factor: {standard_cost / hvfp_cost:.1f}x")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Hierarchical Visual-Feature Proxy Join (HVFP-Join)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Which query to run (1, 2, 3, or all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of records to process"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Similarity threshold for Phase 1"
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLM API URL"
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip Phase 3 VLM verification"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    args = parser.parse_args()

    # Create configuration
    demo_config = DemoConfig(
        data_limit=args.limit,
        similarity_threshold=args.threshold,
        vlm_api_url=args.vlm_url,
        verbose=args.verbose
    )

    # Create HVFP configuration
    hvfp_config = HVFPConfig(
        similarity_threshold=args.threshold,
        vlm_api_url=args.vlm_url,
        detector_confidence_threshold=0.3,
        enable_object_detection=True,
        enable_ocr=True
    )

    # Initialize Spark
    print("Initializing Spark...")
    spark = SparkSession.builder \
        .appName("HVFP-Join Demo") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    try:
        # Load data
        df = load_real_estate_dataset(demo_config, spark)

        # Determine which queries to run
        if args.query == "all":
            queries_to_run = ["query1", "query2", "query3"]
        else:
            queries_to_run = [f"query{args.query}"]

        # Run queries
        all_stats = []
        baseline_costs = []

        for query_key in queries_to_run:
            query_config = SAMPLE_QUERIES[query_key]

            # Run baseline for comparison
            n_candidates, baseline_cost = run_baseline_vlm_only(
                query_config, df, hvfp_config, demo_config.verbose
            )
            baseline_costs.append(baseline_cost)

            # Run HVFP-Join
            result_df, stats = run_hvfp_join_query(
                query_config,
                df,
                hvfp_config,
                spark,
                demo_config.verbose
            )
            all_stats.append(stats)

            # Save results
            if demo_config.output_dir:
                os.makedirs(demo_config.output_dir, exist_ok=True)
                output_path = os.path.join(
                    demo_config.output_dir,
                    f"hvfp_result_{query_key}.csv"
                )
                result_df.toPandas().to_csv(output_path, index=False)
                print(f"Results saved to: {output_path}")

        # Validate cost model
        validate_cost_model(all_stats, baseline_costs)

        # Demo SQL interface
        print("\n" + "="*60)
        print("SQL Interface Demo (skipped - requires image data)")
        print("="*60)
        # demo_sql_interface(spark, df, demo_config)

    finally:
        spark.stop()
        print("\nDemo complete!")


if __name__ == "__main__":
    main()
