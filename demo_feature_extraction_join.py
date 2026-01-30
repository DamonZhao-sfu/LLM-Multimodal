"""
Demo: Feature Extraction Join (FE-Join)

This script demonstrates the Feature Extraction Join operator that:
1. Groups similar images together (reduces VLM calls)
2. Extracts text descriptions from representative images
3. Performs semantic text-text join with right table

Example Scenario:
- Left Table (products_images): product_id, image
- Right Table (product_descriptions): desc_id, category, description
- Join: Match product images with their textual descriptions

Usage:
    python demo_feature_extraction_join.py --num-groups 20 --threshold 0.5
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType

from util.feature_extraction_join import (
    FeatureExtractionJoin,
    FeatureExtractionConfig,
    JoinResult,
    JoinStatistics,
    create_feature_extraction_udf,
    create_semantic_match_udf,
    create_feature_join_udf,
    feature_extraction_join,
)


# ============================================================================
# Sample Data Generation (Generic - No dataset assumption)
# ============================================================================

def generate_sample_left_table(spark: SparkSession, n_images: int = 100):
    """
    Generate sample left table with images.

    In real usage, this would be your actual image table.
    Schema: (id, image_path, category)
    """
    np.random.seed(42)

    # Sample categories - these would have similar visual features
    categories = ["electronics", "furniture", "clothing", "food", "outdoor"]

    data = []
    for i in range(n_images):
        category = np.random.choice(categories)
        data.append({
            "id": f"img_{i:04d}",
            "image_path": f"./sample_images/{category}/image_{i:04d}.jpg",
            "category": category
        })

    pdf = pd.DataFrame(data)
    return spark.createDataFrame(pdf)


def generate_sample_right_table(spark: SparkSession, n_descriptions: int = 50):
    """
    Generate sample right table with text descriptions.

    In real usage, this would be your actual description table.
    Schema: (id, text, category)
    """
    np.random.seed(123)

    # Sample descriptions matching the categories
    descriptions_by_category = {
        "electronics": [
            "High-performance laptop with sleek aluminum design and backlit keyboard",
            "Wireless earbuds with noise cancellation and charging case",
            "Smart watch with heart rate monitor and GPS tracking",
            "4K television with HDR support and smart TV features",
            "Portable bluetooth speaker with waterproof design",
        ],
        "furniture": [
            "Modern wooden dining table with minimalist design",
            "Comfortable leather sofa with adjustable headrests",
            "Ergonomic office chair with lumbar support",
            "Vintage oak bookshelf with five tiers",
            "Contemporary glass coffee table with metal legs",
        ],
        "clothing": [
            "Classic cotton t-shirt in solid color",
            "Slim fit denim jeans with stretch fabric",
            "Wool blend winter coat with hood",
            "Athletic running shoes with mesh upper",
            "Silk evening dress with elegant design",
        ],
        "food": [
            "Organic mixed berry smoothie with protein",
            "Artisanal sourdough bread freshly baked",
            "Gourmet chocolate truffles assortment",
            "Farm fresh vegetable salad with dressing",
            "Premium coffee beans dark roasted",
        ],
        "outdoor": [
            "Camping tent for four people waterproof",
            "Hiking backpack with multiple compartments",
            "Stainless steel water bottle insulated",
            "Folding camping chair with cup holder",
            "LED flashlight with rechargeable battery",
        ],
    }

    data = []
    for i in range(n_descriptions):
        category = np.random.choice(list(descriptions_by_category.keys()))
        text = np.random.choice(descriptions_by_category[category])
        data.append({
            "desc_id": f"desc_{i:04d}",
            "text": text,
            "category": category
        })

    pdf = pd.DataFrame(data)
    return spark.createDataFrame(pdf)


# ============================================================================
# Demo: Feature Extraction Join with SQL
# ============================================================================

def demo_sql_join(
    spark: SparkSession,
    left_df,
    right_df,
    config: FeatureExtractionConfig
):
    """
    Demonstrate Feature Extraction Join using SQL interface.

    This shows how to use UDFs in Spark SQL for multimodal joins.
    """
    print("\n" + "="*70)
    print("Demo: Feature Extraction Join with SQL")
    print("="*70)

    # Register UDFs
    extract_udf = create_feature_extraction_udf(config)
    match_udf = create_semantic_match_udf(config)
    join_udf = create_feature_join_udf(config, threshold=0.5)

    spark.udf.register("EXTRACT_FEATURES", extract_udf)
    spark.udf.register("SEMANTIC_MATCH", match_udf)
    spark.udf.register("FEATURE_JOIN", join_udf)

    # Create temp views
    left_df.createOrReplaceTempView("product_images")
    right_df.createOrReplaceTempView("product_descriptions")

    # Query 1: Extract features and show
    print("\n--- Query 1: Extract Features from Images ---")
    query1 = """
        SELECT
            id,
            category,
            EXTRACT_FEATURES(image_path) as extracted_description
        FROM product_images
        LIMIT 5
    """
    print(f"SQL:\n{query1}")

    # Query 2: Join with extracted features
    print("\n--- Query 2: Feature Extraction Join ---")
    query2 = """
        -- Step 1: Extract features from images (with grouping optimization)
        WITH images_with_features AS (
            SELECT
                id as image_id,
                category as image_category,
                image_path,
                EXTRACT_FEATURES(image_path) as extracted_features
            FROM product_images
        )
        -- Step 2: Join with descriptions using semantic matching
        SELECT
            i.image_id,
            i.image_category,
            d.desc_id,
            d.category as desc_category,
            d.text as description,
            i.extracted_features,
            SEMANTIC_MATCH(i.extracted_features, d.text) as similarity_score
        FROM images_with_features i
        CROSS JOIN product_descriptions d
        WHERE SEMANTIC_MATCH(i.extracted_features, d.text) > 0.5
        ORDER BY similarity_score DESC
        LIMIT 20
    """
    print(f"SQL:\n{query2}")

    # Query 3: Using direct join predicate
    print("\n--- Query 3: Direct Feature Join Predicate ---")
    query3 = """
        -- Direct multimodal join using FEATURE_JOIN UDF
        SELECT
            l.id as image_id,
            l.category as image_category,
            r.desc_id,
            r.category as desc_category,
            r.text as matched_description
        FROM product_images l, product_descriptions d
        WHERE FEATURE_JOIN(l.image_path, r.text) = true
          AND l.category = r.category  -- Optional: additional filter
        ORDER BY l.id
    """
    print(f"SQL:\n{query3}")

    # Query 4: Aggregation with join
    print("\n--- Query 4: Aggregation after Join ---")
    query4 = """
        -- Count matches per category
        WITH joined AS (
            SELECT
                l.category,
                EXTRACT_FEATURES(l.image_path) as features,
                r.text
            FROM product_images l
            CROSS JOIN product_descriptions r
            WHERE SEMANTIC_MATCH(EXTRACT_FEATURES(l.image_path), r.text) > 0.5
        )
        SELECT
            category,
            COUNT(*) as num_matches,
            AVG(length(features)) as avg_feature_length
        FROM joined
        GROUP BY category
        ORDER BY num_matches DESC
    """
    print(f"SQL:\n{query4}")

    return True


# ============================================================================
# Demo: Programmatic Join
# ============================================================================

def demo_programmatic_join(
    left_images: List[str],
    left_ids: List[str],
    right_texts: List[str],
    right_ids: List[str],
    config: FeatureExtractionConfig
):
    """
    Demonstrate Feature Extraction Join using Python API.

    This shows the full join workflow:
    1. Image grouping (similarity-based clustering)
    2. Feature extraction (VLM on group representatives)
    3. Semantic matching (text-text similarity)
    """
    print("\n" + "="*70)
    print("Demo: Feature Extraction Join (Programmatic)")
    print("="*70)

    # Initialize join operator
    fe_join = FeatureExtractionJoin(config)

    # Execute join
    print(f"\nInput:")
    print(f"  Left table: {len(left_images)} images")
    print(f"  Right table: {len(right_texts)} text descriptions")
    print(f"  Target groups: {config.num_groups}")
    print(f"  Similarity threshold: {config.text_similarity_threshold}")

    # Custom extraction prompt (join-aware)
    extraction_prompt = """
    Describe this image focusing on:
    1. Main object/product type
    2. Key visual features (color, material, shape)
    3. Style or category
    4. Any text or branding visible

    Be concise but specific.
    """

    results, stats = fe_join.execute(
        left_images=left_images,
        left_ids=left_ids,
        right_texts=right_texts,
        right_ids=right_ids,
        extraction_prompt=extraction_prompt
    )

    # Display results
    print(f"\n--- Results ---")
    print(f"Total matched pairs: {len(results)}")

    for i, result in enumerate(results[:10]):  # Show first 10
        print(f"\n  Match {i+1}:")
        print(f"    Image ID: {result.left_id}")
        print(f"    Text ID: {result.right_id}")
        print(f"    Group ID: {result.group_id}")
        print(f"    Similarity: {result.similarity_score:.4f}")
        print(f"    Extracted: {result.extracted_description[:100]}...")
        print(f"    Right Text: {result.right_text[:100]}...")

    # Display statistics
    print(f"\n--- Statistics ---")
    print(f"Image groups created: {stats.num_image_groups}")
    print(f"VLM calls made: {stats.vlm_calls}")
    print(f"VLM calls saved: {stats.total_left_rows - stats.vlm_calls}")
    print(f"Cost savings: {stats.vlm_cost_saved:.1%}")
    print(f"\nTiming breakdown:")
    print(f"  Embedding: {stats.embedding_time:.2f}s")
    print(f"  Grouping: {stats.grouping_time:.2f}s")
    print(f"  Extraction: {stats.extraction_time:.2f}s")
    print(f"  Joining: {stats.join_time:.2f}s")
    print(f"  Total: {stats.total_time:.2f}s")

    return results, stats


# ============================================================================
# Demo: Cost Model Analysis
# ============================================================================

def analyze_cost_model(stats: JoinStatistics):
    """
    Analyze the cost model and compare with naive approach.

    Naive approach: Call VLM on every image-text pair
    Our approach: Group images, call VLM per group, do text-text matching
    """
    print("\n" + "="*70)
    print("Cost Model Analysis")
    print("="*70)

    # Cost assumptions (normalized to VLM call = 1.0)
    C_embed = 0.001    # Image embedding cost
    C_group = 0.0001   # Grouping cost per image
    C_vlm = 1.0        # VLM feature extraction cost
    C_text_embed = 0.01  # Text embedding cost
    C_match = 0.0001   # Text matching cost

    N_left = stats.total_left_rows
    N_right = stats.total_right_rows
    N_groups = stats.num_image_groups

    # Naive approach: VLM on every pair
    naive_cost = N_left * N_right * C_vlm
    naive_calls = N_left * N_right

    # Alternative naive: VLM per image, then text matching
    alt_naive_cost = N_left * C_vlm + N_left * N_right * C_match
    alt_naive_calls = N_left

    # Our approach
    our_cost = (
        N_left * C_embed +                    # Embed all images
        N_left * C_group +                    # Group images
        N_groups * C_vlm +                    # VLM per group
        N_groups * C_text_embed +             # Embed group descriptions
        N_right * C_text_embed +              # Embed right texts
        N_groups * N_right * C_match          # Text matching
    )
    our_calls = N_groups

    print(f"\nInput sizes:")
    print(f"  Left table (images): {N_left}")
    print(f"  Right table (texts): {N_right}")
    print(f"  Image groups: {N_groups}")

    print(f"\n--- Naive Approach (VLM per pair) ---")
    print(f"  VLM calls: {naive_calls:,}")
    print(f"  Cost: {naive_cost:,.2f}")

    print(f"\n--- Alternative Naive (VLM per image) ---")
    print(f"  VLM calls: {alt_naive_calls:,}")
    print(f"  Cost: {alt_naive_cost:,.2f}")

    print(f"\n--- Feature Extraction Join (Grouped) ---")
    print(f"  VLM calls: {our_calls}")
    print(f"  Cost: {our_cost:.2f}")
    print(f"  Breakdown:")
    print(f"    - Image embedding: {N_left * C_embed:.4f}")
    print(f"    - Grouping: {N_left * C_group:.4f}")
    print(f"    - VLM extraction: {N_groups * C_vlm:.2f}")
    print(f"    - Text embedding: {(N_groups + N_right) * C_text_embed:.4f}")
    print(f"    - Matching: {N_groups * N_right * C_match:.4f}")

    print(f"\n--- Savings ---")
    print(f"  vs Naive (per pair): {(1 - our_cost/naive_cost)*100:.2f}%")
    print(f"  vs Alt Naive (per image): {(1 - our_cost/alt_naive_cost)*100:.2f}%")
    print(f"  VLM call reduction: {(1 - our_calls/alt_naive_calls)*100:.2f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Feature Extraction Join"
    )
    parser.add_argument(
        "--num-images", type=int, default=100,
        help="Number of images in left table"
    )
    parser.add_argument(
        "--num-texts", type=int, default=50,
        help="Number of texts in right table"
    )
    parser.add_argument(
        "--num-groups", type=int, default=20,
        help="Target number of image groups"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Similarity threshold for matching"
    )
    parser.add_argument(
        "--grouping-method", type=str, default="kmeans",
        choices=["kmeans", "agglomerative", "threshold"],
        help="Image grouping method"
    )
    parser.add_argument(
        "--vlm-url", type=str, default="http://localhost:8000/v1",
        help="VLM API URL"
    )
    parser.add_argument(
        "--demo", type=str, default="all",
        choices=["sql", "programmatic", "cost", "all"],
        help="Which demo to run"
    )

    args = parser.parse_args()

    # Configuration
    config = FeatureExtractionConfig(
        num_groups=args.num_groups,
        text_similarity_threshold=args.threshold,
        grouping_method=args.grouping_method,
        vlm_api_url=args.vlm_url,
        enable_grouping=True
    )

    print("="*70)
    print("Feature Extraction Join Demo")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Images (left): {args.num_images}")
    print(f"  Texts (right): {args.num_texts}")
    print(f"  Target groups: {args.num_groups}")
    print(f"  Grouping method: {args.grouping_method}")
    print(f"  Match threshold: {args.threshold}")

    # Initialize Spark
    print("\nInitializing Spark...")
    spark = SparkSession.builder \
        .appName("Feature Extraction Join Demo") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    try:
        # Generate sample data
        print("\nGenerating sample data...")
        left_df = generate_sample_left_table(spark, args.num_images)
        right_df = generate_sample_right_table(spark, args.num_texts)

        print(f"\nLeft table schema:")
        left_df.printSchema()
        print(f"\nRight table schema:")
        right_df.printSchema()

        # Show sample data
        print("\nSample left table (images):")
        left_df.show(5, truncate=False)

        print("\nSample right table (texts):")
        right_df.show(5, truncate=False)

        # Run demos
        if args.demo in ["sql", "all"]:
            demo_sql_join(spark, left_df, right_df, config)

        if args.demo in ["programmatic", "all"]:
            # Collect data for programmatic demo
            left_rows = left_df.collect()
            right_rows = right_df.collect()

            left_images = [row['image_path'] for row in left_rows]
            left_ids = [row['id'] for row in left_rows]
            right_texts = [row['text'] for row in right_rows]
            right_ids = [row['desc_id'] for row in right_rows]

            results, stats = demo_programmatic_join(
                left_images, left_ids,
                right_texts, right_ids,
                config
            )

            if args.demo in ["cost", "all"]:
                analyze_cost_model(stats)

    finally:
        spark.stop()
        print("\nDemo complete!")


if __name__ == "__main__":
    main()
