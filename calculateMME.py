#!/usr/bin/env python3
"""
Simple script to calculate MME score from your POPE CSV file.

Usage:
    python calculate_score.py your_results.csv
    
Or for batch processing:
    python calculate_score.py --batch POPE_trim_grouping_*.csv
"""

import pandas as pd
import sys
import glob
from pathlib import Path


def calculate_score(csv_path):
    """
    Calculate MME score from CSV with format:
    question_id,answer,predicted,is_correct
    artwork/10002,Yes,Yes,1
    artwork/10002,No,No,1
    ...
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Extract image ID from question_id (part before any comma)
    df['image_id'] = df['question_id'].astype(str).apply(
        lambda x: x.split(',')[0] if ',' in x else x
    )
    
    # Normalize to lowercase
    df['answer'] = df['answer'].astype(str).str.lower().str.strip()
    df['predicted'] = df['predicted'].astype(str).str.lower().str.strip()
    
    # Group by image
    grouped = df.groupby('image_id')
    
    total_images = len(grouped)
    total_questions = len(df)
    
    # Count correct predictions
    correct_predictions = (df['answer'] == df['predicted']).sum()
    
    # Count images where ALL questions are correct
    images_all_correct = 0
    for image_id, group in grouped:
        if (group['answer'] == group['predicted']).all():
            images_all_correct += 1
    
    # Calculate metrics
    acc = correct_predictions / total_questions
    acc_plus = images_all_correct / total_images
    mme_score = (acc + acc_plus) * 100
    
    # Calculate confusion matrix (for yes/no)
    df_yes_no = df[df['predicted'].isin(['yes', 'no'])].copy()
    
    tp = ((df_yes_no['answer'] == 'yes') & (df_yes_no['predicted'] == 'yes')).sum()
    tn = ((df_yes_no['answer'] == 'no') & (df_yes_no['predicted'] == 'no')).sum()
    fp = ((df_yes_no['answer'] == 'no') & (df_yes_no['predicted'] == 'yes')).sum()
    fn = ((df_yes_no['answer'] == 'yes') & (df_yes_no['predicted'] == 'no')).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'mme_score': mme_score,
        'accuracy': acc * 100,
        'acc_plus': acc_plus * 100,
        'precision': precision,
        'recall': recall,
        'total_images': total_images,
        'total_questions': total_questions,
        'correct_questions': correct_predictions,
        'images_all_correct': images_all_correct,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def print_results(csv_path, metrics):
    """Print formatted results."""
    print("\n" + "="*80)
    print(f"Results for: {Path(csv_path).name}")
    print("="*80)
    
    print(f"\n📊 MME SCORE: {metrics['mme_score']:.2f}")
    print("-"*80)
    
    print(f"\n📈 Accuracy Metrics:")
    print(f"  Regular Accuracy:     {metrics['accuracy']:>8.2f}%")
    print(f"  Pair Accuracy:        {metrics['acc_plus']:>8.2f}%")
    
    print(f"\n📊 Classification Metrics:")
    print(f"  Precision:            {metrics['precision']:>8.4f}")
    print(f"  Recall:               {metrics['recall']:>8.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Positives (TP):  {metrics['tp']:>8}")
    print(f"  False Positives (FP): {metrics['fp']:>8}")
    print(f"  True Negatives (TN):  {metrics['tn']:>8}")
    print(f"  False Negatives (FN): {metrics['fn']:>8}")
    
    print(f"\n📝 Statistics:")
    print(f"  Total Images:         {metrics['total_images']:>8}")
    print(f"  Total Questions:      {metrics['total_questions']:>8}")
    print(f"  Correct Questions:    {metrics['correct_questions']:>8}")
    print(f"  Images All Correct:   {metrics['images_all_correct']:>8}")
    
    print("="*80 + "\n")


def batch_process(pattern):
    """Process multiple CSV files matching a pattern."""
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"\nFound {len(files)} files to process")
    print("="*80 + "\n")
    
    results = []
    
    for csv_file in sorted(files):
        metrics = calculate_score(csv_file)
        
        results.append({
            'File': Path(csv_file).name,
            'MME Score': metrics['mme_score'],
            'Accuracy (%)': metrics['accuracy'],
            'Acc Plus (%)': metrics['acc_plus'],
            'Images': metrics['total_images'],
            'Questions': metrics['total_questions']
        })
        
        print(f"✓ {Path(csv_file).name:40s} MME Score: {metrics['mme_score']:>8.2f}")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    # Create DataFrame for nice printing
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    print("\n" + "="*80)
    
    # Find best
    best_idx = df_results['MME Score'].idxmax()
    best = df_results.iloc[best_idx]
    
    print(f"\n🏆 BEST RESULT:")
    print(f"  File: {best['File']}")
    print(f"  MME Score: {best['MME Score']:.2f}")
    print(f"  Accuracy: {best['Accuracy (%)']:.2f}%")
    print("="*80 + "\n")
    
    # Save results
    output_file = "mme_scores_summary.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}\n")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python calculate_score.py <csv_file>")
        print("  python calculate_score.py --batch <pattern>")
        print("\nExamples:")
        print("  python calculate_score.py results.csv")
        print("  python calculate_score.py --batch 'POPE_*.csv'")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Error: --batch requires a file pattern")
            sys.exit(1)
        batch_process(sys.argv[2])
    else:
        csv_path = sys.argv[1]
        
        if not Path(csv_path).exists():
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        
        metrics = calculate_score(csv_path)
        print_results(csv_path, metrics)


if __name__ == "__main__":
    main()

