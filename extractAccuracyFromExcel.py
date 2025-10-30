import os
import pandas as pd
from pathlib import Path

def calculate_accuracy_from_directories(directories):
    """
    Calculate accuracy from CSV files in multiple directories.
    
    Args:
        directories: List of directory paths containing CSV files
        
    Returns:
        Dictionary with results for each directory
    """
    results = {}
    
    for directory in directories:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {directory}")
            continue
        
        # Find all CSV files in the directory
        csv_files = list(dir_path.glob("*.csv"))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {directory}")
            continue
        
        # Process each CSV file (or just the first one if only one expected)
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if 'is_correct' not in df.columns:
                    print(f"Warning: 'is_correct' column not found in {csv_file}")
                    continue
                
                # Calculate accuracy
                total_count = len(df['is_correct'])
                correct_count = df['is_correct'].sum()
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                results[str(csv_file)] = {
                    'directory': str(directory),
                    'csv_file': csv_file.name,
                    'correct_count': int(correct_count),
                    'total_count': total_count,
                    'accuracy': accuracy,
                    'accuracy_percent': accuracy * 100
                }
                
                print(f"\n{csv_file.name}:")
                print(f"  Correct: {correct_count}/{total_count}")
                print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
    
    return results


def main():
    # Example usage - modify these directories as needed
    directories = [
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQA_image_nonprefix_0.056.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQA_image_nonprefix_0.111.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQA_image_nonprefix_0.222.csv"
    ]
    
    print("="*80)
    print("ACCURACY CALCULATION FROM MULTIPLE DIRECTORIES")
    print("="*80)
    
    results = calculate_accuracy_from_directories(directories)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        for file_path, result in results.items():
            print(f"\n{result['directory']}/{result['csv_file']}:")
            print(f"  Accuracy: {result['correct_count']}/{result['total_count']} = {result['accuracy']:.4f} ({result['accuracy_percent']:.2f}%)")
        
        # Calculate overall accuracy if needed
        total_correct = sum(r['correct_count'] for r in results.values())
        total_all = sum(r['total_count'] for r in results.values())
        overall_accuracy = total_correct / total_all if total_all > 0 else 0
        
        print("\n" + "-"*80)
        print(f"OVERALL ACCURACY: {total_correct}/{total_all} = {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
        print("="*80)
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
