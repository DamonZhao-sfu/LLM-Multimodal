import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_cdf(data, label, ax, color=None, linestyle='-'):
    """Plot CDF for given data."""
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, y, label=label, linewidth=2, color=color, linestyle=linestyle)

def create_cdf_plots(csv_files, redundancy_files, dataset_names, colors):
    """Create CDF plots from multiple token counts CSV files and redundancy files."""
    
    # Read all CSV files
    dfs = {}
    redundancy_dfs = {}
    
    for name, file in zip(dataset_names, csv_files):
        dfs[name] = pd.read_csv(file)
        print(f"Loaded {name}: {len(dfs[name])} rows")
    
    for name, file in zip(dataset_names, redundancy_files):
        redundancy_dfs[name] = pd.read_csv(file)
        print(f"Loaded {name} redundancy: {len(redundancy_dfs[name])} unique images")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ===== Subplot 1: CDF of total_pixels =====
    ax1 = axes[0]
    for i, name in enumerate(dataset_names):
        plot_cdf(dfs[name]['total_pixels'], name, ax1, color=colors[i])
    
    ax1.set_xlabel('Total Pixels', fontsize=14)
    ax1.set_ylabel('CDF', fontsize=14)
    ax1.set_title('CDF Distribution of Total Pixels', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')
    
    # ===== Subplot 2: CDF of total_tokens =====
    ax2 = axes[1]
    for i, name in enumerate(dataset_names):
        plot_cdf(dfs[name]['total_tokens'], name, ax2, color=colors[i])
    
    ax2.set_xlabel('Total Tokens', fontsize=14)
    ax2.set_ylabel('CDF', fontsize=14)
    ax2.set_title('CDF Distribution of Total Tokens', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    # ===== Subplot 3: CDF of image redundancy =====
    ax3 = axes[2]
    for i, name in enumerate(dataset_names):
        plot_cdf(redundancy_dfs[name]['redundancy'], name, ax3, color=colors[i])
    
    ax3.set_xlabel('Image Redundancy (# of occurrences)', fontsize=14)
    ax3.set_ylabel('CDF', fontsize=14)
    ax3.set_title('CDF Distribution of Image Redundancy', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('all_datasets_cdf_with_redundancy.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'all_datasets_cdf_with_redundancy.png'")
    
    # Show the plot
    plt.show()
    
    # Print statistics for all datasets
    print("\n" + "="*80)
    print("STATISTICS FOR ALL DATASETS")
    print("="*80)
    
    for name in dataset_names:
        df = dfs[name]
        red_df = redundancy_dfs[name]
        
        print(f"\n{'='*80}")
        print(f"Dataset: {name}")
        print(f"{'='*80}")
        
        print(f"\nTotal Pixels:")
        print(f"  Min: {df['total_pixels'].min():,}")
        print(f"  Max: {df['total_pixels'].max():,}")
        print(f"  Mean: {df['total_pixels'].mean():,.2f}")
        print(f"  Median: {df['total_pixels'].median():,.2f}")
        
        print(f"\nImage Tokens:")
        print(f"  Min: {df['image_tokens'].min()}")
        print(f"  Max: {df['image_tokens'].max()}")
        print(f"  Mean: {df['image_tokens'].mean():.2f}")
        print(f"  Median: {df['image_tokens'].median():.2f}")
        
        print(f"\nText Tokens:")
        print(f"  Min: {df['text_tokens'].min()}")
        print(f"  Max: {df['text_tokens'].max()}")
        print(f"  Mean: {df['text_tokens'].mean():.2f}")
        print(f"  Median: {df['text_tokens'].median():.2f}")
        
        print(f"\nTotal Tokens:")
        print(f"  Min: {df['total_tokens'].min()}")
        print(f"  Max: {df['total_tokens'].max()}")
        print(f"  Mean: {df['total_tokens'].mean():.2f}")
        print(f"  Median: {df['total_tokens'].median():.2f}")
        
        print(f"\nImage Redundancy:")
        print(f"  Unique Images: {len(red_df)}")
        print(f"  Total Rows: {len(df)}")
        print(f"  Min Redundancy: {red_df['redundancy'].min()}")
        print(f"  Max Redundancy: {red_df['redundancy'].max()}")
        print(f"  Mean Redundancy: {red_df['redundancy'].mean():.2f}")
        print(f"  Median Redundancy: {red_df['redundancy'].median():.2f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Dataset':<15} {'Pixels (Mean)':<15} {'Tokens (Mean)':<15} {'Rows':<10} {'Unique Imgs':<12} {'Avg Redundancy':<15}")
    print("-" * 95)
    for name in dataset_names:
        df = dfs[name]
        red_df = redundancy_dfs[name]
        print(f"{name:<15} {df['total_pixels'].mean():>13,.0f} {df['total_tokens'].mean():>13,.2f} {len(df):>10,} {len(red_df):>12,} {red_df['redundancy'].mean():>15,.2f}")

# Usage
if __name__ == "__main__":
    csv_files = [
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQAv2.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQAtext.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/POPE.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/scivqa_token_counts.csv"
    ]
    
    redundancy_files = [
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQAv2_redundancy.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQAtext_redundancy.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/POPE_redundancy.csv",
        #"/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/scivqa_redundancy.csv"
    ]
    
    # Dataset names for legend
    dataset_names = [
        "VQAv2",
        "VQAtext",
        "POPE",
        #"SciVQA"
    ]
    
    # Colors for each dataset
    colors = [
        'blue',      # VQAv2
        'red',       # VQAtext
        'green',     # POPE
        'purple'     # SciVQA
    ]
    
    create_cdf_plots(csv_files, redundancy_files, dataset_names, colors)