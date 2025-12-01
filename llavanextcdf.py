import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_cdf(data, label, ax, color=None, linestyle='-'):
    """Helper function to calculate and plot CDF for a single series."""
    # Drop NaNs to avoid plotting errors
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return
        
    sorted_data = np.sort(clean_data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, y, label=label, linewidth=2, color=color, linestyle=linestyle)

def create_token_distribution_plots(csv_files, dataset_names):
    """
    Creates a grid of subplots. 
    Each subplot represents ONE dataset and contains CDFs for:
    - Image Tokens
    - Text Tokens
    - Total Tokens
    """
    
    # Determine grid size (approximate square or 2 columns)
    n_datasets = len(dataset_names)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    # Define consistent styling for the token types
    token_styles = {
        'image_tokens': {'label': 'Image Tokens', 'color': '#1f77b4', 'style': '--'}, # Blue, dashed
        'text_tokens':  {'label': 'Text Tokens',  'color': '#ff7f0e', 'style': '-.'}, # Orange, dash-dot
        'total_tokens': {'label': 'Total Tokens', 'color': '#2ca02c', 'style': '-'}   # Green, solid
    }

    print("Processing datasets...")
    
    for i, (name, file_path) in enumerate(zip(dataset_names, csv_files)):
        ax = axes[i]
        
        try:
            print(f"  Loading {name}...")
            df = pd.read_csv(file_path)
            
            # Plot Image Tokens
            plot_cdf(
                df['image_tokens'], 
                token_styles['image_tokens']['label'], 
                ax, 
                color=token_styles['image_tokens']['color'], 
                linestyle=token_styles['image_tokens']['style']
            )
            
            # Plot Text Tokens
            plot_cdf(
                df['text_tokens'], 
                token_styles['text_tokens']['label'], 
                ax, 
                color=token_styles['text_tokens']['color'], 
                linestyle=token_styles['text_tokens']['style']
            )
            
            # Plot Total Tokens
            plot_cdf(
                df['total_tokens'], 
                token_styles['total_tokens']['label'], 
                ax, 
                color=token_styles['total_tokens']['color'], 
                linestyle=token_styles['total_tokens']['style']
            )
            
            # Subplot Formatting
            ax.set_title(f"Dataset: {name}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Tokens', fontsize=12)
            ax.set_ylabel('CDF', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='lower right')
            
            # Print quick stats for verification
            print(f"    {name} Avg Total Tokens: {df['total_tokens'].mean():.2f}")
            
        except Exception as e:
            print(f"    Error processing {name}: {e}")
            ax.text(0.5, 0.5, f"Error loading data\n{e}", 
                    ha='center', va='center', transform=ax.transAxes)

    # Hide any unused subplots if n_datasets is odd
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Token Distribution Analysis per Dataset', fontsize=18, y=1.02)
    plt.tight_layout()
    
    output_filename = 'dataset_token_distributions.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_filename}'")
    plt.show()

# Usage
if __name__ == "__main__":
    # File paths based on your previous code
    csv_files = [
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQAv2.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/VQAtext.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/POPE.csv",
        "/scratch/hpc-prf-haqc/haikai/LLM-Multimodal/scivqa_token_counts.csv"
    ]
    
    # Dataset names
    dataset_names = [
        "VQAv2",
        "VQAtext",
        "POPE",
        "SciVQA"
    ]
    
    create_token_distribution_plots(csv_files, dataset_names)
