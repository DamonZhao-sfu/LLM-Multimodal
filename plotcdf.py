import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_cdf(data, label, ax, color=None):
    """Plot CDF for given data."""
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, y, label=label, linewidth=2, color=color)

def create_cdf_plots(csv_file):
    """Create CDF plots from the token counts CSV file."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Subplot 1: CDF of total_pixels =====
    ax1 = axes[0]
    plot_cdf(df['total_pixels'], 'Total Pixels', ax1, color='blue')
    ax1.set_xlabel('Total Pixels', fontsize=12)
    ax1.set_ylabel('CDF', fontsize=12)
    ax1.set_title('CDF Distribution of Total Pixels', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # ===== Subplot 2: CDF of tokens =====
    ax2 = axes[1]
    plot_cdf(df['image_tokens'], 'Image Tokens', ax2, color='red')
    plot_cdf(df['text_tokens'], 'Text Tokens', ax2, color='green')
    plot_cdf(df['total_tokens'], 'Total Tokens', ax2, color='purple')
    ax2.set_xlabel('Number of Tokens', fontsize=12)
    ax2.set_ylabel('CDF', fontsize=12)
    ax2.set_title('CDF Distribution of Tokens', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('VQAtext.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'cdf_distributions.png'")
    
    # Show the plot
    plt.show()
    
    # Print some statistics
    print("\n=== Statistics ===")
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

# Usage
if __name__ == "__main__":
    csv_file = "/home/haikai/LLM-Multimodal/VQAtext.csv"  # Replace with your CSV file path
    create_cdf_plots(csv_file)
