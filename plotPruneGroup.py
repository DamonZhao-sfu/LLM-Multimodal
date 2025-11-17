import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Define the data
data = {
    'POPE': {
        'Trim': {
            576: {'preprocess': 81.23, 'prune': 0, 'total': 287.62, 'accuracy': 78.27},
            128: {'preprocess': 81.23, 'prune': 32.59, 'total': 172.35, 'accuracy': 79.20},
            64: {'preprocess': 81.23, 'prune': 14.05, 'total': 137.45, 'accuracy': 77.37},
            32: {'preprocess': 81.23, 'prune': 5.71, 'total': 121.4, 'accuracy': 75.13}
        },
        'Trim+group': {
            576: {'preprocess': 13.66, 'prune': 0, 'total': 205.56, 'accuracy': 82.23},
            128: {'preprocess': 13.66, 'prune': 6.3, 'total': 111.78, 'accuracy': 80.77},
            64: {'preprocess': 13.66, 'prune': 2.69, 'total': 92.48, 'accuracy': 76.73},
            32: {'preprocess': 13.66, 'prune': 1.36, 'total': 93.28, 'accuracy': 72.90}
        }
    },
    'VQAv2': {
        'Trim': {
            576: {'preprocess': 84.34, 'prune': 0, 'total': 766.82, 'accuracy': 87.60},
            128: {'preprocess': 84.34, 'prune': 33.8, 'total': 439.46, 'accuracy': 85.63},
            64: {'preprocess': 84.34, 'prune': 14.71, 'total': 313.9, 'accuracy': 84.24},
            32: {'preprocess': 84.34, 'prune': 5.24, 'total': 238.31, 'accuracy': 82.30}
        },
        'Trim+group': {
            576: {'preprocess': 14.63, 'prune': 0, 'total': 334.84, 'accuracy': 78.78},
            128: {'preprocess': 14.63, 'prune': 7.12, 'total': 193.02, 'accuracy': 79.16},
            64: {'preprocess': 14.63, 'prune': 2.98, 'total': 172.74, 'accuracy': 79.23},
            32: {'preprocess': 14.63, 'prune': 1.09, 'total': 206.9, 'accuracy': 80.75}
        }
    },
    'VQAtext': {
        'Trim': {
            576: {'preprocess': 78.92, 'prune': 0, 'total': 389.16, 'accuracy': 58.31},
            128: {'preprocess': 78.92, 'prune': 42.98, 'total': 431.38, 'accuracy': 54.23},
            64: {'preprocess': 78.92, 'prune': 27.64, 'total': 406.46, 'accuracy': 50.51},
            32: {'preprocess': 78.92, 'prune': 14.29, 'total': 285.01, 'accuracy': 45.53}
        },
        'Trim+group': {
            576: {'preprocess': 36.58, 'prune': 0, 'total': 365.97, 'accuracy': 28.55},
            128: {'preprocess': 36.58, 'prune': 12.17, 'total': 229.29, 'accuracy': 26.57},
            64: {'preprocess': 36.58, 'prune': 4.99, 'total': 252.98, 'accuracy': 25.49},
            32: {'preprocess': 36.58, 'prune': 1.36, 'total': 232.9, 'accuracy': 23.76}
        }
    },
    'sciVQA': {
        'Trim': {
            576: {'preprocess': 26.27, 'prune': 0, 'total': 146.26, 'accuracy': 57.20},
            128: {'preprocess': 26.27, 'prune': 12.02, 'total': 133.8, 'accuracy': 51.90},
            64: {'preprocess': 26.27, 'prune': 5.12, 'total': 107.35, 'accuracy': 54.20},
            32: {'preprocess': 26.27, 'prune': 2.17, 'total': 102.32, 'accuracy': 52.80}
        },
        'Trim+group': {
            576: {'preprocess': 3.9, 'prune': 0, 'total': 103.94, 'accuracy': 29.40},
            128: {'preprocess': 3.9, 'prune': 1.66, 'total': 65.11, 'accuracy': 30.50},
            64: {'preprocess': 3.9, 'prune': 0.51, 'total': 58.95, 'accuracy': 32.40},
            32: {'preprocess': 3.9, 'prune': 0.07, 'total': 57.33, 'accuracy': 37.70}
        }
    }
}

# Define color schemes for each method
method_colors = {
    'Trim': ['#FFB3BA', '#FF7C88', '#FF4D5A'],           # Light to dark red
    'Trim+group': ['#BAFFC9', '#7CFF94', '#4DFF6B'],    # Light to dark green
}

# Define hatching patterns for different sections
hatch_patterns = ['...', '|||', 'xxx']

# Create figure with 8 subplots (2 rows, 4 columns)
fig = plt.figure(figsize=(24, 12))
datasets = ['POPE', 'VQAv2', 'VQAtext', 'sciVQA']
token_numbers = [576, 128, 64, 32]

# Line colors and markers for accuracy plots
markers = ['o', 's']

# First row: Stacked bar charts
for idx, dataset in enumerate(datasets):
    ax = plt.subplot(2, 4, idx + 1)
    methods = ['Trim', 'Trim+group']
    
    # Set up bar positions
    x = np.arange(len(token_numbers))
    width = 0.35
    offsets = [-width/2, width/2]
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        preprocess_times = []
        prune_times = []
        inference_times = []
        
        for token in token_numbers:
            preprocess = data[dataset][method][token]['preprocess']
            prune = data[dataset][method][token]['prune']
            total = data[dataset][method][token]['total']
            inference = total - preprocess - prune
            
            preprocess_times.append(preprocess)
            prune_times.append(prune)
            inference_times.append(inference)
        
        # Get colors for this method
        colors = method_colors[method]
        
        # Create stacked bars with different hatching patterns
        ax.bar(x + offsets[i], preprocess_times, width, 
               color=colors[0], edgecolor='black', linewidth=0.5,
               hatch=hatch_patterns[0])
        ax.bar(x + offsets[i], prune_times, width, 
               bottom=preprocess_times,
               color=colors[1], edgecolor='black', linewidth=0.5,
               hatch=hatch_patterns[1])
        ax.bar(x + offsets[i], inference_times, width, 
               bottom=np.array(preprocess_times) + np.array(prune_times),
               color=colors[2], edgecolor='black', linewidth=0.5,
               hatch=hatch_patterns[2])
    
    # Customize the plot
    ax.set_xlabel('Token Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset} - Time Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(token_numbers)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create custom legend for this subplot with both methods and patterns
    legend_elements = []
    
    # Add method legends
    for method in methods:
        colors = method_colors[method]
        legend_elements.append(Patch(facecolor=colors[1], label=method, edgecolor='black'))
    
    # Add separator (empty space)
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))
    
    # Add pattern legends
    legend_elements.append(Patch(facecolor='lightgray', hatch='...', 
                                  label='Preprocess', edgecolor='black'))
    legend_elements.append(Patch(facecolor='lightgray', hatch='|||', 
                                  label='Prune', edgecolor='black'))
    legend_elements.append(Patch(facecolor='lightgray', hatch='xxx', 
                                  label='Inference', edgecolor='black'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=1)

# Second row: Accuracy line plots
for idx, dataset in enumerate(datasets):
    ax = plt.subplot(2, 4, idx + 5)
    methods = ['Trim', 'Trim+group']
    
    # Plot lines for each method
    for i, method in enumerate(methods):
        accuracies = []
        
        for token in token_numbers:
            accuracy = data[dataset][method][token]['accuracy']
            accuracies.append(accuracy)
        
        # Get the representative color for this method
        colors = method_colors[method]
        
        # Create line plot
        ax.plot(token_numbers, accuracies, 
                color=colors[2],  # Use the darkest shade
                marker=markers[i], 
                linewidth=2.5, 
                markersize=8,
                label=method,
                alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Token Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset} - Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(token_numbers)
    ax.set_xticklabels(token_numbers)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    
    # Invert x-axis to show 576 -> 32 from left to right
    ax.invert_xaxis()

plt.tight_layout()
plt.savefig('trim_vs_trimgroup_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Chart saved as 'trim_vs_trimgroup_analysis.png'")
print("\nColor scheme:")
print("- Red shades: Trim method")
print("- Green shades: Trim+group method")
print("\nHatching patterns:")
print("- Dots (...): Preprocess")
print("- Vertical lines (|||): Prune")
print("- Diagonal lines (xxx): Inference")
