import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Define the data
data = {
    'VQAv2': {
        'Trim': {
            576: {'preprocess': 84.34, 'prune': 0, 'total': 766.82, 'accuracy': 87.60},
            128: {'preprocess': 84.34, 'prune': 33.8, 'total': 439.46, 'accuracy': 85.63},
            64: {'preprocess': 84.34, 'prune': 14.71, 'total': 313.9, 'accuracy': 84.24},
            32: {'preprocess': 84.34, 'prune': 5.24, 'total': 238.31, 'accuracy': 82.30}
        },
        'CDPruner': {
            576: {'preprocess': 84.34, 'prune': 0, 'total': 816.51, 'accuracy': 87.50},
            128: {'preprocess': 84.34, 'prune': 179.7, 'total': 992.29, 'accuracy': 82.30},
            64: {'preprocess': 84.34, 'prune': 118.94, 'total': 452.61, 'accuracy': 82.08},
            32: {'preprocess': 84.34, 'prune': 86.42, 'total': 394.33, 'accuracy': 81.53}
        },
        'VisPruner': {
            576: {'preprocess': 84.34, 'prune': 0, 'total': 814.01, 'accuracy': 87.50},
            128: {'preprocess': 84.34, 'prune': 111.92, 'total': 654.02, 'accuracy': 86.55},
            64: {'preprocess': 84.34, 'prune': 74.88, 'total': 538.42, 'accuracy': 85.25},
            32: {'preprocess': 84.34, 'prune': 67.36, 'total': 482.58, 'accuracy': 85.76}
        }
    },
    'VQAtext': {
        'Trim': {
            576: {'preprocess': 78.92, 'prune': 0, 'total': 389.16, 'accuracy': 58.31},
            128: {'preprocess': 78.92, 'prune': 42.98, 'total': 431.38, 'accuracy': 54.23},
            64: {'preprocess': 78.92, 'prune': 27.64, 'total': 406.46, 'accuracy': 50.51},
            32: {'preprocess': 78.92, 'prune': 14.29, 'total': 285.01, 'accuracy': 45.53}
        },
        'CDPruner': {
            576: {'preprocess': 77.8, 'prune': 0, 'total': 395.87, 'accuracy': 58.31},
            128: {'preprocess': 77.8, 'prune': 76.64, 'total': 383.13, 'accuracy': 53.03},
            64: {'preprocess': 77.8, 'prune': 44.66, 'total': 335.25, 'accuracy': 50.39},
            32: {'preprocess': 77.8, 'prune': 23.13, 'total': 291.42, 'accuracy': 34.97}
        },
        'VisPruner': {
            576: {'preprocess': 78.92, 'prune': 0, 'total': 400.39, 'accuracy': 58.31},
            128: {'preprocess': 78.92, 'prune': 40.39, 'total': 372.61, 'accuracy': 55.31},
            64: {'preprocess': 78.92, 'prune': 20.69, 'total': 320.78, 'accuracy': 51.17},
            32: {'preprocess': 78.92, 'prune': 17.22, 'total': 315.85, 'accuracy': 45.83}
        },

    },
    'POPE': {
        'Trim': {
            576: {'preprocess': 81.23, 'prune': 0, 'total': 287.62, 'accuracy': 78.27},
            128: {'preprocess': 81.23, 'prune': 32.59, 'total': 172.35, 'accuracy': 79.20},
            64: {'preprocess': 81.23, 'prune': 14.05, 'total': 137.45, 'accuracy': 77.37},
            32: {'preprocess': 81.23, 'prune': 5.71, 'total': 121.4, 'accuracy': 75.13}
        },
        'CDPruner': {
            576: {'preprocess': 81.23, 'prune': 0, 'total': 285.63, 'accuracy': 78.27},
            128: {'preprocess': 81.23, 'prune': 160.6, 'total': 331.61, 'accuracy': 75.60},
            64: {'preprocess': 81.23, 'prune': 105.36, 'total': 256.32, 'accuracy': 74.43},
            32: {'preprocess': 81.23, 'prune': 77.98, 'total': 218.43, 'accuracy': 71.53}
        },
        'VisPruner': {
            576: {'preprocess': 81.23, 'prune': 0, 'total': 294.39, 'accuracy': 78.27},
            128: {'preprocess': 81.23, 'prune': 100.91, 'total': 301.28, 'accuracy': 78.87},
            64: {'preprocess': 81.23, 'prune': 67.42, 'total': 245.3, 'accuracy': 77.23},
            32: {'preprocess': 81.23, 'prune': 61.04, 'total': 243.75, 'accuracy': 74.47}
        }
    },
    'sciVQA': {
        'Trim': {
            576: {'preprocess': 26.27, 'prune': 0, 'total': 146.26, 'accuracy': 57.20},
            128: {'preprocess': 26.27, 'prune': 12.02, 'total': 133.8, 'accuracy': 51.90},
            64: {'preprocess': 26.27, 'prune': 5.12, 'total': 107.35, 'accuracy': 54.20},
            32: {'preprocess': 26.27, 'prune': 2.17, 'total': 102.32, 'accuracy': 52.80}
        },
        'CDPruner': {
            576: {'preprocess': 26.27, 'prune': 0, 'total': 146.49, 'accuracy': 57.10},
            128: {'preprocess': 26.27, 'prune': 42.36, 'total': 246.1, 'accuracy': 53.70},
            64: {'preprocess': 26.27, 'prune': 21.65, 'total': 241.33, 'accuracy': 52.50},
            32: {'preprocess': 26.27, 'prune': 13.66, 'total': 227.68, 'accuracy': 51.60}
        },
        'VisPruner': {
            576: {'preprocess': 26.27, 'prune': 0, 'total': 149.17, 'accuracy': 57.20},
            128: {'preprocess': 26.27, 'prune': 33.69, 'total': 269.71, 'accuracy': 55.00},
            64: {'preprocess': 26.27, 'prune': 23.18, 'total': 229.61, 'accuracy': 53.30},
            32: {'preprocess': 26.27, 'prune': 21.88, 'total': 228.1, 'accuracy': 52.80}
        }
    }
}

# Define color schemes for each method
method_colors = {
    'Trim': ['#FFB3BA', '#FF7C88', '#FF4D5A'],           # Light to dark red
    'CDPruner': ['#BAFFC9', '#7CFF94', '#4DFF6B'],      # Light to dark green
    'VisPruner': ['#BAE1FF', '#7CC8FF', '#4DB8FF'],     # Light to dark blue
}

# Define hatching patterns for different sections
# preprocess: dots, prune: vertical lines, inference: diagonal lines
# Define hatching patterns for different sections
# preprocess: dots, prune: vertical lines, inference: diagonal lines
hatch_patterns = ['...', '|||', 'xxx']

# Create figure with 8 subplots (2 rows, 4 columns)
fig = plt.figure(figsize=(24, 12))
datasets = ['VQAv2', 'VQAtext', 'POPE', 'sciVQA']
token_numbers = [576, 128, 64, 32]

# Line colors and markers for accuracy plots
markers = ['o', 's', '^']

# First row: Stacked bar charts
for idx, dataset in enumerate(datasets):
    ax = plt.subplot(2, 4, idx + 1)
    methods = list(data[dataset].keys())
    
    # Set up bar positions
    x = np.arange(len(token_numbers))
    width = 0.25
    offsets = [-width, 0, width]
    
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
               hatch=hatch_patterns[0], label='Preprocess' if i == 0 and idx == 0 else '')
        ax.bar(x + offsets[i], prune_times, width, 
               bottom=preprocess_times,
               color=colors[1], edgecolor='black', linewidth=0.5,
               hatch=hatch_patterns[1], label='Prune' if i == 0 and idx == 0 else '')
        ax.bar(x + offsets[i], inference_times, width, 
               bottom=np.array(preprocess_times) + np.array(prune_times),
               color=colors[2], edgecolor='black', linewidth=0.5,
               hatch=hatch_patterns[2], label='Inference' if i == 0 and idx == 0 else '')
    
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
        # Use the medium shade (prune color) as representative color
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
    methods = list(data[dataset].keys())
    
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
                marker=markers[i % 3], 
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
plt.savefig('time_breakdown_and_accuracy_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Chart saved as 'time_breakdown_and_accuracy_analysis.png'")
print("\nColor scheme:")
print("- Red shades: Trim methods")
print("- Green shades: CDPruner methods")
print("- Blue shades: VisPruner methods")
print("\nHatching patterns:")
print("- Dots (...): Preprocess")
print("- Vertical lines (|||): Prune")
print("- Diagonal lines (xxx): Inference")