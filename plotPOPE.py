# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Data
# data = {
#     'Datasets': ['POPE']*16,
#     'Total Executi Accuracy': [487.51, 469.85, 385.52, 360.62,  525.65, 532.14, 447.49, 424.25,  
#                                 401.43, 215.68, 180.32, 165.93,  455.19,  250.39,  218.54, 200.79],
#     'Token Preserve Percentage': ['100%', '22.2%', '11.1%', '5.6%', '100%', '22.2%', '11.1%', '5.6%',
#                                    '100%', '22.2%', '11.1%', '5.6%', '100%', '22.2%', '11.1%', '5.6%'],
#     'Accuracy': [78.27, 78.87, 77.23,74.47, 79.20, 79.60, 76.60, 72.9, 82.23, 80.43, 76.3, 69.2, 79.2, 79.5, 74.6, 64.7],
#     'Strategy': ['VisPruner']*4 + ['VisPruner + V1']*4 + ['VisPruner+Grouping']*4 + ['VisPruner+V1+Grouping']*4
# }

# df = pd.DataFrame(data)

# # Convert percentage to numeric for plotting
# df['Token_Numeric'] = df['Token Preserve Percentage'].str.rstrip('%').astype(float)

# # Get unique strategies and token percentages
# strategies = df['Strategy'].unique()
# token_percentages = sorted(df['Token_Numeric'].unique(), reverse=True)

# # Create figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # ===== Plot 1: Bar chart for Total Execution Time =====
# x = np.arange(len(token_percentages))
# width = 0.2
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# for i, strategy in enumerate(strategies):
#     strategy_data = df[df['Strategy'] == strategy].sort_values('Token_Numeric', ascending=False)
#     execution_times = strategy_data['Total Executi Accuracy'].values
#     offset = width * (i - 1.5)
#     ax1.bar(x + offset, execution_times, width, label=strategy, color=colors[i])

# ax1.set_xlabel('Token Preserve Percentage', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Total Execution Time', fontsize=12, fontweight='bold')
# ax1.set_title('Total Execution Time vs Token Preserve Percentage', fontsize=14, fontweight='bold')
# ax1.set_xticks(x)
# ax1.set_xticklabels([f'{int(p)}%' for p in token_percentages])
# ax1.legend(loc='upper right')
# ax1.grid(axis='y', alpha=0.3, linestyle='--')

# # ===== Plot 2: Line chart for Accuracy =====
# markers = ['o', 's', '^', 'D']
# for i, strategy in enumerate(strategies):
#     strategy_data = df[df['Strategy'] == strategy].sort_values('Token_Numeric', ascending=False)
#     token_pcts = strategy_data['Token_Numeric'].values
#     accuracies = strategy_data['Accuracy'].values
#     ax2.plot(token_pcts, accuracies, marker=markers[i], linewidth=2, 
#              markersize=8, label=strategy, color=colors[i])

# ax2.set_xlabel('Token Preserve Percentage', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax2.set_title('Accuracy vs Token Preserve Percentage of POPE dataset', fontsize=14, fontweight='bold')
# ax2.set_xticks(token_percentages)
# ax2.set_xticklabels([f'{int(p)}%' for p in token_percentages])
# ax2.legend(loc='best')
# ax2.grid(True, alpha=0.3, linestyle='--')
# ax2.invert_xaxis()  # So 100% is on the left

# plt.tight_layout()
# plt.show()
# plt.savefig("pope.png")

import matplotlib.pyplot as plt
import numpy as np

# Data from the Excel table
labels = ['5.60%', '11.20%', '22.40%', '44.80%', '56%', '67.20%', '78.40%', '89.60%']
pruning_time = [42.96, 49.4, 64.97, 97.6, 113.99, 129.43, 144.65, 146.54]
total_time = [104.66, 121.12, 200.42, 344.38, 439.57, 534.82, 571.82, 588.49]
accuracy = [78.83, 78.50, 80.33, 82.43, 82.10, 82.37, 80.53, 80.53]

# Calculate inference time (total - pruning)
inference_time = [total_time[i] - pruning_time[i] for i in range(len(total_time))]

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# X-axis positions
x = np.arange(len(labels))
width = 0.6

# Create stacked bar chart
bars1 = ax1.bar(x, inference_time, width, label='Inference Time', 
                color='#4CAF50', alpha=0.8, edgecolor='white', linewidth=1.5)
bars2 = ax1.bar(x, pruning_time, width, bottom=inference_time, 
                label='Pruning Time', color='#FF9800', alpha=0.8, 
                edgecolor='white', linewidth=1.5)

# Set labels and title for primary axis
ax1.set_xlabel('Token Keep Ratio', fontsize=13, fontweight='bold')
ax1.set_ylabel('Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Execution Time Breakdown and Accuracy vs Token Keep Ratio of POPE dataset with Llava-Next', 
              fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelsize=11)

# Add grid for better readability
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax1.set_axisbelow(True)

# Create secondary y-axis for accuracy
ax2 = ax1.twinx()
line = ax2.plot(x, accuracy, color='#E91E63', marker='o', linewidth=3, 
                markersize=8, label='Accuracy', markeredgecolor='white', 
                markeredgewidth=1.5)

# Set labels for secondary axis
ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_ylim(60, 85)
ax2.tick_params(axis='y', labelsize=11)

# Combine legends from both axes
bars_legend = ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
line_legend = ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Add value labels on bars (optional - can be removed if cluttered)
for i, (inf, pru) in enumerate(zip(inference_time, pruning_time)):
    # Label for total time at the top of the bar
    total = inf + pru
    ax1.text(i, total + 15, f'{total:.1f}', ha='center', va='bottom', 
             fontsize=9, fontweight='bold')

# Add value labels on accuracy line points
for i, acc in enumerate(accuracy):
    ax2.text(i, acc + 0.8, f'{acc:.1f}%', ha='center', va='bottom', 
             fontsize=9, fontweight='bold', color='#E91E63')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('pope_execution_time_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as 'pope_execution_time_accuracy.png'")