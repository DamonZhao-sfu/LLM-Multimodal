import matplotlib.pyplot as plt
import numpy as np

# Data for trim+Async+group (trim+opt)
preserve_ratios = [5.6, 11.2, 22.4, 44.8, 56.0, 67.2, 78.4, 89.6]
trim_opt_pruning = [42.96, 49.4, 64.97, 97.6, 113.99, 129.43, 144.65, 146.54]
trim_opt_total = [104.66, 121.12, 200.42, 344.38, 439.57, 534.82, 571.82, 588.49]
trim_opt_accuracy = [76.83, 78.50, 80.33, 82.43, 82.10, 82.37, 80.53, 80.53]

# Data for trim+Async (trim)
trim_pruning = [244.71, 294.64, 387.41, 595.11, 690.52, 796.3, 899.2, 896.82]
trim_total = [323.61, 382.28, 517.69, 826.73, 1114.36, 1182.67, 1446.52, 1405.65]
trim_accuracy = [72.77, 75.80, 77.30, 79.23, 78.73, 78.93, 77.80, 77.53]

# Calculate execution time (total - pruning)
trim_opt_execution = [t - p for t, p in zip(trim_opt_total, trim_opt_pruning)]
trim_execution = [t - p for t, p in zip(trim_total, trim_pruning)]

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Set bar width and positions
bar_width = 0.35
x = np.arange(len(preserve_ratios))

# Create stacked bars for trim+opt
bars1_exec = ax1.bar(x - bar_width/2, trim_opt_execution, bar_width, 
                     label='trim+opt (execution)', color='#2E86AB', alpha=0.8)
bars1_prune = ax1.bar(x - bar_width/2, trim_opt_pruning, bar_width, 
                      bottom=trim_opt_execution, label='trim+opt (pruning)', 
                      color='#A23B72', alpha=0.8)

# Create stacked bars for trim
bars2_exec = ax1.bar(x + bar_width/2, trim_execution, bar_width, 
                     label='trim (execution)', color='#F18F01', alpha=0.8)
bars2_prune = ax1.bar(x + bar_width/2, trim_pruning, bar_width, 
                      bottom=trim_execution, label='trim (pruning)', 
                      color='#C73E1D', alpha=0.8)

# Configure left y-axis (time)
ax1.set_xlabel('Token Preserve Ratio (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Execution Time (ms)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{r}%' for r in preserve_ratios])
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Create second y-axis for accuracy
ax2 = ax1.twinx()
line1 = ax2.plot(x + bar_width/2, trim_opt_accuracy, 'o-', 
                 linewidth=2.5, markersize=8, color='#2E86AB', 
                 label='trim+opt accuracy', markeredgecolor='white', markeredgewidth=1.5)
line2 = ax2.plot(x + bar_width/2, trim_accuracy, 's-', 
                 linewidth=2.5, markersize=8, color='#F18F01', 
                 label='trim accuracy', markeredgecolor='white', markeredgewidth=1.5)

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(70, 85)

# Combine legends
bars_legend = ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
lines_legend = ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Title
plt.title('Token Pruning Performance: Execution Time and Accuracy\n(LLava1.6 POPE Dataset)', 
          fontsize=14, fontweight='bold', pad=20)

# Layout adjustment
fig.tight_layout()
plt.show()
plt.savefig("./llava16pope.png")
# Print summary statistics
print("=== Performance Summary ===\n")
print("trim+opt (trim+Async+group):")
print(f"  Avg Total Time: {np.mean(trim_opt_total):.2f} ms")
print(f"  Avg Pruning Time: {np.mean(trim_opt_pruning):.2f} ms")
print(f"  Avg Accuracy: {np.mean(trim_opt_accuracy):.2f}%\n")

print("trim (trim+Async):")
print(f"  Avg Total Time: {np.mean(trim_total):.2f} ms")
print(f"  Avg Pruning Time: {np.mean(trim_pruning):.2f} ms")
print(f"  Avg Accuracy: {np.mean(trim_accuracy):.2f}%\n")

print(f"Speedup (trim+opt vs trim): {np.mean(trim_total)/np.mean(trim_opt_total):.2f}x")
print(f"Accuracy gain: {np.mean(trim_opt_accuracy) - np.mean(trim_accuracy):.2f}%")
