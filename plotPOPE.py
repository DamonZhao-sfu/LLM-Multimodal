import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
data = {
    'Datasets': ['POPE']*16,
    'Total Executi Accuracy': [487.51, 469.85, 385.52, 360.62,  525.65, 532.14, 447.49, 424.25,  
                                401.43, 215.68, 180.32, 165.93,  455.19,  250.39,  218.54, 200.79],
    'Token Preserve Percentage': ['100%', '22.2%', '11.1%', '5.6%', '100%', '22.2%', '11.1%', '5.6%',
                                   '100%', '22.2%', '11.1%', '5.6%', '100%', '22.2%', '11.1%', '5.6%'],
    'Accuracy': [78.27, 78.87, 77.23,74.47, 79.20, 79.60, 76.60, 72.9, 82.23, 80.43, 76.3, 69.2, 79.2, 79.5, 74.6, 64.7],
    'Strategy': ['VisPruner']*4 + ['VisPruner + V1']*4 + ['VisPruner+Grouping']*4 + ['VisPruner+V1+Grouping']*4
}

df = pd.DataFrame(data)

# Convert percentage to numeric for plotting
df['Token_Numeric'] = df['Token Preserve Percentage'].str.rstrip('%').astype(float)

# Get unique strategies and token percentages
strategies = df['Strategy'].unique()
token_percentages = sorted(df['Token_Numeric'].unique(), reverse=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== Plot 1: Bar chart for Total Execution Time =====
x = np.arange(len(token_percentages))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, strategy in enumerate(strategies):
    strategy_data = df[df['Strategy'] == strategy].sort_values('Token_Numeric', ascending=False)
    execution_times = strategy_data['Total Executi Accuracy'].values
    offset = width * (i - 1.5)
    ax1.bar(x + offset, execution_times, width, label=strategy, color=colors[i])

ax1.set_xlabel('Token Preserve Percentage', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Execution Time', fontsize=12, fontweight='bold')
ax1.set_title('Total Execution Time vs Token Preserve Percentage', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(p)}%' for p in token_percentages])
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ===== Plot 2: Line chart for Accuracy =====
markers = ['o', 's', '^', 'D']
for i, strategy in enumerate(strategies):
    strategy_data = df[df['Strategy'] == strategy].sort_values('Token_Numeric', ascending=False)
    token_pcts = strategy_data['Token_Numeric'].values
    accuracies = strategy_data['Accuracy'].values
    ax2.plot(token_pcts, accuracies, marker=markers[i], linewidth=2, 
             markersize=8, label=strategy, color=colors[i])

ax2.set_xlabel('Token Preserve Percentage', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy vs Token Preserve Percentage of POPE dataset', fontsize=14, fontweight='bold')
ax2.set_xticks(token_percentages)
ax2.set_xticklabels([f'{int(p)}%' for p in token_percentages])
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.invert_xaxis()  # So 100% is on the left

plt.tight_layout()
plt.show()
plt.savefig("pope.png")