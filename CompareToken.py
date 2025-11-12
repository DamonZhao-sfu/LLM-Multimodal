import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
tokens = [576, 128, 64, 32]
x_pos = np.arange(len(tokens))

# VisPruner数据
vis_total_time = [487.51, 469.85, 385.52, 380.62]
vis_pruning_time = [0, 120.36, 89.04, 87.12]
vis_accuracy = [78.27, 78.87, 77.23, 74.47]

# CDPruner数据
cd_total_time = [479.39, 478.91, 379.21, 330.08]
cd_pruning_time = [0, 241.83, 186.59, 159.21]
cd_accuracy = [78.27, 75.80, 74.43, 71.53]

# trimTokenator数据
trim_total_time = [486.87, 297.63, 251.35, 230.01]
trim_pruning_time = [0, 113.82, 95.28, 86.94]
trim_accuracy = [78.27, 79.20, 77.37, 75.13]

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 第一个子图：堆叠柱状图
bar_width = 0.25
colors_execution = ['#3498db', '#e74c3c', '#2ecc71']
colors_pruning = ['#94C6CD', '#E29135', '#72B063']

# 绘制每个方法的柱状图
for i, (method, total, pruning, color_exec, color_prune) in enumerate([
    ('VisPruner(ICCV 2025)', vis_total_time, vis_pruning_time, colors_execution[0], colors_pruning[0]),
    ('CDPruner(NIPS 2025)', cd_total_time, cd_pruning_time, colors_execution[1], colors_pruning[1]),
    ('trimTokenator(Arxiv)', trim_total_time, trim_pruning_time, colors_execution[2], colors_pruning[2])
]):
    positions = x_pos + i * bar_width
    
    # 计算非剪枝时间 (Total - Pruning)
    non_pruning = [t - p for t, p in zip(total, pruning)]
    
    # 绘制非剪枝时间（底部）
    ax1.bar(positions, non_pruning, bar_width, label=f'{method} (Other)', 
            color=color_exec, alpha=0.7)
    
    # 绘制剪枝时间（顶部）
    ax1.bar(positions, pruning, bar_width, bottom=non_pruning,
            label=f'{method} (Pruning)', color=color_prune, alpha=0.9)

ax1.set_xlabel('Token', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time(s)', fontsize=12, fontweight='bold')
ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x_pos + bar_width)
ax1.set_xticklabels(tokens)
ax1.legend(loc='upper right', fontsize=9, ncol=2)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 第二个子图：准确率折线图
line_styles = ['-o', '-s', '-^']
line_colors = ['#3498db', '#e74c3c', '#2ecc71']

ax2.plot(tokens, vis_accuracy, line_styles[0], label='VisPruner', 
         color=line_colors[0], linewidth=2, markersize=8)
ax2.plot(tokens, cd_accuracy, line_styles[1], label='CDPruner', 
         color=line_colors[1], linewidth=2, markersize=8)
ax2.plot(tokens, trim_accuracy, line_styles[2], label='trimTokenator', 
         color=line_colors[2], linewidth=2, markersize=8)

ax2.set_xlabel('Token', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Comparison of Different Token Pruning Strategy', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(tokens)
ax2.set_xticklabels(tokens)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([70, 80])
ax2.invert_xaxis()
# 在折线图上标注数值
for i, token in enumerate(tokens):
    ax2.text(token, vis_accuracy[i], f'{vis_accuracy[i]:.1f}', 
             ha='center', va='bottom', fontsize=8)
    ax2.text(token, cd_accuracy[i], f'{cd_accuracy[i]:.1f}', 
             ha='center', va='bottom', fontsize=8)
    ax2.text(token, trim_accuracy[i], f'{trim_accuracy[i]:.1f}', 
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('token_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已生成并保存为 'token_performance_comparison.png'")
print("\n数据汇总:")
print("=" * 60)
for token, vt, vp, va, ct, cp, ca, tt, tp, ta in zip(
    tokens, vis_total_time, vis_pruning_time, vis_accuracy,
    cd_total_time, cd_pruning_time, cd_accuracy,
    trim_total_time, trim_pruning_time, trim_accuracy
):
    print(f"\nToken={token}:")
    print(f"  VisPruner:     总时间={vt:.2f}s, 剪枝={vp:.2f}s, 准确率={va:.2f}%")
    print(f"  CDPruner:      总时间={ct:.2f}s, 剪枝={cp:.2f}s, 准确率={ca:.2f}%")
    print(f"  trimTokenator: 总时间={tt:.2f}s, 剪枝={tp:.2f}s, 准确率={ta:.2f}%")