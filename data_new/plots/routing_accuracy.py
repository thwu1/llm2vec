import matplotlib.pyplot as plt
import numpy as np

# Benchmark names
benchmarks = [
    'Overall', 'ASDiv', 'GPQA', 'GSM8K', 'LogiQA', 'MathQA', 
    'MedMCQA', 'MMLU', 'PIQA', 'SocialQA', 'TruthfulQA'
]

# Accuracy values for each method (Matrix Factorization, Single-Best, Weighted)
matrix_factorization = [0.6697, 0.6667, 0.3000, 0.8929, 0.4510, 0.5527, 0.7514, 0.8489, 0.8731, 0.3148, 0.5270]
single_best = [0.6050, 0.7020, 0.3100, 0.8929, 0.5294, 0.5992, 0.7542, 0.8625, 0.8955, 0.3951, 0.5541]
weighted = [0.5304, 0.1298, 0.2528, 0.6783, 0.4142, 0.4378, 0.5783, 0.7168, 0.8441, 0.3115, 0.3439]

# Number of benchmarks
n_benchmarks = len(benchmarks)

# X-axis positions for the benchmarks
x = np.arange(n_benchmarks)

# Bar width
bar_width = 0.25

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
bar1 = ax.bar(x - bar_width, matrix_factorization, width=bar_width, label='Matrix Factorization', color='steelblue')
bar2 = ax.bar(x, single_best, width=bar_width, label='Single-Best', color='lightcoral')
bar3 = ax.bar(x + bar_width, weighted, width=bar_width, label='Weighted', color='gold')

# Labels, title, and ticks
ax.set_xlabel('Benchmark', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Routing Accuracy Comparison Across Benchmarks', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, rotation=45, ha='right')

# Add a grid and a legend
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig("routing_accuracy.png")