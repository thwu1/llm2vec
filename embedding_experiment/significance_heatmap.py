import json
import matplotlib.pyplot as plt
import numpy as np
import ast

def convert_keys_to_tuples(d):
    new_dict = {}
    for key, value in d.items():
        # Convert the key from string to tuple
        tuple_key = ast.literal_eval(key)
        new_dict[tuple_key] = value
    return new_dict

with open("mse_differences_result.json", 'r') as json_file:
    significance_differences = convert_keys_to_tuples(json.load(json_file))

# Variance data
# variances = {
#     'gpqa': 0.000059, 'social': 0.000034, 'logiqa': 0.003208, 'piqa': 0.003250, 'mathqa': 0.003919,
#     'asdiv': 0.006696, 'medmcqa': 0.010461, 'truthfulqa': 0.012863, 'mmlu': 0.020451, 'gsm8k': 0.067687
# }
# Sorted by Dataset Size
variances = {
    'gpqa': 2400, 
    # 'social': 1900, 
    'logiqa': 651, 'piqa': 1800, 'mathqa': 3000,
    'asdiv': 2300, 'medmcqa': 4200, 'truthfulqa': 800, 'mmlu': 14000, 'gsm8k': 1300
}

# Sort benchmarks by increasing variance
sorted_benchmarks = sorted(variances, key=variances.get)

# Create a 10x10 matrix
size = len(sorted_benchmarks)
heatmap_matrix = np.zeros((size, size))

position = dict(zip(sorted_benchmarks, range(size)))

# Fill the matrix with values from the dictionary
for key, value in significance_differences.items():
    if key[0] == "social" or key[1] == "social":
        continue
    x = position[key[0]]
    y = position[key[1]]

    i = size - y - 1
    j = x

    heatmap_matrix [i, j] = value
        
# Set the color limits (vmin and vmax)
vmin, vmax = -2, 2 # Adjust these values based on your data to improve visualization

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_matrix, cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label='Value')
plt.xticks(ticks=np.arange(size), labels=sorted_benchmarks, rotation=90)
plt.yticks(ticks=np.arange(size), labels=sorted_benchmarks[::-1])
plt.title('Heatmap of Benchmark Values')
# Annotate each cell with the numeric value
for i in range(size):
    for j in range(size):
        text = plt.text(j, i, format(heatmap_matrix[i, j], '.3f'),
                       ha='center', va='center', color='black')
        
# plt.savefig("mse_significance_heatmap_by_dataset_size.png")
plt.show()

# Calculate row sums
row_sums = heatmap_matrix.sum(axis=1)
print(row_sums)

# Plot the row sums
plt.figure(figsize=(8,6))
plt.bar(np.arange(size), row_sums, color='skyblue')
plt.xticks(ticks=np.arange(size), labels=reversed(sorted_benchmarks), rotation=45, fontsize=16)
plt.xlabel('Benchmark', fontsize=20)
plt.ylabel('Level of Effect', fontsize=20)
plt.title('Row Sum of Contribution Matrix $C$', fontsize=20)
plt.tight_layout()
plt.savefig("heatmap_row_sum.png")
plt.show()

# Calculate column sums
column_sums = heatmap_matrix.sum(axis=0)

# Plot the column sums
plt.figure(figsize=(8,6))
plt.bar(np.arange(size), list(reversed(column_sums)), color='salmon')
plt.xticks(ticks=np.arange(size), labels=reversed(sorted_benchmarks), rotation=45, fontsize=16)
plt.xlabel('Benchmark', fontsize=20)
plt.ylabel('Level of Effect', fontsize=20)
plt.title('Column Sum of Contribution Matrix $C$', fontsize=20)
plt.tight_layout()
plt.savefig("heatmap_column_sum.png")
plt.show()