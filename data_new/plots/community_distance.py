import matplotlib.pyplot as plt
import numpy as np

# Data from the table
communities = ['7B', '13B', '70B', 'Coding', 'Bio/Med', 'Physics']
intra_distances = [9.600, 8.917, 9.219, 9.460, 8.937, 8.509]
inter_distances = [9.624, 9.327, 9.474, 9.591, 9.314, 9.319]

# X-axis position of the bars
x = np.arange(len(communities))

# Width of the bars
width = 0.3

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the side-by-side bars with more distinct colors
rects1 = ax.bar(x - width/2, intra_distances, width, label='Intra-Community', color='#1f77b4')  # Blue
rects2 = ax.bar(x + width/2, inter_distances, width, label='Inter-Community', color='#ff7f0e')  # Orange

# Add labels, title, and custom ticks on the x-axis
ax.set_xlabel('Community', fontsize=20)
ax.set_ylabel('Averaged L2 Distance', fontsize=20)
ax.set_title('Averaged Intra-Community vs Inter-Community L2 Distances', fontsize=20)

# Increase the y-axis range to emphasize differences
ax.set_ylim(8, 10)

# Set x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(communities, fontsize=20)
ax.legend(fontsize=16)

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.6)

# # Function to add labels on top of the bars
# def add_labels(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.3f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# # Apply the function to both bar sets
# add_labels(rects1)
# add_labels(rects2)

# Adjust layout to avoid cutting off labels
fig.tight_layout()

# Show plot
plt.show()
plt.savefig("community_distance.png")