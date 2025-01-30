import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd

# Load embeddings
embeddings = torch.load("tsne_model_embeddings_dim_2.pth").detach().cpu().numpy()
print(embeddings.shape)
# Load model order CSV
model_order = pd.read_csv("model_order.csv")

# Ensure the model order matches embeddings
assert len(model_order) == embeddings.shape[0], "Model order and embeddings length mismatch!"

# Filter based on a substring in model names
filter_substring = "70b"  # Change this substring to filter by other criteria
filtered_indices = model_order[model_order['model_name'].str.contains(filter_substring, case=False)].index.tolist()
filtered_names = model_order.loc[filtered_indices, 'model_name'].tolist()

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

# Prepare data for plotting
colors = plt.cm.tab20(np.linspace(0, 1, len(filtered_indices)))  # Use a colormap for the filtered subset

plt.figure(figsize=(12, 10))

# Plot all embeddings with gray for unselected models
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='lightgray', s=30, alpha=0.5, label='_nolegend_')

# Highlight filtered embeddings
for i, idx in enumerate(filtered_indices):
    x, y = embeddings_2d[idx]
    plt.scatter(x, y, color=colors[i], label=filtered_names[i], s=100)

# Add legend with filtered model names
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title=f'Models with "{filter_substring}"')
plt.title('t-SNE Visualization of Embeddings', fontsize=16)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()

# Show and save the plot
plt.savefig("test_filtered.png", dpi=300)
plt.show()