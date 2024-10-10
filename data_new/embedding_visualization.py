import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings_pca(embeddings, indices=None, keyword=None):
    """
    Performs PCA on the embeddings and visualizes them in 2D with optional color grouping.
    
    Parameters:
    - embeddings: numpy array of shape (n_samples, n_features) containing the embeddings.
    - indices: list or numpy array of indices to color differently. If None, all points are colored the same.
    
    """
    # Perform PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Plot the embeddings in 2D
    plt.figure(figsize=(8, 6))
    
    if indices is None:
        # If no indices provided, plot all embeddings in the same color
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue', label='Embeddings')
    else:
        # Create two groups, one for colored indices and one for non-colored
        non_colored_indices = np.setdiff1d(np.arange(embeddings.shape[0]), indices)
        
        # Plot the non-colored embeddings
        plt.scatter(reduced_embeddings[non_colored_indices, 0], reduced_embeddings[non_colored_indices, 1],
                    color='blue', label='Class 0')
        
        # Plot the colored embeddings
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1],
                    color='red', label=keyword)
    
    # Add labels and a legend
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.title('PCA of Embeddings in 2D')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
    plt.savefig("embeddings_visualization_test.png")


# Load embeddings
embeddings_tensor = torch.load('optimal_mf_model_embeddings.pth')
embeddings = embeddings_tensor.detach().cpu().numpy()

# Load model information
model_info = pd.read_csv('model_order.csv', index_col=0)
model_info = model_info.sort_values(by='model_id').reset_index(drop=True)

print(embeddings.shape)
print(model_info)

keyword = "70b"
index_ls = [index for index,model in enumerate(list(model_info['model_name'])) if keyword in model.lower() ]
print(index_ls)
visualize_embeddings_pca(embeddings, index_ls, keyword)

def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def find_closest_embedding_sums(embeddings, top_n=10):
    """
    Iterate through all possible tuples of 3 embeddings (a, b, c) and find the top N tuples where normalized (a + b) 
    is closest to normalized c using cosine similarity.
    
    Parameters:
    - embeddings: numpy array of shape (n_samples, n_features) containing the embeddings.
    - top_n: int, the number of top results to return.
    
    Returns:
    - top_results: List of tuples [(i, j, k, cosine_similarity), ...] where (i, j, k) are the indices of embeddings and 
      cosine_similarity is the cosine similarity between normalized (a + b) and normalized c.
    """
    num_embeddings = embeddings.shape[0]
    similarities = []

    # Iterate over all possible triples (i, j, k)
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            for k in range(num_embeddings):
                if i != k and j != k:
                    # Compute the sum of embeddings[i] and embeddings[j]
                    sum_embedding = embeddings[i] + embeddings[j]
                    
                    # Normalize (a + b) and c
                    sum_embedding_normalized = normalize(sum_embedding)
                    c_normalized = normalize(embeddings[k])
                    
                    # Compute cosine similarity between (a + b) and c
                    similarity = cosine_similarity(sum_embedding_normalized, c_normalized)
                    
                    # Store the tuple (i, j, k) and the similarity (higher means more similar)
                    similarities.append((i, j, k, similarity))

    # Sort by cosine similarity in descending order (we want the highest similarities)
    similarities.sort(key=lambda x: -x[3])
    top_results = similarities[:top_n]
    
    return top_results

# top_results = find_closest_embedding_sums(embeddings)

# # Print the top 10 results
# for i, (a_idx, b_idx, c_idx, sim) in enumerate(top_results):
#     print(f"Rank {i + 1}: Embeddings {list(model_info['model_name'])[a_idx]}, {list(model_info['model_name'])[b_idx]}, {list(model_info['model_name'])[c_idx]} with cosine similarity {sim:.4f}")

difference_1 = normalize(embeddings[96] - embeddings[16])
difference_2 = normalize(embeddings[44] - embeddings[14])
print(cosine_similarity(difference_1, difference_2))