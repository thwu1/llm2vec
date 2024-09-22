import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

embeddings_tensor = torch.load('optimal_mf_model_embeddings.pth')
embeddings = embeddings_tensor.detach().cpu().numpy()
# print(embeddings.shape)

model_info = pd.read_csv('model_order.csv', index_col=0)
print(model_info.head())
model_info = model_info.sort_values(by='model_id').reset_index(drop=True)
print(model_info.head())

# similarity_matrix = cosine_similarity(embeddings)
# model_names = model_info['model_name'].tolist()
# # print(model_names)
# similarity_df = pd.DataFrame(similarity_matrix, index=model_names, columns=model_names)
# # print(similarity_df)
# # similarity_df.to_csv('mf_embedding_similarity_matrix.csv')

# def get_k_most_similar_pairs(similarity_df, k):
#     model_names = similarity_df.index.to_list()
#     pairs = []
    
#     # Iterate over the upper triangle of the matrix (excluding the diagonal)
#     for i in range(len(model_names)):
#         for j in range(i + 1, len(model_names)):
#             pairs.append((model_names[i], model_names[j], similarity_df.iat[i, j]))
    
#     # Sort the pairs by similarity value in descending order
#     pairs.sort(key=lambda x: x[2], reverse=True)
    
#     # Extract the top k pairs
#     top_k_pairs = pairs[:k]
    
#     return top_k_pairs

# for pair in get_k_most_similar_pairs(similarity_df, 50):
#     model_1, model_2, similarity = pair
#     print(f"Model 1: {model_1}, Model 2: {model_2}, Similarity: {similarity}")

# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)

# model_names = model_info['model_name'].tolist()
# colors = cm.rainbow(np.linspace(0, 1, len(model_names)))

# plt.figure(figsize=(40, 30))
# for i in range(len(model_names)):
#     plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i], label=model_names[i], marker='o')
# # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color=colors[i], label=model_names[i], marker='o')

# # Annotate each point with the model name
# for i, model_name in enumerate(model_names):
#     # if "llama" in model_name:
#     plt.annotate(model_name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize = 8)

# plt.title('PCA of Model Embeddings')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()
# plt.savefig('MF_model_embeddings_full.png', dpi=300, bbox_inches='tight')

