import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

# Load embeddings
embeddings_tensor = torch.load('optimal_mf_model_embeddings.pth')
embeddings = embeddings_tensor.detach().cpu().numpy()

# Load model information
model_info = pd.read_csv('model_order.csv', index_col=0)
model_info = model_info.sort_values(by='model_id').reset_index(drop=True)

# Compute pairwise L2 distances
l2_distances = squareform(pdist(embeddings, metric='euclidean'))

# Create a DataFrame for the L2 distance matrix
model_names = model_info['model_name'].tolist()
distance_df = pd.DataFrame(l2_distances, index=model_names, columns=model_names)

community_name = "math"
community_models = [name for name in model_names if (community_name in name.lower() and '7b' in name.lower())]
other_models = [name for name in model_names if name not in community_models]
print(f"Community Selected: {community_name}")
print(f"Models within Community: {community_models}")
# # Step 1: Calculate average L2 distance between models with "community" in their name
# community_models = [name for name in model_names if community_name in name.lower()]
# print(community_models)
# community_pairs = []

# for i in range(len(community_models)):
#     for j in range(i + 1, len(community_models)):
#         model_1, model_2 = community_models[i], community_models[j]
#         distance = distance_df.loc[model_1, model_2]
#         community_pairs.append(distance)

# if len(community_pairs) > 0:
#     avg_community_distance = np.mean(community_pairs)
#     print(f"Average L2 distance between community models: {avg_community_distance}")
# else:
#     print("Not enough community models to calculate distances.")

# # Step 2: Calculate average L2 distance between any two random models
# random_pairs = []

# for i in range(len(model_names)):
#     for j in range(i + 1, len(model_names)):
#         model_1, model_2 = model_names[i], model_names[j]
#         distance = distance_df.loc[model_1, model_2]
#         random_pairs.append(distance)

# avg_random_distance = np.mean(random_pairs)
# print(f"Average L2 distance between random models: {avg_random_distance}")

# # Step 3: Compare the two
# if len(community_pairs) > 0:
#     if avg_community_distance < avg_random_distance:
#         print("Community models are closer to each other than random models on average.")
#     else:
#         print("Community models are not closer to each other than random models on average.")

# Intra-community (within "community" models)
intra_community_distances = []
for i in range(len(community_models)):
    for j in range(i + 1, len(community_models)):
        intra_community_distances.append(distance_df.loc[community_models[i], community_models[j]])

# Inter-community ("community" models vs other models)
inter_community_distances = []
for community_model in community_models:
    for other_model in other_models:
        inter_community_distances.append(distance_df.loc[community_model, other_model])

# Calculate averages
avg_intra_community = np.mean(intra_community_distances)
avg_inter_community = np.mean(inter_community_distances)

print(f"Average intra-community L2 distance within 'community' models: {avg_intra_community}")
print(f"Average inter-community L2 distance between 'community' models and others: {avg_inter_community}")

if avg_intra_community < avg_inter_community:
    print("Community models are closer to each other than to other models.")
else:
    print("Community models are not closer to each other than to other models.")