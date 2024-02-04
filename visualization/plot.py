from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import os
import umap

def visualize_embeddings(data_path, method="pca"):
    assert method in ["pca", "umap"]
    json_data = json.load(open(data_path, "r"))
    embeddings = []
    categories = []
    for item in json_data:
        for embed in item["embedding"]:
            embeddings.append(embed)
            categories.append(item["category"])

    X = np.array(embeddings)
    reducer = umap.UMAP() if method == "umap" else PCA(n_components=2)
    reduced_X = reducer.fit_transform(X)

    # Plot with seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=categories, palette="viridis")
    plt.title(f"{method.upper()} of Embeddings Color-coded by Category")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Category")
    plt.savefig(f"visualization/{data_path.split('/')[-1][:-5]}.png")
    # plt.show()

def compare_embeddings(data_paths, method="pca"):
    assert method in ["pca", "umap"]
    embeddings = []
    categories = []
    for data_path in data_paths:
        json_data = json.load(open(data_path, "r"))
        model_name = data_path.split("/")[-1][:-5]
        for item in json_data:
            for embed in item["embedding"]:
                embeddings.append(embed)
                categories.append(model_name)

    X = np.array(embeddings)
    reducer = umap.UMAP() if method == "umap" else PCA(n_components=2)
    reduced_X = reducer.fit_transform(X)


    # Plot with seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=categories, palette="viridis")
    plt.title(f"{method.upper()} of Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Data Path")
    plt.savefig(f"visualization/compare.png")
    # plt.show()

data_paths = os.listdir("evaluation/outputs/all-mpnet-base-v2")
filtered_data_paths = [f"evaluation/outputs/all-mpnet-base-v2/{data_path}" for data_path in data_paths if data_path.endswith("_embed.json")]

compare_embeddings(filtered_data_paths)

for data_path in filtered_data_paths:
    visualize_embeddings(data_path)