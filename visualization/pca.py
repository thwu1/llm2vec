from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import os

data_path = "/data/tianhao/llm2vec/evaluation/outputs/mt-bench-openchat_3.5_embed.json"

def visualize_embeddings(data_path):
    json_data = json.load(open(data_path, "r"))
    embeddings = []
    categories = []
    for item in json_data:
        for embed in item["embedding"]:
            embeddings.append(embed)
            categories.append(item["category"])

    X = np.array(embeddings)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    # Plot with seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=categories, palette="viridis")
    plt.title("PCA of Embeddings Color-coded by Category")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Category")
    plt.savefig(f"visualization/{data_path.split('/')[-1][:-5]}.png")
    # plt.show()

def compare_embeddings(data_paths):
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
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)


    # Plot with seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=categories, palette="viridis")
    plt.title("PCA of Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Data Path")
    plt.savefig(f"visualization/compare.png")
    # plt.show()

data_paths = os.listdir("evaluation/outputs")
filtered_data_paths = [f"evaluation/outputs/{data_path}" for data_path in data_paths if data_path.endswith("_embed.json")]

compare_embeddings(filtered_data_paths)