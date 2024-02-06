from sklearn.metrics import pairwise_distances
import numpy as np
import os
import json
import argparse

def in_class_dis(X, metric="cosine"):
    distances = pairwise_distances(X, metric=metric)
    return np.sum(distances) / (distances.shape[0] * (distances.shape[0] - 1))

def between_class_dis(X,Y, metric="cosine"):
    distances = pairwise_distances(X,Y, metric=metric)
    return np.mean(distances)

def get_stats(dp1, dp2, metric="cosine"):
    json_data1 = json.load(open(dp1, "r"))
    json_data2 = json.load(open(dp2, "r"))
    in_class_dis1 = []
    in_class_dis2 = []
    between_class_dis_ = []
    for item1, item2 in zip(json_data1,json_data2):
        assert item1["prompt"] == item2["prompt"]
        X = np.array(item1["embedding"])
        Y = np.array(item2["embedding"])
        in_class_dis1.append(in_class_dis(X, metric=metric))
        in_class_dis2.append(in_class_dis(Y, metric=metric))
        between_class_dis_.append(between_class_dis(X,Y, metric=metric))
    print(f"Average in-class distance for {dp1}: {np.mean(in_class_dis1)}")
    print(f"Average in-class distance for {dp2}: {np.mean(in_class_dis2)}")
    print(f"Average between-class distance: {np.mean(between_class_dis_)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="evaluation/outputs/all-mpnet-base-v2")
    parser.add_argument("--metric", type=str, default="cosine")
    args = parser.parse_args()

    data_paths = os.listdir(args.dir)
    filtered_data_paths = [f"{args.dir}/{data_path}" for data_path in data_paths if data_path.endswith("_embed.json")]

    for i in range(len(filtered_data_paths)):
        for j in range(i+1, len(filtered_data_paths)):
            get_stats(filtered_data_paths[i], filtered_data_paths[j], metric=args.metric)