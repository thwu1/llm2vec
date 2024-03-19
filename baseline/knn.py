from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import torch
import json
import os
import tqdm

pwd = os.getcwd()
print(pwd)


def get_train_test():
    train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
    test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")

    train_data_group = [
        group for _, group in train_data.sort_values("model_id").groupby("model_id")
    ]
    test_data_group = [
        group for _, group in test_data.sort_values("model_id").groupby("model_id")
    ]

    return train_data_group, test_data_group


def get_input(data):
    embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
    prompt_id = data["prompt_id"]
    label = data["label"]
    return torch.tensor([embeddings[i] for i in prompt_id]).tolist(), label.tolist()


def evaluate(train_ls, test_ls):
    accu = []
    model_names = []
    for i in tqdm.tqdm(range(len(train_ls))):
        X_train, y_train = get_input(train_ls[i])
        X_test, y_test = get_input(test_ls[i])
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, y_train)
        pred_y = neigh.predict(X_test).tolist()
        bool_ls = list(map(lambda x, y: int(x == y), pred_y, y_test))
        accu.append(sum(bool_ls) / len(y_test))
        model_names.append(train_ls[i]["model_name"].iloc[0])

    for acc, model in zip(accu, model_names):
        print(f"Model: {model}, Accuracy: {acc}")
    print("Mean Accuracy:", sum(accu) / len(accu))


train_ls, test_ls = get_train_test()
evaluate(train_ls, test_ls)
