from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import torch
import json
import os
from tqdm import tqdm
import random
from sentence_transformers import SentenceTransformer
import numpy as np

# pwd = os.getcwd()
# print(pwd)


# def get_train_test(base_model_only):
#     train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
#     test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")
    
#     if base_model_only:
#         train_data = train_data[
#             ~train_data["model_name"].str.contains("vote|moe")
#         ].reset_index(drop=True)
#         test_data = test_data[
#             ~test_data["model_name"].str.contains("vote|moe")
#         ].reset_index(drop=True)

#     train_data_group = [
#         group for _, group in train_data.sort_values("model_id").groupby("model_id")
#     ]
#     test_data_group = [
#         group for _, group in test_data.sort_values("model_id").groupby("model_id")
#     ]

#     return train_data_group, test_data_group

def load_train_test_split(base_model_only, sentence_transformer):
    pwd = os.getcwd()

    train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
    test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")

    if base_model_only:
        train_data = train_data[
            ~train_data["model_name"].str.contains("vote|moe")
        ].reset_index(drop=True)
        test_data = test_data[
            ~test_data["model_name"].str.contains("vote|moe")
        ].reset_index(drop=True)

    train_data = train_data.sort_values(["model_id","prompt_id"])
    test_data = test_data.sort_values(["model_id","prompt_id"])

    model_order = list(test_data['model_name'].unique())
    train_prompt_order = list(train_data[train_data['model_name']==model_order[0]]['prompt'])
    test_prompt_order = list(test_data[test_data['model_name']==model_order[0]]['prompt'])

    model = SentenceTransformer(sentence_transformer)

    train_question_vectors = model.encode(train_prompt_order, show_progress_bar=True)
    train_x = torch.tensor(train_question_vectors)
    train_x = train_x.unsqueeze(0).repeat(len(model_order), 1, 1)
    train_y = torch.tensor(list(train_data['label']))
    train_y = train_y.reshape(len(model_order),len(train_prompt_order))

    test_question_vectors = model.encode(test_prompt_order, show_progress_bar=True)
    test_x = torch.tensor(test_question_vectors)
    test_x = test_x.unsqueeze(0).repeat(len(model_order), 1, 1)
    test_y = torch.tensor(list(test_data['label']))
    test_y = test_y.reshape(len(model_order),len(test_prompt_order))

    return model_order, train_prompt_order, test_prompt_order, train_x, test_x, train_y, test_y

# def get_input(data):
#     embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
#     prompt_id = data["prompt_id"]
#     label = data["label"]
#     return torch.tensor([embeddings[i] for i in prompt_id]).tolist(), label.tolist()


def evaluate(model_order, train_x, test_x, train_y, test_y, num_neighbors):
    # random.shuffle(train_ls)
    accu = []
    num_model, num_train_questions, q_embed_dim = train_x.shape
    NUM_NEIGHBORS = num_neighbors
    for i in tqdm(range(num_model)):
        X_train, y_train = train_x[i, :, :].tolist(), train_y[i, :].tolist()
        X_test, y_test = test_x[i, :, :].tolist(), test_y[i, :].tolist()
        neigh = KNeighborsClassifier(n_neighbors=NUM_NEIGHBORS)
        neigh.fit(X_train, y_train)
        pred_y = neigh.predict(X_test).tolist()
        bool_ls = list(map(lambda x, y: int(x == y), pred_y, y_test))
        accu.append(sum(bool_ls) / len(y_test))
        # if i % 100 == 0:
        #     print(f"Mean Accuracy for {i} models:", sum(accu) / len(accu))

    # for acc, model in zip(accu, model_order):
    #     print(f"Model: {model}, Accuracy: {acc}")
        
    print(f"Mean Test Accuracy for {num_neighbors} neighbors:", sum(accu) / len(accu))

BASE_MODEL_ONLY = True
SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
NUM_NEIGHBORS = 51
neighbor_sizes = range(45,60)

model_order, train_prompt_order, test_prompt_order,\
    train_x, test_x, train_y, test_y = load_train_test_split(base_model_only=BASE_MODEL_ONLY, sentence_transformer=SENTENCE_TRANSFORMER)

for num_neighbors in neighbor_sizes:
    evaluate(model_order, train_x, test_x, train_y, test_y, num_neighbors)
