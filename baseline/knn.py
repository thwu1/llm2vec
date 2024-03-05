from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
import torch
import json


MODEL_NAMES = [
    "vicuna_13b_v1.5",
    "vicuna_33b_v1.3",
    "tulu_30b",
    "SOLAR_10.7B_Instruct_v1.0",
    "Qwen_14B_Chat",
    "Llama_2_13b_chat_hf",
    "Mistral_7B_v0.1",
    "zephyr_7b_beta",
    "vicuna_7b_v1.5",
    "Llama_2_7b_chat_hf",
    "Starling_LM_7B_alpha",
    "baize_v2_13b",
    "Yi_34B_Chat",
    "koala_13B_HF",
    "mpt_7b_chat",
    "dolly_v2_12b",
    "stablelm_tuned_alpha_7b",
    "Orca_2_13b",
    "vicuna_7b_v1.5_16k",
    "openchat_3.5",
    "WizardLM_13B_V1.2",
    "openchat_3.5_0106",
    "Nous_Hermes_13b",
    "LlamaGuard_7b",
]


def read_data():
    data_dir = "/home/thw/llm2vec"
    # names = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv(os.path.join(data_dir, "triple_correctness.csv"), engine="python")
    num_models = data.model_id.unique().shape[0]
    num_prompts = data.prompt_id.unique().shape[0]
    return data, num_models, num_prompts


def train_test_split(test_ratio=0.1, mode="random"):
    assert mode in ["random", "ood"]
    data, num_models, num_prompts = read_data()
    print(data)
    prompt_id = data["prompt_id"]

    if mode == "random":
        # Group the data by model_id
        grouped_data = [group for _, group in data.groupby("model_id")]

        train_data_list = []
        test_data_list = []

        for group in grouped_data:
            # Shuffle the data within each group
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)

            # Calculate the number of samples for train and test sets
            num_test = int(len(group) * test_ratio)
            num_train = len(group) - num_test

            # Split the data into train and test sets
            train_data = group.iloc[:num_train]
            test_data = group.iloc[num_train:]

            train_data_list.append(train_data)
            test_data_list.append(test_data)

        return train_data_list, test_data_list, num_models, num_prompts


train_ls, test_ls, _, _ = train_test_split()


def get_input(data):
    embeddings = json.load(open("/home/thw/llm2vec/embeddings.json"))
    prompt_id = data["prompt_id"]
    label = data["label"]
    return torch.tensor([embeddings[i] for i in prompt_id]).tolist(), label.tolist()


def evaluate(train_ls, test_ls):
    accu = []
    for i in range(len(train_ls)):
        X_train, y_train = get_input(train_ls[i])
        X_test, y_test = get_input(test_ls[i])
        neigh = KNeighborsClassifier(n_neighbors=21)
        neigh.fit(X_train, y_train)
        pred_y = neigh.predict(X_test).tolist()
        bool_ls = list(map(lambda x, y: int(x == y), pred_y, y_test))
        accu.append(sum(bool_ls) / len(y_test))

    for acc, model in zip(accu, MODEL_NAMES):
        print(f"Model: {model}, Accuracy: {acc}")
    print("Mean Accuracy:", sum(accu) / len(accu))


evaluate(train_ls, test_ls)
