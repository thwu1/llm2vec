from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import os
import pickle
import torch
import numpy as np
import json

pwd = os.getcwd()

LABEL_A = 0
LABEL_B = 1
LABEL_C = 2
LABEL_D = 3


def convert_to_label(output):
    if isinstance(output, int):
        return output
    if output == "A":
        return int(LABEL_A)
    elif output == "B":
        return int(LABEL_B)
    elif output == "C":
        return int(LABEL_C)
    elif output == "D":
        return int(LABEL_D)
    else:
        raise ValueError(f"Output {output} not recognized")


DATASET_NAME = "RZ412/mmlu_responses_1k_augmented"

data = load_dataset(DATASET_NAME)["train"]
# print(data[0])
categories = [item["subject"] for item in data]
prompts = [item["test_questions"] for item in data]
# print(data)
CATEGORIES = list(set(categories))
MODEL_NAME = [item["model"] for item in data[0]["answers"]]


def category_to_label(category):
    return CATEGORIES.index(category)


def model_to_label(model):
    return MODEL_NAME.index(model)


# get the matrix of correctness
correctness = []
for item in data:
    # print(item)
    assert all(
        [
            item["answers"][id]["model"] == model_name
            for id, model_name in enumerate(MODEL_NAME)
        ]
    )
    correctness.append(
        [
            convert_to_label(item["answers"][id]["answer"])
            == convert_to_label(item["reference_answers"])
            for id, model_name in enumerate(MODEL_NAME)
        ]
    )

# use df to save the correctness matrix
df = pd.DataFrame(correctness, columns=MODEL_NAME).astype(int)
df.to_csv(f"{pwd}/data/mmlu_correctness_1k.csv", index=False)


def train_val_test_split(test_ratio=0.05, val_ratio=0.05, seed=42):
    df = pd.read_csv(f"{pwd}/data/mmlu_correctness_1k.csv")
    df = df.sample(frac=1, random_state=seed).reset_index()

    # Calculate split indices
    test_index = int(len(df) * (1 - test_ratio))
    val_index = int(test_index * (1 - val_ratio / (1 - test_ratio)))

    # Split the DataFrame into train and test
    train_df = df.iloc[:val_index].reset_index(drop=True)
    val_df = df.iloc[val_index:test_index].reset_index(drop=True)
    test_df = df.iloc[test_index:].reset_index(drop=True)

    model_column_indices = [
        train_df.columns.get_loc(col) for col in train_df.columns[1:]
    ]

    # Convert train_df and test_df to tuples
    train_tuples = [
        (row[1][0], train_df.columns[col_idx], row[1][col_idx])
        for row in train_df.iterrows()
        for col_idx in model_column_indices
    ]
    val_tuples = [
        (row[1][0], val_df.columns[col_idx], row[1][col_idx])
        for row in val_df.iterrows()
        for col_idx in model_column_indices
    ]
    test_tuples = [
        (row[1][0], test_df.columns[col_idx], row[1][col_idx])
        for row in test_df.iterrows()
        for col_idx in model_column_indices
    ]

    train_df_tuples = pd.DataFrame(
        train_tuples, columns=["prompt_id", "model_name", "label"]
    )
    val_df_tuples = pd.DataFrame(
        val_tuples, columns=["prompt_id", "model_name", "label"]
    )
    test_df_tuples = pd.DataFrame(
        test_tuples, columns=["prompt_id", "model_name", "label"]
    )

    return train_df_tuples, val_df_tuples, test_df_tuples


# print(train_val_test_split())

train_df_tuples, val_df_tuples, test_df_tuples = train_val_test_split()
# add the column record the category of the prompt
train_df_tuples["category"] = [categories[i] for i in train_df_tuples["prompt_id"]]
val_df_tuples["category"] = [categories[i] for i in val_df_tuples["prompt_id"]]
test_df_tuples["category"] = [categories[i] for i in test_df_tuples["prompt_id"]]

train_df_tuples["category_id"] = [
    category_to_label(category) for category in train_df_tuples["category"]
]
val_df_tuples["category_id"] = [
    category_to_label(category) for category in val_df_tuples["category"]
]
test_df_tuples["category_id"] = [
    category_to_label(category) for category in test_df_tuples["category"]
]

train_df_tuples["model_id"] = [
    model_to_label(model) for model in train_df_tuples["model_name"]
]
val_df_tuples["model_id"] = [
    model_to_label(model) for model in val_df_tuples["model_name"]
]
test_df_tuples["model_id"] = [
    model_to_label(model) for model in test_df_tuples["model_name"]
]

train_df_tuples["prompt"] = [prompts[i] for i in train_df_tuples["prompt_id"]]
val_df_tuples["prompt"] = [prompts[i] for i in val_df_tuples["prompt_id"]]
test_df_tuples["prompt"] = [prompts[i] for i in test_df_tuples["prompt_id"]]

col_order = [
    "prompt_id",
    "model_id",
    "category_id",
    "label",
    "prompt",
    "model_name",
    "category",
]

train_df_tuples.sample(frac=1, random_state=42).reindex(columns=col_order).to_csv(
    f"{pwd}/data/mmlu_train.csv", index=False
)
val_df_tuples.sample(frac=1, random_state=42).reindex(columns=col_order).to_csv(
    f"{pwd}/data/mmlu_val.csv", index=False
)
test_df_tuples.sort_values(by="prompt_id").reindex(columns=col_order).to_csv(
    f"{pwd}/data/mmlu_test.csv", index=False
)

def store_train_val_test_split(base_model_only, sentence_transformer="all-mpnet-base-v2"):
    pwd = os.getcwd()

    train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
    val_data = pd.read_csv(f"{pwd}/data/mmlu_val.csv")
    test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")

    if base_model_only:
        train_data = train_data[
            ~train_data["model_name"].str.contains("vote|moe")
        ].reset_index(drop=True)
        val_data = val_data[
            ~val_data["model_name"].str.contains("vote|moe")
        ].reset_index(drop=True)
        test_data = test_data[
            ~test_data["model_name"].str.contains("vote|moe")
        ].reset_index(drop=True)

    train_data = train_data.sort_values(["model_id","prompt_id"])
    val_data = val_data.sort_values(["model_id","prompt_id"])
    test_data = test_data.sort_values(["model_id","prompt_id"])

    model_order = list(test_data['model_name'].unique())
    train_prompt_order = list(train_data[train_data['model_name']==model_order[0]]['prompt'])
    val_prompt_order = list(val_data[val_data['model_name']==model_order[0]]['prompt'])
    test_prompt_order = list(test_data[test_data['model_name']==model_order[0]]['prompt'])

    model = SentenceTransformer(sentence_transformer)

    train_question_vectors = model.encode([prompt.replace("Answer:", "").strip("\n") for prompt in train_prompt_order], show_progress_bar=True)
    train_x = torch.tensor(train_question_vectors)
    train_x = train_x.unsqueeze(0).repeat(len(model_order), 1, 1)
    train_y = torch.tensor(list(train_data['label']))
    train_y = train_y.reshape(len(model_order),len(train_prompt_order))

    val_question_vectors = model.encode([prompt.replace("Answer:", "").strip("\n") for prompt in val_prompt_order], show_progress_bar=True)
    val_x = torch.tensor(val_question_vectors)
    val_x = val_x.unsqueeze(0).repeat(len(model_order), 1, 1)
    val_y = torch.tensor(list(val_data['label']))
    val_y = val_y.reshape(len(model_order),len(val_prompt_order))

    test_question_vectors = model.encode([prompt.replace("Answer:", "").strip("\n") for prompt in test_prompt_order], show_progress_bar=True)
    test_x = torch.tensor(test_question_vectors)
    test_x = test_x.unsqueeze(0).repeat(len(model_order), 1, 1)
    test_y = torch.tensor(list(test_data['label']))
    test_y = test_y.reshape(len(model_order),len(test_prompt_order))

    # Storing
    if base_model_only:
        with open(f'{pwd}/data/model_order_base_only.pkl', 'wb') as file:
            pickle.dump(model_order, file)
        with open(f'{pwd}/data/train_prompt_base_only.pkl', 'wb') as file:
            pickle.dump(train_prompt_order, file)
        with open(f'{pwd}/data/val_prompt_base_only.pkl', 'wb') as file:
            pickle.dump(train_prompt_order, file)
        with open(f'{pwd}/data/test_prompt_base_only.pkl', 'wb') as file:
            pickle.dump(test_prompt_order, file)
        torch.save(train_x, f'{pwd}/data/train_x_base_only.pth')
        torch.save(train_y, f'{pwd}/data/train_y_base_only.pth')
        torch.save(val_x, f'{pwd}/data/val_x_base_only.pth')
        torch.save(val_y, f'{pwd}/data/val_y_base_only.pth')
        torch.save(test_x, f'{pwd}/data/test_x_base_only.pth')
        torch.save(test_y, f'{pwd}/data/test_y_base_only.pth')
    else:
        with open(f'{pwd}/data/model_order_full.pkl', 'wb') as file:
            pickle.dump(model_order, file)
        with open(f'{pwd}/data/train_prompt_full.pkl', 'wb') as file:
            pickle.dump(train_prompt_order, file)
        with open(f'{pwd}/data/val_prompt_full.pkl', 'wb') as file:
            pickle.dump(train_prompt_order, file)
        with open(f'{pwd}/data/test_prompt_full.pkl', 'wb') as file:
            pickle.dump(test_prompt_order, file)
        torch.save(train_x, f'{pwd}/data/train_x_full.pth')
        torch.save(train_y, f'{pwd}/data/train_y_full.pth')
        torch.save(val_x, f'{pwd}/data/val_x_full.pth')
        torch.save(val_y, f'{pwd}/data/val_y_full.pth')
        torch.save(test_x, f'{pwd}/data/test_x_full.pth')
        torch.save(test_y, f'{pwd}/data/test_y_full.pth')

    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
    return model_order, train_prompt_order, val_prompt_order, test_prompt_order, \
           train_x, train_y, val_x, val_y, test_x, test_y

model_order, train_prompt_order, val_prompt_order, test_prompt_order, \
           train_x, train_y, val_x, val_y, test_x, test_y = store_train_val_test_split(base_model_only=True)
store_train_val_test_split(base_model_only=False)

# precompute the embeddings
with open('data/mmlu_train.csv') as f:
    train_data = pd.read_csv(f)

with open('data/mmlu_test.csv') as f:
    test_data = pd.read_csv(f)

with open('data/mmlu_val.csv') as f:
    val_data = pd.read_csv(f)

concat_data = pd.concat([train_data, test_data, val_data])

d = {prompt_id: prompt for prompt_id, prompt in zip(concat_data["prompt_id"], concat_data["prompt"])}
ls = [d[prompt_id] for prompt_id in range(1000)]
embedder = SentenceTransformer("all-mpnet-base-v2")
embeddings = torch.tensor(embedder.encode(ls), dtype=torch.float32)
torch.save(embeddings, 'data/prompt_embeddings.pth')
