from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import os

# import numpy as np
# import json

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


data = load_dataset("RZ412/mmlu_responses_1k")["train"]
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
    assert all([item["answers"][id]["model"] == model_name for id, model_name in enumerate(MODEL_NAME)])
    correctness.append(
        [convert_to_label(item["answers"][id]["answer"]) == convert_to_label(item["reference_answers"]) for id, model_name in enumerate(MODEL_NAME)]
    )

# use df to save the correctness matrix
df = pd.DataFrame(correctness, columns=MODEL_NAME).astype(int)
df.to_csv(f"{pwd}/data/mmlu_correctness_1k.csv", index=False)


def train_test_split(test_ratio=0.1, seed=42):
    df = pd.read_csv(f"{pwd}/data/mmlu_correctness_1k.csv")
    df = df.sample(frac=1, random_state=seed).reset_index()

    # Calculate the split index
    split_index = int(len(df) * (1 - test_ratio))

    # Split the DataFrame into train and test
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    model_column_indices = [train_df.columns.get_loc(col) for col in train_df.columns[1:]]

    # Convert train_df and test_df to tuples
    train_tuples = [(row[1][0], train_df.columns[col_idx], row[1][col_idx]) for row in train_df.iterrows() for col_idx in model_column_indices]
    test_tuples = [(row[1][0], test_df.columns[col_idx], row[1][col_idx]) for row in test_df.iterrows() for col_idx in model_column_indices]

    train_df_tuples = pd.DataFrame(train_tuples, columns=["prompt_id", "model_name", "label"])
    test_df_tuples = pd.DataFrame(test_tuples, columns=["prompt_id", "model_name", "label"])

    return train_df_tuples, test_df_tuples


print(train_test_split())

train_df_tuples, test_df_tuples = train_test_split()
# add the column record the category of the prompt
train_df_tuples["category"] = [categories[i] for i in train_df_tuples["prompt_id"]]
test_df_tuples["category"] = [categories[i] for i in test_df_tuples["prompt_id"]]

train_df_tuples["category_id"] = [category_to_label(category) for category in train_df_tuples["category"]]
test_df_tuples["category_id"] = [category_to_label(category) for category in test_df_tuples["category"]]

train_df_tuples["model_id"] = [model_to_label(model) for model in train_df_tuples["model_name"]]
test_df_tuples["model_id"] = [model_to_label(model) for model in test_df_tuples["model_name"]]

train_df_tuples["prompt"] = [prompts[i] for i in train_df_tuples["prompt_id"]]
test_df_tuples["prompt"] = [prompts[i] for i in test_df_tuples["prompt_id"]]

col_order = ["prompt_id", "model_id", "category_id", "label", "prompt", "model_name", "category"]

train_df_tuples.sample(frac=1, random_state=42).reindex(columns=col_order).to_csv(f"{pwd}/data/mmlu_train.csv", index=False)
test_df_tuples.sort_values(by="prompt_id").reindex(columns=col_order).to_csv(f"{pwd}/data/mmlu_test.csv", index=False)

# precompute the embeddings
model = SentenceTransformer("all-mpnet-base-v2")
cleaned_prompts = [prompt.replace("Answer:", "").strip("\n") for prompt in prompts]
embeddings = model.encode(cleaned_prompts)
json.dump(embeddings.tolist(), open(f"{pwd}/data/embeddings.json", "w"))
