import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Target Columns: ['prompt_id', 'model_id', 'category_id', 'label', 'prompt', 'model_name','category']
# Example: 
# prompt_id                                                    983
# model_id                                                    2190
# category_id                                                   18
# label                                                          1
# prompt         Which of the following areas is not covered by...
# model_name                MBeagleX_7B+Mixtral_11Bx2_MoE_19B_vote
# category                                         business ethics

# data = pd.read_csv("transformed_responses_mf.csv")
# print(data.shape)
# raise Exception
data = pd.read_csv("all_responses.csv", index_col=0)
# print(data.head())
# Set seed for reproducibility
np.random.seed(42)

# TODO:
# Assign each model_name a model_id
# Assign each prompt a numerical prompt_id
# Column rename: correctness --> label, question_text --> prompt
# Category: "_".join(question_id.split("_")[0:-1])
# Assign each category a category_id

# Step 1: Assign each model_name a model_id
model_id_mapping = {model: idx for idx, model in enumerate(data['model_name'].unique())}
data['model_id'] = data['model_name'].map(model_id_mapping)

# Step 2: Assign each prompt a numerical prompt_id
prompt_id_mapping = {prompt: idx for idx, prompt in enumerate(data['question_text'].unique())}
data['prompt_id'] = data['question_text'].map(prompt_id_mapping)

# Step 3: Rename columns: correctness --> label, question_text --> prompt
data.rename(columns={'correctness': 'label', 'question_text': 'prompt'}, inplace=True)

# Step 4: Extract and assign category from question_id
data['category'] = data['question_id'].apply(lambda x: "_".join(x.split("_")[:-1]))
data['benchmark'] = data['question_id'].apply(lambda x: x.split("_")[0])
prompt_id_to_benchmark = data[['prompt_id', 'benchmark']].drop_duplicates().set_index('prompt_id')
prompt_id_to_benchmark.to_csv("prompt_id_to_benchmark.csv")
raise ValueError("STOP!")
# Step 5: Assign each category a category_id
category_id_mapping = {category: idx for idx, category in enumerate(data['category'].unique())}
data['category_id'] = data['category'].map(category_id_mapping)

# Select the target columns in the desired order
transformed_data = data[['prompt_id', 'model_id', 'category_id', 'label', 'prompt', 'model_name', 'category']]

# Print the first few rows of the transformed dataset to verify
print(transformed_data.head())

# Initialize the sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Get unique prompts and their IDs
# unique_prompts = transformed_data[['prompt_id', 'prompt']].drop_duplicates().set_index('prompt_id')
unique_prompts = transformed_data[['prompt_id', 'prompt']].set_index('prompt_id')
# print(unique_prompts.shape)
# raise Exception
# unique_prompts.to_csv("prompt_order.csv")

# Compute embeddings for each unique prompt
embeddings = model.encode(unique_prompts['prompt'].tolist(), convert_to_tensor=True)

# Create an embedding tensor of shape (num_questions, embedding_dim)
num_questions = len(unique_prompts)
embedding_dim = embeddings.shape[1]
embedding_tensor = torch.zeros((num_questions, embedding_dim))

# Assign embeddings to the tensor
for idx, (prompt_id, _) in enumerate(unique_prompts.iterrows()):
    embedding_tensor[prompt_id] = embeddings[idx]

# Save the embedding tensor to a file
torch.save(embedding_tensor, 'new_prompt_embeddings_36054.pth')

# Print the shape of the embedding tensor to verify
print(f"Embedding tensor shape: {embedding_tensor.shape}")

# Save the transformed dataset to a new CSV file
transformed_data.to_csv("transformed_responses_mf.csv", index=False)

# Get unique question_id values
unique_question_ids = transformed_data['prompt_id'].unique()

# Shuffle and split question_ids
shuffled_question_ids = np.random.permutation(unique_question_ids)
# val_question_ids = shuffled_question_ids[:3000]
# test_question_ids = shuffled_question_ids[3000:6000]
# train_question_ids = shuffled_question_ids[6000:]
pc_train_val_question_ids = shuffled_question_ids[6000:9000]
pc_train_question_ids = shuffled_question_ids[9000:]

# Split the data based on the question_id splits
# val_data = transformed_data[transformed_data['prompt_id'].isin(val_question_ids)]
# test_data = transformed_data[transformed_data['prompt_id'].isin(test_question_ids)]
# train_data = transformed_data[transformed_data['prompt_id'].isin(train_question_ids)]
pc_train_data = transformed_data[transformed_data['prompt_id'].isin(pc_train_question_ids)]
pc_train_val_data = transformed_data[transformed_data['prompt_id'].isin(pc_train_val_question_ids)]


# Save the splits into separate CSV files
# val_data.to_csv("new_validation_set.csv", index=False)
# test_data.to_csv("new_test_set.csv", index=False)
# train_data.to_csv("new_train_set.csv", index=False)
pc_train_data.to_csv("pc_new_train_set.csv", index=False)
pc_train_val_data.to_csv("pc_new_train_val_set.csv", index=False)

# Print the number of rows in each split to verify
# print(f"Training set size: {train_data.shape}")
# print(f"Validation set size: {val_data.shape}")
# print(f"Test set size: {test_data.shape}")
print(f"PC Training set size: {pc_train_data.shape}")
print(f"PC Validation set size: {pc_train_val_data.shape}")

