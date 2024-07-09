import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

np.random.seed(42)

transformed_data = pd.read_csv("transformed_responses_mf.csv")
print(transformed_data.head())

# Get unique question_id values
unique_question_ids = transformed_data['prompt_id'].unique()

# Shuffle and split question_ids
shuffled_question_ids = np.random.permutation(unique_question_ids)
val_question_ids = shuffled_question_ids[:3000]
test_question_ids = shuffled_question_ids[3000:6000]
train_question_ids = shuffled_question_ids[6000:]

# Split the data based on the question_id splits
val_data = transformed_data[transformed_data['prompt_id'].isin(val_question_ids)]
test_data = transformed_data[transformed_data['prompt_id'].isin(test_question_ids)]
train_data = transformed_data[transformed_data['prompt_id'].isin(train_question_ids)]
# pc_train_data = transformed_data[transformed_data['prompt_id'].isin(pc_train_question_ids)]
# pc_train_val_data = transformed_data[transformed_data['prompt_id'].isin(pc_train_val_question_ids)]

# Print out the number of unique prompt IDs available in each category
categories = train_data['category'].unique()
print(f"Number of unique categories: {len(categories)}")
print("Number of unique prompt IDs in each category:")
for category in categories:
    category_data = train_data[train_data['category'] == category]
    unique_prompt_ids = category_data['prompt_id'].nunique()
    print(f"Category '{category}': {unique_prompt_ids} unique prompt IDs")

# Define the number of samples (K) to be selected from each category
K = 40

# Create a reference set by sampling K prompt_ids from each category
reference_set = pd.DataFrame()

# Sample K prompt_ids from each category and create the reference set
for category in categories:
    category_data = train_data[train_data['category'] == category]
    sampled_prompt_ids = category_data['prompt_id'].sample(n=K, random_state=42)
    reference_set = pd.concat([reference_set, category_data[category_data['prompt_id'].isin(sampled_prompt_ids)]], ignore_index=True)

print("Reference Set:")
print(reference_set.head())
print(reference_set.shape)
reference_set.to_csv("reference_set_evenly_from_benchmark.csv", index=False)

# # Save the splits into separate CSV files
# # val_data.to_csv("new_validation_set.csv", index=False)
# # test_data.to_csv("new_test_set.csv", index=False)
# # train_data.to_csv("new_train_set.csv", index=False)
# pc_train_data.to_csv("pc_new_train_set.csv", index=False)
# pc_train_val_data.to_csv("pc_new_train_val_set.csv", index=False)

# # Print the number of rows in each split to verify
# # print(f"Training set size: {train_data.shape}")
# # print(f"Validation set size: {val_data.shape}")
# # print(f"Test set size: {test_data.shape}")
# print(f"PC Training set size: {pc_train_data.shape}")
# print(f"PC Validation set size: {pc_train_val_data.shape}")