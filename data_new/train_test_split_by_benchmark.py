import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

data = pd.read_csv("all_responses.csv", index_col=0)
# data['question_source'] = data['question_id'].apply(lambda x: '_'.join(x.rsplit('_', 1)[:-1]))
data['source'] = data['question_id'].apply(lambda x: x.split('_')[0])
all_benchmarks = data['source'].unique()
print(all_benchmarks)

model_id_mapping = {model: idx for idx, model in enumerate(data['model_name'].unique())}
data['model_id'] = data['model_name'].map(model_id_mapping)
prompt_id_mapping = {prompt: idx for idx, prompt in enumerate(data['question_text'].unique())}
data['prompt_id'] = data['question_text'].map(prompt_id_mapping)
data.rename(columns={'correctness': 'label', 'question_text': 'prompt'}, inplace=True)
data['category'] = data['question_id'].apply(lambda x: "_".join(x.split("_")[:-1]))
category_id_mapping = {category: idx for idx, category in enumerate(data['category'].unique())}
data['category_id'] = data['category'].map(category_id_mapping)
transformed_data = data[['prompt_id', 'model_id', 'label', 'prompt', 'model_name', 'source', 'category', 'category_id']]

print(transformed_data.head())

model = SentenceTransformer('all-mpnet-base-v2')
unique_prompts = transformed_data[['prompt_id', 'prompt']].drop_duplicates().set_index('prompt_id')
embeddings = model.encode(unique_prompts['prompt'].tolist(), convert_to_tensor=True)

num_questions = len(unique_prompts)
embedding_dim = embeddings.shape[1]
embedding_tensor = torch.zeros((num_questions, embedding_dim))

for idx, (prompt_id, _) in enumerate(unique_prompts.iterrows()):
    embedding_tensor[prompt_id] = embeddings[idx]

torch.save(embedding_tensor, 'mf_embedding_test/prompt_embeddings.pth')
print(f"Embedding tensor shape: {embedding_tensor.shape}")
transformed_data.to_csv("mf_embedding_test/transformed_responses_mf.csv", index=False)
unique_question_ids = transformed_data['prompt_id'].unique()

train_benchmark_ls = ['gpqa', 'mmlu', 'medmcqa', 'piqa', 'asdiv', 
                      'truthfulqa', 'logiqa', 'gsm8k']
test_benchmark_ls = ['mathqa', 'social']
mf_embedding_check_train_data = transformed_data[transformed_data['source'].isin(train_benchmark_ls)]
mf_embedding_check_test_data = transformed_data[transformed_data['source'].isin(test_benchmark_ls)]


# Save the splits into separate CSV files
# val_data.to_csv("new_validation_set.csv", index=False)
# test_data.to_csv("new_test_set.csv", index=False)
# train_data.to_csv("new_train_set.csv", index=False)
mf_embedding_check_train_data.to_csv("mf_embedding_test/mf_embedding_check_train_set.csv", index=False)
mf_embedding_check_test_data.to_csv("mf_embedding_test/mf_embedding_check_test_set.csv", index=False)

# Print the number of rows in each split to verify
# print(f"Training set size: {train_data.shape}")
# print(f"Validation set size: {val_data.shape}")
# print(f"Test set size: {test_data.shape}")
print(f"PC Training set size: {mf_embedding_check_train_data.shape}")
print(f"PC Validation set size: {mf_embedding_check_test_data.shape}")
