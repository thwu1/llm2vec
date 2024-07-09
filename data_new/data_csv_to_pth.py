import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

DATA_PATH = "evenly_by_benchmark/reference_set_evenly_from_benchmark.csv"
OUTPUT_PATH_X = "evenly_by_benchmark/reference_set_evenly_from_benchmark_x.pth"
OUTPUT_PATH_Y = "evenly_by_benchmark/reference_set_evenly_from_benchmark_y.pth"

df = pd.read_csv(DATA_PATH)

print(df.head(5))

df = df.groupby(['model_id', 'prompt_id', 'prompt']).agg({'label': 'max'}).reset_index()

print(df.head(5))

# Load a sentence transformer model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Compute the embeddings for each unique question
unique_questions = df[['prompt_id', 'prompt']].drop_duplicates()
print(unique_questions.head(5))
unique_questions['embedding'] = unique_questions['prompt'].apply(lambda x: model.encode(x))

# Merge the embeddings back to the original dataframe
df = df.merge(unique_questions[['prompt_id', 'embedding']], on='prompt_id')
print(df.head(5))

# Pivot the dataframe to create the correctness matrix
correctness_matrix = df.pivot(index='model_id', columns='prompt_id', values='label').fillna(0).astype(int)
# print(correctness_matrix)
# correctness_matrix.to_csv("correctness_matrix.csv")

# Convert the correctness matrix to a numpy array
correctness_array = correctness_matrix.values

# Stack the embeddings to create the final tensor
question_embeddings = np.stack(unique_questions.set_index('prompt_id').loc[correctness_matrix.columns]['embedding'].values)
final_tensor = np.array([question_embeddings] * correctness_array.shape[0])

# # Reshape to the desired format: (num_models, question_embedding_dim, num_questions)
# final_tensor = np.transpose(final_tensor, (0, 2, 1))

# Add correctness_array as an additional dimension
final_tensor = np.concatenate([final_tensor, correctness_array[:,:,np.newaxis]], axis=2)

# The final tensor shape is (num_models, question_embedding_dim + 1, num_questions)
# where the last dimension in question_embedding_dim + 1 is the correctness
print(f"Final tensor shape: {final_tensor.shape}")
final_tensor_x = final_tensor[:,:,:-1]
final_tensor_y = final_tensor[:,:,-1]
print(f"Final X shape: {final_tensor_x.shape}")
print(f"Final Y shape: {final_tensor_y.shape}")
# print(final_tensor_x)
# print(final_tensor_y)
torch.save(final_tensor_x, OUTPUT_PATH_X, pickle_protocol=4)
torch.save(final_tensor_y, OUTPUT_PATH_Y, pickle_protocol=4)
print("Tensor saved successfully.")