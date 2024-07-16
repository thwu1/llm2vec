import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from itertools import combinations

def train_test_split_by_benchmark(train_benchmarks, test_benchmarks, train_output_path, test_output_path):
    data = pd.read_csv("all_responses.csv", index_col=0)
    # data['question_source'] = data['question_id'].apply(lambda x: '_'.join(x.rsplit('_', 1)[:-1]))
    data['source'] = data['question_id'].apply(lambda x: x.split('_')[0])
    all_benchmarks = data['source'].unique()
    # print(all_benchmarks)

    model_id_mapping = {model: idx for idx, model in enumerate(data['model_name'].unique())}
    data['model_id'] = data['model_name'].map(model_id_mapping)
    prompt_id_mapping = {prompt: idx for idx, prompt in enumerate(data['question_text'].unique())}
    data['prompt_id'] = data['question_text'].map(prompt_id_mapping)
    data.rename(columns={'correctness': 'label', 'question_text': 'prompt'}, inplace=True)
    data['category'] = data['question_id'].apply(lambda x: "_".join(x.split("_")[:-1]))
    category_id_mapping = {category: idx for idx, category in enumerate(data['category'].unique())}
    data['category_id'] = data['category'].map(category_id_mapping)
    transformed_data = data[['prompt_id', 'model_id', 'label', 'prompt', 'model_name', 'source', 'category', 'category_id']]

    # print(transformed_data.head())

    model = SentenceTransformer('all-mpnet-base-v2')
    unique_prompts = transformed_data[['prompt_id', 'prompt']].drop_duplicates().set_index('prompt_id')
    embeddings = model.encode(unique_prompts['prompt'].tolist(), convert_to_tensor=True)

    num_questions = len(unique_prompts)
    embedding_dim = embeddings.shape[1]
    embedding_tensor = torch.zeros((num_questions, embedding_dim))

    for idx, (prompt_id, _) in enumerate(unique_prompts.iterrows()):
        embedding_tensor[prompt_id] = embeddings[idx]

    # torch.save(embedding_tensor, 'mf_embedding_test/prompt_embeddings.pth')
    # print(f"Embedding tensor shape: {embedding_tensor.shape}")
    # transformed_data.to_csv("mf_embedding_test/transformed_responses_mf.csv", index=False)
    unique_question_ids = transformed_data['prompt_id'].unique()

    mf_embedding_check_train_data = transformed_data[transformed_data['source'].isin(train_benchmarks)]
    mf_embedding_check_test_data = transformed_data[transformed_data['source'].isin(test_benchmarks)]

    # Save the splits into separate CSV files
    # val_data.to_csv("new_validation_set.csv", index=False)
    # test_data.to_csv("new_test_set.csv", index=False)
    # train_data.to_csv("new_train_set.csv", index=False)
    mf_embedding_check_train_data.to_csv(train_output_path, index=False)
    mf_embedding_check_test_data.to_csv(test_output_path, index=False)

    # Print the number of rows in each split to verify
    # print(f"Training set size: {train_data.shape}")
    # print(f"Validation set size: {val_data.shape}")
    # print(f"Test set size: {test_data.shape}")
    print(f"PC Training set size: {mf_embedding_check_train_data.shape}")
    print(f"PC Validation set size: {mf_embedding_check_test_data.shape}")
    print(f"Result saved as {train_output_path} and {test_output_path}\n\n")

if __name__ == "__main__":
    ALL_BENCHMARK = ['medmcqa','piqa','asdiv','logiqa','truthfulqa','mathqa','mmlu','gsm8k','gpqa','social']

    for i, test_benchmark in enumerate(ALL_BENCHMARK):
        train_benchmarks = ALL_BENCHMARK[:i] + ALL_BENCHMARK[i+1:]
        test_benchmarks = [test_benchmark]
        print(f"Train Benchmarks: {train_benchmarks}")
        print(f"Test Benchmarks: {test_benchmarks}")

        train_output_path = f"mf_embedding_test/for_paper/loo_{test_benchmark}_train.csv"
        test_output_path = f"mf_embedding_test/for_paper/loo_{test_benchmark}_test.csv"
        # print(train_output_path)
        # print(test_output_path)
        # train_test_split_by_benchmark(train_benchmarks, test_benchmarks, train_output_path, test_output_path)

    comb = combinations(ALL_BENCHMARK, 2)
    print(len(list(comb)))
    for pair in comb:
        test_benchmarks = list(pair)
        train_benchmarks = [item for item in ALL_BENCHMARK if item not in pair]
        print(f"Train Benchmarks: {train_benchmarks}")
        print(f"Test Benchmarks: {test_benchmarks}")

        train_output_path = f"mf_embedding_test/for_paper/loo_{pair[0]}_{pair[1]}_train.csv"
        test_output_path = f"mf_embedding_test/for_paper/loo_{pair[0]}_{pair[1]}_test.csv"
        # print(train_output_path)
        # print(test_output_path)
        # print()
        # train_test_split_by_benchmark(train_benchmarks, test_benchmarks, train_output_path, test_output_path)

    # TRAIN_BENCHMARK_LS = ['medmcqa','piqa','asdiv','logiqa','mmlu','gpqa','social',]
    # TEST_BENCHMARK_LS = ['truthfulqa','mathqa',]
    # TRAIN_DATA_OUTPUT_PATH = "mf_embedding_test/for_paper/loo_truthfulqa_mathqa_train.csv"
    # TEST_DATA_OUTPUT_PATH = "mf_embedding_test/for_paper/loo_truthfulqa_mathqa_test.csv"