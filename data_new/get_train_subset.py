import random
import pandas as pd
import numpy as np

def sample_subset(df, subset_size):
    unique_ids = df['prompt_id'].unique()
    np.random.shuffle(unique_ids)
    print(len(unique_ids))
    print(unique_ids[0:50])
    selected_ids = unique_ids[:subset_size]
    # print(selected_ids)
    subset_df = df[df['prompt_id'].isin(selected_ids)]
    
    return subset_df

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    TRAIN_SET_PATH = "new_train_set.csv" # 29673 prompts
    SUBSET_SIZE = 25000
    OUTPUT_SUBSET_PATH = "new_train_subset_25k.csv"
    train_set = pd.read_csv(TRAIN_SET_PATH)
    # print(train_set.head())
    # print(len(list(train_set['prompt_id'].unique())))
    train_subset = sample_subset(train_set, SUBSET_SIZE)
    train_subset.to_csv(OUTPUT_SUBSET_PATH)


