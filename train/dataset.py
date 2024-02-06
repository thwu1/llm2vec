import json
import torch
import os
import numpy as np
import time
from tqdm import trange
import random
from torch.utils.data import random_split, DataLoader

torch.manual_seed(42)

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        data_ls = os.listdir(dir)
        self.label_name = {idx: name for idx, name in enumerate(data_ls)}
        self.label_num = {name: idx for idx, name in enumerate(data_ls)}
        self.data = self._merge_data(dir, data_ls)
        self.n_responses = len(self.data[0]["embedding"][0])

    def _merge_data(self, dir, data_ls):
        data = []
        first_data = json.load(open(os.path.join(dir, data_ls[0]), "r"))
        for dict in first_data:
            data.append({"question_id": dict["question_id"], "embedding": [dict["embedding"]]})
        for path in data_ls[1:]:
            new_data = json.load(open(os.path.join(dir, path), "r"))
            for dict1, dict2 in zip(data, new_data):
                assert dict1["question_id"] == dict2["question_id"]
                dict1["embedding"].append(dict2["embedding"])

        for dict in data:
            dict["embedding"] = torch.tensor(dict["embedding"])
        return data
    
    def get_data(self):
        return self.data

    def get_label_name(self):
        return self.label_name

    # def _sample_unique_pairs(self, max_idx, n_pairs):
    #     a1 = np.random.randint(0, max_idx, size=n_pairs)
    #     a2 = np.random.randint(0, max_idx - 1, size=n_pairs)

    #     a2 = np.where(a2 >= a1, a2 + 1, a2)
    #     return np.stack((a1, a2), axis=1).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indices = random.sample(range(self.n_responses), 2)
        return {
            0: self.data[idx]["embedding"][:, indices[0], :],
            1: self.data[idx]["embedding"][:, indices[1], :],
        }

    def get_dataloader(self, batch_size=32, shuffle=True, test_split=0.1):
        # Calculate split sizes
        total_size = len(self)
        test_size = int(total_size * test_split)
        train_size = total_size - test_size
        
        # Split the dataset
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        print("len(train_dataset):{}, len(test_dataset):{}".format(len(train_dataset), len(test_dataset)))
        def custom_collate_fn(batch):
            # Stack the items for each key, resulting in shape [bs, 12, 768] for both keys 0 and 1
            stacked_0 = torch.stack([item[0] for item in batch])
            stacked_1 = torch.stack([item[1] for item in batch])

            # Transpose the stacked tensors to get shape [12, bs, 768]
            transposed_0 = stacked_0.transpose(0, 1)
            transposed_1 = stacked_1.transpose(0, 1)
            # print(batch[3][0][6])
            # print(transposed_0[6][3])
            return [transposed_0, transposed_1]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
        return train_loader, test_loader


# dataset = ContrastiveDataset("evaluation/outputs/all-mpnet-base-v2")
# print(len(dataset))

# dataloader = dataset.get_dataloader()

# for batch in dataloader:
#     print(batch[0].shape)
#     print(batch[1].shape)
#     break
