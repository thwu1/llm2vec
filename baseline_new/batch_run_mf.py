import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import time
from torch.optim import Adam, SGD
from tqdm import tqdm
import wandb
import json
import random
import argparse
from ray.train import Checkpoint
import ray

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

pwd = os.getcwd()

def split_and_load(global_train_data, global_test_data, batch_size=64, subset_size=None, base_model_only=False):
    # print(train_data.head())
    # print(test_data.head())

    if base_model_only:
        train_data = global_train_data[~global_train_data["model_name"].str.contains("vote|moe")].reset_index(drop=True)
        test_data = global_test_data[~global_test_data["model_name"].str.contains("vote|moe")].reset_index(drop=True)
    elif subset_size:
        model_subset = random.sample(list(test_data["model_name"]), subset_size)
        # print(model_subset[:10])
        train_data = global_train_data[global_train_data["model_name"].isin(model_subset)].reset_index(drop=True)
        test_data = global_test_data[global_test_data["model_name"].isin(model_subset)].reset_index(drop=True)
        # print(train_data.head())
        # print(test_data.head())

    max_category = max(train_data["category_id"].max(), test_data["category_id"].max())
    min_category = min(train_data["category_id"].min(), test_data["category_id"].min())
    num_categories = max_category - min_category + 1

    class CustomDataset(Dataset):
        def __init__(self, data):
            # print(data["model_id"])
            model_ids = torch.tensor(data["model_id"], dtype=torch.int64)
            # Get unique model IDs and their corresponding new indices
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            # Map original IDs to their ranks
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids

            # print("Original IDs:", model_ids)
            # print("Ranked IDs:", ranked_model_ids)

            # print(self.models)
            self.prompts = torch.tensor(data["prompt_id"], dtype=torch.int64)
            self.labels = torch.tensor(data["label"], dtype=torch.int64)
            self.categories = torch.tensor(data["category_id"], dtype=torch.int64)
            self.num_models = len(data["model_id"].unique())
            self.num_prompts = len(data["prompt_id"].unique())
            self.num_classes = len(data["label"].unique())
            print(f"number of models: {self.num_models}, number of prompts: {self.num_prompts}")

        def get_num_models(self):
            return self.num_models

        def get_num_prompts(self):
            return self.num_prompts

        def get_num_classes(self):
            return self.num_classes

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return (
                self.models[index],
                self.prompts[index],
                self.labels[index],
                self.categories[index],
            )

        def get_dataloaders(self, batch_size):
            return DataLoader(self, batch_size, shuffle=False)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    num_models = train_dataset.get_num_models()
    num_prompts = train_dataset.get_num_prompts() + test_dataset.get_num_prompts()
    num_classes = train_dataset.get_num_classes()

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = test_dataset.get_dataloaders(batch_size)

    MODEL_NAMES = list(np.unique(list(train_data["model_name"])))
    return (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    )


class TextMF(torch.nn.Module):
    def __init__(self, embedding_path, dim, num_models, num_prompts, text_dim=768, num_classes=2, alpha=0.05):
        super().__init__()
        self._name = "TextMF"
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        # embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
        embeddings = torch.load(embedding_path)
        self.Q.weight.data.copy_(embeddings)
        self.text_proj = nn.Sequential(torch.nn.Linear(text_dim, dim))
        self.alpha = alpha

        # self.classifier = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, num_classes))
        self.classifier = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, model, prompt, category, test_mode=False):
        # print(model.shape)
        # print(self.P)
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)

        return self.classifier(p * q)

    def get_embedding(self):
        return (
            self.P.weight.detach().to("cpu").tolist(),
            self.Q.weight.detach().to("cpu").tolist(),
        )

    @torch.no_grad()
    def predict(self, model, prompt, category):
        logits = self.forward(model, prompt, category, test_mode=True)
        return torch.argmax(logits, dim=1)

def evaluator(net, test_iter, devices):
    net.eval()
    ls_fn = nn.CrossEntropyLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models, prompts, labels, categories in test_iter:
            # Assuming devices refer to potential GPU usage
            models = models.to(devices[0])  # Move data to the appropriate device
            prompts = prompts.to(devices[0])
            labels = labels.to(devices[0])
            categories = categories.to(devices[0])

            logits = net(models, prompts, categories)
            loss = ls_fn(logits, labels)  # Calculate the loss
            pred_labels = net.predict(models, prompts, categories)
            correct += (pred_labels == labels).sum().item()
            ls_list.append(loss.item())  # Store the sqrt of MSE (RMSE)
            num_samples += labels.shape[0]
    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples

def train_recsys_rating(
    net,
    train_iter,
    test_iter,
    num_models,
    num_prompts,
    batch_size,
    num_epochs,
    loss=nn.CrossEntropyLoss(reduction="mean"),
    devices=["cuda"],
    evaluator=evaluator,
    **kwargs,
):
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    def train_loop():  # Inner function for one epoch of training
        net.train()  # Set the model to training mode
        train_loss_sum, n = 0.0, 0
        start_time = time.time()
        for idx, (models, prompts, labels, categorys) in enumerate(train_iter):
            # Assuming devices refer to potential GPU usage
            # print(models)
            models = models.to(devices[0])
            prompts = prompts.to(devices[0])
            labels = labels.to(devices[0])
            categorys = categorys.to(devices[0])

            output = net(models, prompts, categorys)
            ls = loss(output, labels)

            optimizer.zero_grad()  # Clear the gradients
            ls.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            train_loss_sum += ls.item() * labels.shape[0]
            n += labels.shape[0]

        return train_loss_sum / n, time.time() - start_time

    train_losses = []
    test_losses = []
    test_acces = []
    embeddings = []
    progress_bar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        train_loss, train_time = train_loop()
        train_losses.append(train_loss)
        info = {"train_loss": train_loss, "epoch": epoch}

        if evaluator:
            test_ls, test_acc = evaluator(net, test_iter, devices)
            test_losses.append(test_ls)
            test_acces.append(test_acc)
            info.update({"test_loss": test_ls, "test_acc": test_acc, "epoch": epoch})
        else:
            test_ls = None  # No evaluation

        embeddings.append(net.get_embedding()[0])

        ray.train.report({"test_acc": test_acc}, checkpoint=None)
        # wandb.log(info)

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_ls, test_acc=test_acc)
        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_ls, test_acc=test_acc)
        progress_bar.update(1)

    progress_bar.close()
    return max(test_acces)

def run_mf(train_data_path, test_data_path, embedding_dim, alpha, embedding_path, saved_embedding_path):
    print("Start Loading Dataset")
    global_train_data = pd.read_csv(train_data_path)
    global_test_data = pd.read_csv(test_data_path)
    print("Finish Loading Dataset")

    embedding_dim = embedding_dim
    batch_size = 2048
    num_epochs = 50
    subset_size = None
    base_model_only = True
    alpha = alpha
    device = torch.device("cuda")

    (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    ) = split_and_load(global_train_data=global_train_data, global_test_data=global_test_data,
                       batch_size=batch_size, subset_size=subset_size, base_model_only=base_model_only,)

    mf = TextMF(
        embedding_path=embedding_path,
        dim=embedding_dim,
        num_models=num_models,
        num_prompts=35673, # TODO: fix this
        num_classes=num_classes,
        alpha=alpha,
    ).to(device)

    # num_epochs = 1
    max_test_acc = train_recsys_rating(
        mf,
        train_loader,
        test_loader,
        num_models,
        num_prompts,
        batch_size,
        num_epochs,
        devices=[device],
    )
    print(f"Embedding Dim: {embedding_dim}, Alpha: {alpha}")
    print(f"Max Test Accuracy: {max_test_acc}")

    torch.save(mf.P.weight, saved_embedding_path)

if __name__ == "__main__":
    pwd = os.getcwd()
    # TRAIN_DATA_PATH = f"{pwd}/data_new/mf_embedding_test/loo_truthfulqa_mathqa_train.csv"
    # TEST_DATA_PATH = f"{pwd}/data_new/mf_embedding_test/loo_truthfulqa_mathqa_test.csv"
    # SAVED_EMBEDDING_PATH = "data_new/mf_embedding_test/loo_truthfulqa_mathqa_embedding.pth"

    embeddings_folder_path = f"{pwd}/data_new/mf_embedding_test/for_paper/data"
    all_combo_ls = [item for item in os.listdir(embeddings_folder_path) if "train.csv" in item]
    # Index from 0 to 55
    start_index = 42
    end_index = 56

    embedding_dim = 232
    alpha = 0.001
    embedding_path = f"{pwd}/data_new/new_prompt_embeddings.pth"
    for i in range(start_index, end_index):
        train_data_path = f"{pwd}/data_new/mf_embedding_test/for_paper/data/{all_combo_ls[i]}"
        test_data_path = train_data_path.replace("train.csv", "test.csv")
        saved_embedding_path = f"{pwd}/data_new/mf_embedding_test/for_paper/embeddings/{all_combo_ls[i].replace('train.csv', 'embedding.pth')}"
        print(train_data_path)
        print(test_data_path)
        print(saved_embedding_path)
        run_mf(train_data_path, test_data_path, embedding_dim, alpha, embedding_path, saved_embedding_path)