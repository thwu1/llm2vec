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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

pwd = os.getcwd()

# data = pd.read_csv(f"{pwd}/data/mmlu_correctness_1k.csv")
# print(MODEL_NAMES[0:10])


def split_and_load(batch_size=64, subset_size=None, base_model_only=False):
    train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
    test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")
    # print(train_data.head())
    # print(test_data.head())

    if base_model_only:
        train_data = train_data[~train_data["model_name"].str.contains("vote|moe")].reset_index(drop=True)
        test_data = test_data[~test_data["model_name"].str.contains("vote|moe")].reset_index(drop=True)
        # print(sorted(list(train_data['model_id'].unique())))
        # print(sorted(list(test_data['model_id'].unique())))
    elif subset_size:
        model_subset = random.sample(list(test_data["model_name"]), subset_size)
        # print(model_subset[:10])
        train_data = train_data[train_data["model_name"].isin(model_subset)].reset_index(drop=True)
        test_data = test_data[test_data["model_name"].isin(model_subset)].reset_index(drop=True)
        # print(train_data.head())
        # print(test_data.head())

    max_category = max(train_data["category_id"].max(), test_data["category_id"].max())
    min_category = min(train_data["category_id"].min(), test_data["category_id"].min())
    num_categories = max_category - min_category + 1

    class CustomDataset(Dataset):
        def __init__(self, data, test=False):
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
            prompt_ids = torch.tensor(data["prompt_id"], dtype=torch.int64)
            unique_ids, inverse_indices = torch.unique(prompt_ids, sorted=True, return_inverse=True)
            # Map original IDs to their ranks
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_prompt_ids = torch.tensor([id_to_rank[id.item()] for id in prompt_ids])
            if test:
                self.prompts = ranked_prompt_ids + 900
            else:
                self.prompts = ranked_prompt_ids

            print("Original IDs:", prompt_ids)
            print("Ranked IDs:", ranked_prompt_ids)

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
    def __init__(self, dim, num_models, num_prompts, text_dim=768, num_classes=2, alpha=0.05):
        super().__init__()
        self._name = "TextMF"
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
        self.Q.weight.data.copy_(torch.tensor(embeddings))
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


class FM(torch.nn.Module):
    def __init__(self, dim, num_models, num_prompts, num_categories, text_dim=768):
        super().__init__()
        self._name = "FM"
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
        self.Q.weight.data.copy_(torch.tensor(embeddings))
        self.text_proj = nn.Sequential(torch.nn.Linear(text_dim, dim))
        self.category_embedding = torch.nn.Embedding(num_categories, dim)
        self.classifier = nn.Sequential(nn.Linear(dim, 2))

    def forward(self, model, prompt, category):
        p = self.P(model)
        q = self.text_proj(self.Q(prompt))
        v = self.category_embedding(category)
        return self.classifier(p * q + q * v + p * v)

    def get_embedding(self):
        return (
            self.P.weight.detach().to("cpu").tolist(),
            self.Q.weight.detach().to("cpu").tolist(),
        )

    @torch.no_grad()
    def predict(self, model, prompt, category):
        logits = self.forward(model, prompt, category)
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


def plot_evolving_embeddings(embeddings, stride=10):
    markers = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "h"]
    embeddings = np.array(embeddings)[0::stride]
    num_points, num_models, dim = embeddings.shape
    if dim > 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings.reshape(-1, dim)).reshape(num_points, num_models, 2)

    plt.figure()

    for i in range(num_models):
        # Define a single color for each model
        color = plt.cm.rainbow(i / (num_models - 1))
        marker = markers[i % len(markers)]

        # Define brightness for each point within the model
        brightness = np.linspace(0.0, 0.25, num_points)  # Adjust brightness from 0.5 to 1

        for j in range(num_points - 1):
            line_color = color[:3] + (brightness[j],)  # Set brightness component of the color
            plt.plot(
                embeddings[j : j + 2, i, 0],
                embeddings[j : j + 2, i, 1],
                marker=marker,
                linestyle="-",
                color=line_color,
                markersize=5,
            )

        plt.plot([], [], label=MODEL_NAMES[i], color=color, marker=marker)

    plt.legend()
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.show()


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
    lr = 3e-4
    weight_decay = 1e-5
    wandb.init(project="llm2vec")
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    wandb.config.update(
        {
            "model": net.eval(),
            "num_models": num_models,
            "num_prompts": num_prompts,
            "loss": loss,
            "num_epochs": num_epochs,
            "kwargs": kwargs,
            "optimizer": "adam",
            "lr": lr,
            "weight_decay": weight_decay,
            "test_ratio": 0.1,
            "batch_size": batch_size,
        }
    )

    def train_loop():  # Inner function for one epoch of training
        net.train()  # Set the model to training mode
        train_loss_sum, n = 0.0, 0
        start_time = time.time()
        for idx, (models, prompts, labels, categorys) in enumerate(train_iter):
            # Assuming devices refer to potential GPU usage
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

        wandb.log(info)

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_ls, test_acc=test_acc)
        progress_bar.update(1)

    progress_bar.close()
    wandb.finish()

    # plot evolution of embeddings
    np.save("embeddings.npy", embeddings)
    plot_evolving_embeddings(embeddings, stride=max(num_epochs // 50, 1))
    return max(test_acces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--base_model_only", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.05, help="noise level")
    args = parser.parse_args()

    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    subset_size = args.subset_size
    base_model_only = args.base_model_only
    alpha = args.alpha
    device = torch.device("cpu")

    (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    ) = split_and_load(batch_size=batch_size, subset_size=subset_size, base_model_only=base_model_only)

    mf = TextMF(
        dim=embedding_dim,
        num_models=num_models,
        num_prompts=num_prompts, # TODO: Need to change back
        num_classes=num_classes,
        alpha=alpha,
    ).to(device)

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
    # fm = FM(
    #     dim=32,
    #     num_models=num_models,
    #     num_prompts=num_prompts,
    #     num_categories=num_categories,
    # ).to("cuda")
    # train_recsys_rating(
    #     fm,
    #     train_loader,
    #     test_loader,
    #     num_models,
    #     num_prompts,
    #     BATCH_SIZE,
    #     NUM_EPOCHS,
    #     devices=["cuda"],
    # )