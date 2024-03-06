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

torch.manual_seed(42)
np.random.seed(42)

pwd = os.getcwd()

data = pd.read_csv("{pwd}/data/mmlu_correctness_1k.csv")
MODEL_NAMES = [key for key in data.columns]
print(MODEL_NAMES)


def split_and_load(batch_size=256):
    train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
    test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")

    class CustomDataset(Dataset):
        def __init__(self, data):

            self.models = torch.tensor(data["model_id"], dtype=torch.int64)
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
            return self.models[index], self.prompts[index], self.labels[index]

        def get_dataloaders(self, batch_size):
            return DataLoader(self, batch_size, shuffle=False)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    num_models = train_dataset.get_num_models()
    num_prompts = train_dataset.get_num_prompts() + test_dataset.get_num_prompts()
    num_classes = train_dataset.get_num_classes()

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = test_dataset.get_dataloaders(batch_size)

    return num_models, num_prompts, num_classes, train_loader, test_loader


class MF(torch.nn.Module):
    def __init__(self, dim, num_models, num_prompts, num_classes=2):
        super().__init__()
        self._name = "MF"
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, dim)
        # self.classifier = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, num_classes))
        self.classifier = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, model, prompt):
        p = self.P(model)
        q = self.Q(prompt)
        return self.classifier(p * q)

    def get_embedding(self):
        return self.P.weight.detach().to("cpu").tolist(), self.Q.weight.detach().to("cpu").tolist()

    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt)
        return torch.argmax(logits, dim=1)


class TextMF(torch.nn.Module):
    def __init__(self, dim, num_models, num_prompts, text_dim=768, num_classes=2):
        super().__init__()
        self._name = "TextMF"
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
        self.Q.weight.data.copy_(torch.tensor(embeddings))
        self.text_proj = nn.Sequential(torch.nn.Linear(text_dim, dim))

        # self.classifier = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, num_classes))
        self.classifier = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, model, prompt):
        p = self.P(model)
        q = self.text_proj(self.Q(prompt))
        return self.classifier(p * q)

    def get_embedding(self):
        return self.P.weight.detach().to("cpu").tolist(), self.Q.weight.detach().to("cpu").tolist()

    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt)
        return torch.argmax(logits, dim=1)


def evaluator(net, test_iter, devices):
    net.eval()
    ls_fn = nn.CrossEntropyLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models, prompts, labels in test_iter:
            # Assuming devices refer to potential GPU usage
            models = models.to(devices[0])  # Move data to the appropriate device
            prompts = prompts.to(devices[0])
            labels = labels.to(devices[0])

            logits = net(models, prompts)
            loss = ls_fn(logits, labels)  # Calculate the loss
            pred_labels = net.predict(models, prompts)
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
        for idx, (models, prompts, labels) in enumerate(train_iter):
            # Assuming devices refer to potential GPU usage
            models = models.to(devices[0])
            prompts = prompts.to(devices[0])
            labels = labels.to(devices[0])

            output = net(models, prompts)
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


batch_size = 512
num_epochs = 100
num_models, num_prompts, num_classes, train_loader, test_loader = split_and_load()
mf = TextMF(
    dim=64,
    num_models=num_models,
    num_prompts=num_prompts,
    num_classes=num_classes,
).to("cuda")
train_recsys_rating(mf, train_loader, test_loader, num_models, num_prompts, batch_size, num_epochs, devices=["cuda"])
