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

MODEL_NAMES = [
    "vicuna_13b_v1.5",
    "vicuna_33b_v1.3",
    "tulu_30b",
    "SOLAR_10.7B_Instruct_v1.0",
    "Qwen_14B_Chat",
    "Llama_2_13b_chat_hf",
    "Mistral_7B_v0.1",
    "zephyr_7b_beta",
    "vicuna_7b_v1.5",
    "Llama_2_7b_chat_hf",
    "Starling_LM_7B_alpha",
    "baize_v2_13b",
    "Yi_34B_Chat",
    "koala_13B_HF",
    "mpt_7b_chat",
    "dolly_v2_12b",
    "stablelm_tuned_alpha_7b",
    "Orca_2_13b",
    "vicuna_7b_v1.5_16k",
    "openchat_3.5",
    "WizardLM_13B_V1.2",
    "openchat_3.5_0106",
    "Nous_Hermes_13b",
    "LlamaGuard_7b",
]


def read_data():
    data_dir = "/home/thw/llm2vec"
    # names = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv(os.path.join(data_dir, "triple_correctness.csv"), engine="python")
    num_models = data.model_id.unique().shape[0]
    num_prompts = data.prompt_id.unique().shape[0]
    return data, num_models, num_prompts


def split_and_load(test_ratio=0.9, batch_size=256, mode="random"):
    assert mode in ["random", "ood"]
    data, num_models, num_prompts = read_data()

    class TripleDataset(Dataset):
        def __init__(self, models, prompts, labels, num_models, num_prompts):
            self.models = torch.tensor(models, dtype=torch.int64)
            self.prompts = torch.tensor(prompts, dtype=torch.int64)
            self.labels = torch.tensor(labels, dtype=torch.int64)
            self.num_models = num_models
            self.num_prompts = num_prompts

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return self.models[index], self.prompts[index], self.labels[index]

        def get_dataloaders(self, batch_size, test_ratio=0.1, mode="random"):
            if mode == "random":
                num_test = int(len(self) * test_ratio)
                num_train = len(self) - num_test
                train_set, test_set = torch.utils.data.random_split(self, [num_train, num_test])
            else:
                num_test_prompts = int(self.num_prompts * test_ratio)
                num_train_prompts = self.num_prompts - num_test_prompts
                # split num_prompts into train and test
                train_prompts, test_prompts = torch.utils.data.random_split(torch.arange(self.num_prompts), [num_train_prompts, num_test_prompts])
                # use filtered prompts to split data
                train_indices = [i for i, prompt in enumerate(self.prompts) if prompt in train_prompts]
                train_set = torch.utils.data.Subset(self, train_indices)
                test_indices = [i for i, prompt in enumerate(self.prompts) if prompt not in train_prompts]
                test_set = torch.utils.data.Subset(self, test_indices)

                print(f"train set: {len(train_set)}, test set: {len(test_set)}")
            return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size, shuffle=False)

    data = TripleDataset(data["model_id"], data["prompt_id"], data["label"], num_models, num_prompts)
    # train test split
    train_iter, test_iter = data.get_dataloaders(batch_size, test_ratio, mode=mode)

    return num_models, num_prompts, train_iter, test_iter


# data, num_models, num_prompts = read_data()
# sparsity = 1 - len(data) / (num_models * num_prompts)
# print(f"number of models: {num_models}, number of prompts: {num_prompts}")
# print(f"matrix sparsity: {sparsity:f}")
# print(data.head(5))

# plot in plt
# plt.figure(figsize=(10, 5))
# plt.hist(data["rating"], bins=5, ec="black")
# plt.xlabel("Rating")
# plt.ylabel("Count")
# plt.title("Distribution of labels in MovieLens 100K")
# plt.show()


class MF(torch.nn.Module):
    def __init__(self, dim, num_models, num_prompts, num_classes=4):
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
        embeddings = json.load(open("/home/thw/llm2vec/embeddings.json"))
        self.Q.weight.data.copy_(torch.tensor(embeddings))
        self.text_proj = nn.Sequential(torch.nn.Linear(text_dim, 2 * text_dim), nn.ReLU(), torch.nn.Linear(2 * text_dim, dim))

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


# class AutoRec(nn.Module):
#     def __init__(self, num_models, num_prompts, dim=500, dropout=0.5):
#         super(AutoRec, self).__init__()
#         self.encoder = nn.Sequential(nn.Linear(num_prompts, dim), nn.Sigmoid())
#         self.decoder = nn.Sequential(nn.Linear(dim, num_prompts))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, user, item):
#         input = torch.zeros((len(user), num_prompts), dtype=torch.float32)
#         input[range(len(user)), item] = 1
#         return self.decoder(self.dropout(self.encoder(input)))


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


def train_recsys_rating(net, batch_size, num_epochs, loss=nn.CrossEntropyLoss(reduction="mean"), devices=["cuda"], evaluator=None, **kwargs):
    num_models, num_prompts, train_iter, test_iter = split_and_load(batch_size=batch_size, test_ratio=0.1, mode="ood")
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


_, num_models, num_prompts = read_data()
# _, _, train, test = split_and_load(mode="ood")

mf = TextMF(dim=10, num_models=num_models, num_prompts=num_prompts).to("cuda")
train_recsys_rating(mf, batch_size=512, num_epochs=1000, evaluator=evaluator)
# embeddings = np.load("embeddings.npy")
# plot_evolving_embeddings(embeddings, stride=20)
# for i in range(len(MODEL_NAMES)):
#     print(MODEL_NAMES[i], embeddings[-1][i])
