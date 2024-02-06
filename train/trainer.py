from train.modeling import BertModelCustom, BertForContrastiveLearning
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os


class ContrastiveTrainer:
    def __init__(self, model, dataloader, test_dataloader, config, label_name):
        self.model = model
        # embedding_params = [p for name, p in self.model.named_parameters() if "embedding" in name and p.requires_grad]  # Adjust based on your model structure
        # other_params = [p for name, p in self.model.named_parameters() if "embedding" not in name and p.requires_grad]
        # print(f"Embedding params: {embedding_params}")
        # param_groups = [
        #     {'params': embedding_params, 'lr': 1e-5},
        #     {'params': other_params, 'lr': 1e-5}
        # ]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **config["optimizer_config"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **config["scheduler_config"])
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.label_name = label_name
        self.config = config
        self.info_nce_temp = self.config["trainer_config"]["info_nce_temp"]
        self.num_epochs = self.config["trainer_config"]["num_epochs"]

        self.logger = wandb
        self.logger.init(project="contrastive-learning", config=config)

    def info_nce_loss(self, feats, temperature=1.0):

        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool).to(cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll

    def train(self):
        self.model.train()
        step = 0
        device = self.model.get_device()
        print(f"Device: {device}")
        for epoch in range(self.num_epochs):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                inputs = torch.cat([batch[0], batch[1]], dim=0).to(device)
                outputs = self.model(inputs)

                loss = self.info_nce_loss(outputs, self.info_nce_temp)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                step += 1
                self.logger.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

                if step % 100 == 0:
                    # self.save_model(step)
                    self.evaluate(step)

    def evaluate(self, step):
        batch = None
        for item in self.test_dataloader:
            batch = item
            break
        with torch.no_grad():
            inputs = torch.cat([batch[0], batch[1]], dim=0).to(self.model.get_device())
            outputs = self.model(inputs)
            test_loss = self.info_nce_loss(outputs, 1.0)
        self.logger.log({"test_loss": test_loss.item()}, commit=False)
        print(f"Test loss: {test_loss.item()}")
        model_embeddings = self.model.encode(batch[0].to(self.model.get_device()), output_tensor=False)
        self.plot(model_embeddings, step)

    def save_model(self, step):
        self.model.save_pretrained(f"checkpoint_{step}")

    def load_model(self):
        pass

    def plot(self, model_embeddings, step):
        if not os.path.exists("eval_figs"):
            os.makedirs("eval_figs")

        X = np.array(model_embeddings)
        reducer = PCA(n_components=2)
        reduced_X = reducer.fit_transform(X)
        name = [self.label_name[i] for i in range(len(self.label_name))]

        # Plot with seaborn
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=name, palette="viridis")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Models embedding visualized by PCA")
        plt.savefig(f"eval_figs/{step}.png")
        plt.close()
