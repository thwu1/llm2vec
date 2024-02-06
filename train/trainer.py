from train.modeling import BertModelCustom, BertForContrastiveLearning
import torch
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


class ContrastiveTrainer:
    def __init__(self, model, dataloader, test_dataloader, optimizer_config, label_name):
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.label_name = label_name
        self.info_nce_temp = 1.0
        # self.criterion = torch.nn.CrossEntropyLoss()

        wandb.init(project="contrastive-learning")
        self.logger = wandb

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
        for epoch in range(100000):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                inputs = torch.cat([batch[0], batch[1]], dim=0).to(device)
                outputs = self.model(inputs)

                loss = self.info_nce_loss(outputs, self.info_nce_temp)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                step += 1
                self.logger.log({"loss": loss.item()})
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
        self.logger.log({"test_loss": test_loss.item()})
        print(f"Test loss: {test_loss.item()}")
        model_embeddings = self.model.encode(batch[0].to(self.model.get_device()), output_tensor=False)
        self.plot(model_embeddings, step)

    def save_model(self, step):
        self.model.save_pretrained(f"checkpoint_{step}")

    def load_model(self):
        pass

    def plot(self, model_embeddings, step):

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
        


        
