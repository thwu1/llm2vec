import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import random
import os

def load_data(base_model_only):
    pwd = os.getcwd()

    if base_model_only:
        with open(f'{pwd}/data/model_order_base_only.pkl', 'rb') as file:
            model_order = pickle.load(file)
        with open(f'{pwd}/data/train_prompt_base_only.pkl', 'rb') as file:
            train_prompt_order = pickle.load(file)
        with open(f'{pwd}/data/val_prompt_base_only.pkl', 'rb') as file:
            val_prompt_order = pickle.load(file)
        with open(f'{pwd}/data/test_prompt_base_only.pkl', 'rb') as file:
            test_prompt_order = pickle.load(file)
        train_x = torch.load(f'{pwd}/data/train_x_base_only.pth')
        train_y = torch.load(f'{pwd}/data/train_y_base_only.pth')
        val_x = torch.load(f'{pwd}/data/val_x_base_only.pth')
        val_y = torch.load(f'{pwd}/data/val_y_base_only.pth')
        test_x = torch.load(f'{pwd}/data/test_x_base_only.pth')
        test_y = torch.load(f'{pwd}/data/test_y_base_only.pth')
    else:
        with open(f'{pwd}/data/model_order_full.pkl', 'rb') as file:
            model_order = pickle.load(file)
        with open(f'{pwd}/data/train_prompt_full.pkl', 'rb') as file:
            train_prompt_order = pickle.load(file)
        with open(f'{pwd}/data/val_prompt_full.pkl', 'rb') as file:
            val_prompt_order = pickle.load(file)
        with open(f'{pwd}/data/test_prompt_full.pkl', 'rb') as file:
            test_prompt_order = pickle.load(file)
        train_x = torch.load(f'{pwd}/data/train_x_full.pth')
        train_y = torch.load(f'{pwd}/data/train_y_full.pth')
        val_x = torch.load(f'{pwd}/data/val_x_full.pth')
        val_y = torch.load(f'{pwd}/data/val_y_full.pth')
        test_x = torch.load(f'{pwd}/data/test_x_full.pth')
        test_y = torch.load(f'{pwd}/data/test_y_full.pth')

    # print(model_order, train_prompt_order, val_prompt_order, test_prompt_order)
    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
    return model_order, train_prompt_order, val_prompt_order, test_prompt_order, train_x, train_y, val_x, val_y, test_x, test_y


class CustomDataset(Dataset):
    # x: (batch_size, num_total_questions, question_embedding_dim)
    # y: (batch_size, num_total_questions)
    def __init__(self,x,y):
        self.data = x
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # Shape (num_models, question_embedding_dim)
        y = self.labels[idx]  # Shape (num_models)
        
        return x, y

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, linear=False, layernorm=False):  # hidden_sizes is a list
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(input_size, hidden_sizes[0]))
        if not linear:
            self.mlp.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            self.mlp.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if not linear:
                self.mlp.append(nn.ReLU())
            if layernorm:
                self.mlp.append(nn.LayerNorm(hidden_sizes[i]))
        self.mlp.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):  # Input a sequence of questions and output z
    # TODO: Try LayerNorm
    def __init__(self, c_dim, z_dim, linear=False, layernorm=False):
        super().__init__()
        self.mean_network = MLP(input_size=c_dim, hidden_sizes=[512, 256], output_size=z_dim, 
                                linear=linear, layernorm=layernorm)
        self.variance_network = MLP(input_size=c_dim, hidden_sizes=[512, 256], output_size=z_dim, 
                                    linear=linear, layernorm=layernorm)

    def forward(self, cs):  # Input multiple instances of context and output a latent representation of the task
        # Input: cs(context) of shape (batch_size, len, c_dim), batch_size = num_models, len = sample_subset_size
        mu = self.mean_network(cs)  # Shape: [batch_size(num_models), len(num_sample), z_dim]
        sigma_squared = F.softplus(self.variance_network(cs))  # Shape: [batch_size(num_models), len(num_sample), z_dim]
        bs, length, z_dim = list(mu.shape)

        z_params = [
            self._product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
        ]  # Shape: tuple of 2 [batch_size(num_models), len(num_sample), z_dim]
        zs = []
        logprobs = []
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params])
        for z_mean, z_var in zip(z_means, z_vars):
            tmp_z, tmp_prob = self.sample_z(z_mean, z_var)
            zs.append(tmp_z)
            logprobs.append(tmp_prob)
        zs = torch.stack(zs)
        logprobs = torch.stack(logprobs)
        return zs, logprobs, z_means, z_vars

    def sample_z(self, z_means, z_vars):  # Sample from a multivariate gaussian
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        z = [d.rsample() for d in posteriors]
        prob = [d.log_prob(z_) for d, z_ in zip(posteriors, z)]
        return torch.stack(z), torch.stack(prob)

    def _product_of_gaussians(self, mus, sigmas_squared):
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        prod_sigma_squared_inv = torch.cumsum(torch.reciprocal(sigmas_squared), dim=0)
        prod_sigma_squared = 1.0 / prod_sigma_squared_inv
        mu = prod_sigma_squared * torch.cumsum(mus / sigmas_squared, dim=0)
        return mu, prod_sigma_squared


class Decoder(nn.Module):  # Get input from encoder which is z and a new question, concat or project to same dim and dot product
    def __init__(self, q_dim, z_dim, linear=False, layernorm=False):
        super().__init__()
        self.network = MLP(
            input_size=z_dim + q_dim, hidden_sizes=[128, 16], output_size=1,
            linear=linear, layernorm=layernorm
        )

    def forward(self, zs, qs):
        # zs: [batch_size, len, z_dim]
        # qs: [batch_size, len, q_dim]
        # return [batch_size, len]
        x = torch.cat(
            (zs, qs), dim=-1
        )
        return self.network(x)
class Trainer:  # batch-wise autoregressively input k question and get (k+1)_th questions' answer
    def __init__(self, encoder, decoder, sample_length, train_dataloader, val_dataloader=None, test_dataloader=None, 
                 use_kl=True, kl_weight=1, device='cpu', train_on_subset=False):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.sample_length = sample_length
        self.use_kl = use_kl
        self.kl_weight = kl_weight
        self.device = device
        self.train_on_subset = train_on_subset

    def train(self, epochs=10):
        test_accuracies = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}: ")
            self.train_epoch(self.train_dataloader)

            # if self.test_dataloader and self.val_dataloader:
            avg_loss, avg_accuracy = self.evaluate(self.val_dataloader, self.test_dataloader)
            test_accuracies.append(avg_accuracy)
        
        print(f"Max Test Accuracy: {max(test_accuracies)}")

    def kl_loss(self, z_means, z_vars):
        z_means = z_means.to(self.device)
        z_vars = z_vars.to(self.device)
        batch_size, subset_length, z_dim = z_means.shape

        prior = torch.distributions.Normal(torch.zeros(z_dim, device=self.device), torch.ones(z_dim, device=self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
            torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.mean(torch.stack(kl_divs))

        return kl_div_sum * self.kl_weight

    def train_epoch(self, train_dataloader):
        total_loss = 0
        gen_indices = True
        for batch in tqdm(train_dataloader):
            # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
            p_embeds, labels = batch
            p_embeds = p_embeds.to(self.device)
            labels = labels.to(self.device)
            batch_size, num_total_question, q_dim = p_embeds.shape
            if gen_indices:
                gen_indices = False
                indices = torch.randperm(num_total_question)
            p_embeds = p_embeds[:, indices, :]
            labels = labels[:, indices]

            for i in range(0, num_total_question, self.sample_length):
                p_embeds_sample = p_embeds[:, i : i + self.sample_length, :]  # Shape: [batch_size, sample_size, prompt_embed_dim]
                labels_sample = labels[:, i : i + self.sample_length]  # Shape: [batch_size, sample_size]
                # print(p_embeds_sample.shape, labels_sample.shape)
                # TODO: Add an option of random sampling of subset size (Exponential Distribution, clamp at 30-900)

                cs = torch.cat((p_embeds_sample, labels_sample.unsqueeze(-1)), dim=-1).to(self.device)

                zs, logprobs, z_means, z_vars = self.encoder(cs)
                posteriors = torch.exp(logprobs)
                preds = self.decoder(zs[:, :-1], p_embeds_sample[:, 1:]).squeeze(-1)
                loss = self.loss_fn(preds, labels_sample[:, 1:].float())
                loss = (loss * posteriors[:, :-1]).mean()
                if self.use_kl:
                    loss += self.kl_loss(z_means, z_vars)
                total_loss += loss.item()

                # Calculate accuracy
                probabilities = torch.sigmoid(preds)
                predicted_labels = (probabilities > 0.5).float()
                correct_predictions = (predicted_labels == labels_sample[:, 1:].float()).float()
                accuracy = correct_predictions.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f'Loss: {loss}')
                if self.train_on_subset:
                    break

        # print(f'Training Loss: {loss}')
        print(f'Training Accuracy (Autoregressive): {accuracy}')

    def evaluate(self, val_dataloader, test_dataloader):
        # TODO: Add one more accuracy evaluation method (max sample and evaluate all questions left)
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        
        # Generate Z on Validation Set
        p_embeds_val, labels_val = next(iter(val_dataloader))
        p_embeds_val = p_embeds_val.to(self.device)
        labels_val = labels_val.to(self.device)
        cs = torch.cat((p_embeds_val, labels_val.unsqueeze(-1)), dim=-1).to(self.device)
        zs, logprobs, z_means, z_vars = self.encoder(cs)

        # TODO: Evaluate on Whole test dataset
        p_embeds_test, labels_test = next(iter(test_dataloader))
        p_embeds_test = p_embeds_test.to(self.device)
        labels_test = labels_test.to(self.device)

        # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
        batch_size, num_test_question, q_dim = p_embeds_test.shape
        
        posteriors = torch.exp(logprobs)
        # TODO: Check if this implementation is correct
        preds = self.decoder(zs[:, -1, :].unsqueeze(1).repeat(1, num_test_question, 1), p_embeds_test).squeeze(-1)
        loss = self.loss_fn(preds, labels_test.float())
        loss = (loss * posteriors[:, :-1]).mean()
        if self.use_kl:
            loss += self.kl_loss(z_means, z_vars)
        total_loss += loss.item()
        
        # Calculate accuracy
        probabilities = torch.sigmoid(preds)
        predicted_labels = (probabilities > 0.5).float()
        correct_predictions = (predicted_labels == labels_test.float()).float()
        test_accuracy = correct_predictions.mean()

        # avg_loss = total_loss
        # avg_accuracy = accuracy
        # print(f'Test Loss: {avg_loss}')
        print(f'Test Accuracy (Max Context Length): {test_accuracy}')

        self.encoder.train()
        self.decoder.train()
            
        return total_loss, test_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 512
# TODO: Try OpenAI's Ada Embedding (remember to cache)
SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
SAMPLE_LENGTH = 50 # Sample length choices: 50 or 100
USE_KL = True

# TODO: Hyperparameter Search
Z_DIM = 32 # Z_DIM choices: 32,64,96,128
NUM_EPOCHS = 10
KL_WEIGHT = 5 # Weight choices: 1,3,5,10
BASE_MODEL_ONLY = True
TRAIN_ON_SUBSET = False

print("Start Initializing Dataset...")
model_order, train_prompt_order, val_prompt_order, test_prompt_order, train_x, train_y, val_x, val_y, test_x, test_y = load_data(
        base_model_only=BASE_MODEL_ONLY)
train_dataset = CustomDataset(train_x, train_y)
val_dataset = CustomDataset(val_x, val_y)
test_dataset = CustomDataset(test_x, test_y)
print("Finish Initializing Dataset")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
encoder = Encoder(c_dim=EMBEDDING_DIM+1, z_dim=Z_DIM, linear=True)
decoder = Decoder(q_dim=EMBEDDING_DIM, z_dim=Z_DIM)
trainer = Trainer(encoder, decoder, SAMPLE_LENGTH, train_dataloader, val_dataloader, test_dataloader, 
                  use_kl=USE_KL, kl_weight = KL_WEIGHT, device=device, train_on_subset=TRAIN_ON_SUBSET)

trainer.train(epochs=NUM_EPOCHS)
