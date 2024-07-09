import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import random,os,ray,time,pickle
from ray.train import Checkpoint
from sklearn.cluster import KMeans

def load_data(pwd):
    train_x = torch.tensor(torch.load(f'{pwd}/data_new/new_train_x.pth')).float()
    train_y = torch.tensor(torch.load(f'{pwd}/data_new/new_train_y.pth')).float()
    train_val_x = torch.tensor(torch.load(f'{pwd}/data_new/evenly_by_benchmark/reference_set_evenly_from_benchmark_x.pth')).float()
    train_val_y = torch.tensor(torch.load(f'{pwd}/data_new/evenly_by_benchmark/reference_set_evenly_from_benchmark_y.pth')).float()
    val_x = torch.tensor(torch.load(f'{pwd}/data_new/new_val_x.pth')).float()
    val_y = torch.tensor(torch.load(f'{pwd}/data_new/new_val_y.pth')).float()
    test_x = torch.tensor(torch.load(f'{pwd}/data_new/new_test_x.pth')).float()
    test_y = torch.tensor(torch.load(f'{pwd}/data_new/new_test_y.pth')).float()
    
    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, train_val_x, train_val_y, val_x, val_y, test_x, test_y

def create_clusters(x, K):
    # Perform k-means clustering on the question embeddings (29673 x 768)
    question_embeddings = x[0]  # shape: (29673, 768)
    kmeans_model = KMeans(n_clusters=K, random_state=42)
    clusters = kmeans_model.fit_predict(question_embeddings)

    # Create a mapping from cluster indices to question indices
    cluster_to_indices = {i: [] for i in range(K)}
    for idx, cluster in enumerate(clusters):
        cluster_to_indices[cluster].append(idx)

    return cluster_to_indices, kmeans_model

def create_cluster_batches(x, y, clusters):
    cluster_batches = []
    for cluster, indices in clusters.items():
        cluster_x = x[:, indices, :]
        cluster_y = y[:, indices]
        cluster_batches.append((cluster_x, cluster_y))
    return cluster_batches

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
        self.during_eval = False

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
            # start_time = time.time()
            tmp_z, tmp_prob = self.sample_z(z_mean, z_var)
            zs.append(tmp_z)
            logprobs.append(tmp_prob)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"Time taken to sample Z: {elapsed_time:.2f} seconds")
        zs = torch.stack(zs)
        logprobs = torch.stack(logprobs)
        return zs, logprobs, z_means, z_vars

    def sample_z(self, z_means, z_vars):  # Sample from a multivariate gaussian
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        if self.during_eval:
            # print("Evaluating, Output mean")
            z = [d.loc for d in posteriors]
        else:
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
    def __init__(self, q_dim, z_dim, use_concat=False, linear=False, layernorm=False, normalize=False):
        super().__init__()
        self.q_proj = nn.Sequential(torch.nn.Linear(q_dim, z_dim))
        self.use_concat = use_concat
        self.linear = linear
        self.layernorm = layernorm
        self.normalize = normalize
        if self.use_concat:
            input_dim = z_dim + q_dim
        else:
            input_dim = z_dim
        self.classifier = MLP(
            input_size=input_dim, hidden_sizes=[128, 16], output_size=1, linear=linear, layernorm=layernorm)

    def forward(self, zs, qs):
        # zs: [batch_size, len, z_dim]
        # qs: [batch_size, len, q_dim]
        # return [batch_size, len]

        # Normalize zs
        if self.normalize:
            zs = F.normalize(zs, p=2, dim=2)

        if self.use_concat:
            x = torch.cat((zs, qs), dim=-1)
        else:
            qs = self.q_proj(qs)
            x = zs * qs

        y = self.classifier(x)
        return y
    
class Trainer:  # batch-wise autoregressively input k question and get (k+1)_th questions' answer
    def __init__(self, encoder, decoder, sample_length, train_dataloader, test_dataloader=None, ref_dataloader=None,
                 lr = 1e-3, use_kl=True, kl_weight=1, device='cpu', train_on_subset=False, train_break_threshold=100,
                 test_seeds=None, use_clustering=False, train_clusters=None, kmeans_model=None, test_clusters=None):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.ref_dataloader = ref_dataloader
        self.optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.sample_length = sample_length
        self.use_kl = use_kl
        self.kl_weight = kl_weight
        self.device = device
        self.train_on_subset = train_on_subset
        self.train_break_threshold = train_break_threshold
        self.use_clustering = use_clustering
        if self.use_clustering:
            self.train_clusters = train_clusters
            self.test_clusters = test_clusters
            self.kmeans_model = kmeans_model
        if self.train_on_subset:
            print("TRAINING ON SUBSET!")

        NUM_TESTS = 5
        self.test_num = NUM_TESTS
        if test_seeds:
            self.test_seeds = test_seeds
        else:
            TEST_SEEDS = [random.randint(1, 1000) for i in range(NUM_TESTS)]
            self.test_seeds = TEST_SEEDS
        print(f"Random Seeds: {self.test_seeds}")


    def train(self, epochs=10, ar_train=True, ar_eval=False):
        test_accuracies = []
        # Prepare the "Reference Set"
        if self.use_clustering:
            self.prepare_clustered_reference_set()
        else:
            self.prepare_reference_set()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}: ")
            
            if self.use_clustering:
                self.train_epoch_clustered(self.train_clusters)
            elif ar_train:
                self.train_epoch_autoregressive(self.train_dataloader)
            else:
                self.train_epoch_max_context(self.train_dataloader)

            if self.use_clustering:
                test_accuracy = self.evaluate_clustered()
            elif ar_eval:
                test_accuracy = self.evaluate_autoregressive(self.val_dataloader, self.test_dataloader)
            else:
                test_accuracy = self.evaluate_max_context()
            
            # ray.train.report({"test_accuracy": test_accuracy}, checkpoint=None)
            test_accuracies.append(test_accuracy)
        
        print(f"Max Test Accuracy: {max(test_accuracies)}")
        ray.train.report({"test_acc": max(test_accuracies).float().item()}, checkpoint=None)
        return max(test_accuracies).float().item()

    def kl_loss(self, z_means, z_vars):
        z_means = z_means.to(self.device)
        z_vars = z_vars.to(self.device)
        batch_size, subset_length, z_dim = z_means.shape

        prior = torch.distributions.Normal(torch.zeros(z_dim, device=self.device), torch.ones(z_dim, device=self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
            torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.mean(torch.stack(kl_divs))

        return kl_div_sum

    def train_epoch_autoregressive(self, train_dataloader):
        total_loss = 0
        gen_indices = True
        train_accuracies = []
        PARALLEL = False
        RANDOMIZE = True
        for batch in tqdm(train_dataloader):
            # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
            p_embeds, labels = batch
            p_embeds = p_embeds.to(self.device)
            labels = labels.to(self.device)
            batch_size, num_total_question, q_dim = p_embeds.shape
            if RANDOMIZE:
                if gen_indices:
                    gen_indices = False
                    indices = torch.randperm(num_total_question)
                p_embeds = p_embeds[:, indices, :]
                labels = labels[:, indices]
            if PARALLEL:
                # start_time = time.time()
                if self.train_on_subset:
                    assert num_total_question >= self.sample_length * self.train_break_threshold, "Threshold exceeds num_total_question"
                    p_embeds_sample = p_embeds[:, 0:self.sample_length*self.train_break_threshold, :]
                    labels_sample = labels[:, 0:self.sample_length*self.train_break_threshold]
                    # print(p_embeds_sample.shape, labels_sample.shape)
                    p_embeds_sample = p_embeds_sample.reshape(batch_size * self.train_break_threshold, self.sample_length, q_dim)
                    labels_sample = labels_sample.reshape(batch_size * self.train_break_threshold, self.sample_length)

                    cs = torch.cat((p_embeds_sample, labels_sample.unsqueeze(-1)), dim=-1).to(self.device)

                    zs, logprobs, z_means, z_vars = self.encoder(cs)
                    posteriors = torch.exp(logprobs)
                    preds = self.decoder(zs[:, :-1], p_embeds_sample[:, 1:]).squeeze(-1)
                    # print(preds.shape, labels_sample[:, 1:].shape)
                    loss = self.loss_fn(preds, labels_sample[:, 1:].float())
                    # print(posteriors[:, :-1])
                    # print(loss.shape)
                    # print(posteriors[:, :-1].shape)
                    loss = (loss * posteriors[:, :-1].sum(dim=-1)).mean()
                    if self.use_kl:
                        loss += self.kl_loss(z_means, z_vars) * self.kl_weight
                    total_loss += loss.item()

                    # Calculate accuracy
                    probabilities = torch.sigmoid(preds)
                    predicted_labels = (probabilities > 0.5).float()
                    correct_predictions = (predicted_labels == labels_sample[:, 1:].float()).float()
                    # print(correct_predictions.shape)
                    # print(torch.mean(correct_predictions, dim=0))
                    accuracy = correct_predictions.mean()
                    train_accuracies.append(accuracy)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    num_full_samples = num_total_question // self.sample_length
                    remainder = num_total_question % self.sample_length
                    p_embeds_full = p_embeds[:, :num_full_samples * self.sample_length, :].reshape(
                        batch_size * num_full_samples, self.sample_length, q_dim)
                    labels_full = labels[:, :num_full_samples * self.sample_length].reshape(
                        batch_size * num_full_samples, self.sample_length)

                    cs_full = torch.cat((p_embeds_full, labels_full.unsqueeze(-1)), dim=-1).to(self.device)

                    zs_full, logprobs_full, z_means_full, z_vars_full = self.encoder(cs_full)
                    posteriors_full = torch.exp(logprobs_full)
                    preds_full = self.decoder(zs_full[:, :-1], p_embeds_full[:, 1:]).squeeze(-1)
                    loss_full = self.loss_fn(preds_full, labels_full[:, 1:].float())
                    loss_full = (loss_full * posteriors_full[:, :-1].sum(dim=-1)).mean()
                    if self.use_kl:
                        loss_full += self.kl_loss(z_means_full, z_vars_full) * self.kl_weight
                    total_loss += loss_full.item()

                    # Calculate accuracy
                    probabilities_full = torch.sigmoid(preds_full)
                    predicted_labels_full = (probabilities_full > 0.5).float()
                    correct_predictions_full = (predicted_labels_full == labels_full[:, 1:].float()).float()
                    # print(correct_predictions.shape)
                    # print(torch.mean(correct_predictions, dim=0))
                    accuracy_full = correct_predictions_full.mean()

                    self.optimizer.zero_grad()
                    loss_full.backward()
                    self.optimizer.step()

                    p_embeds_remainder = p_embeds[:, num_full_samples * self.sample_length:, :]
                    labels_remainder = labels[:, num_full_samples * self.sample_length:]

                    cs_remainder = torch.cat((p_embeds_remainder, labels_remainder.unsqueeze(-1)), dim=-1).to(self.device)

                    zs_remainder, logprobs_remainder, z_means_remainder, z_vars_remainder = self.encoder(cs_remainder)
                    posteriors_remainder = torch.exp(logprobs_remainder)
                    preds_remainder = self.decoder(zs_remainder[:, :-1], p_embeds_remainder[:, 1:]).squeeze(-1)
                    loss_remainder = self.loss_fn(preds_remainder, labels_remainder[:, 1:].float())
                    loss_remainder = (loss_remainder * posteriors_remainder[:, :-1].sum(dim=-1)).mean()
                    if self.use_kl:
                        loss_remainder += self.kl_loss(z_means_remainder, z_vars_remainder) * self.kl_weight
                    total_loss += loss_remainder.item()

                    # Calculate accuracy
                    probabilities_remainder = torch.sigmoid(preds_remainder)
                    predicted_labels_remainder = (probabilities_remainder > 0.5).float()
                    correct_predictions_remainder = (predicted_labels_remainder == labels_remainder[:, 1:].float()).float()
                    accuracy_remainder = correct_predictions_remainder.mean()

                    train_accuracies.append((accuracy_full * num_full_samples * self.sample_length + 
                                            accuracy_remainder * remainder)/num_total_question)

                    self.optimizer.zero_grad()
                    loss_remainder.backward()
                    self.optimizer.step()
                    # print(f'Loss: {loss}')
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Time taken to run parallel training: {elapsed_time:.2f} seconds")
            else:
                # start_time = time.time()
                for i in range(0, num_total_question, self.sample_length):
                    p_embeds_sample = p_embeds[:, i : i + self.sample_length, :]  # Shape: [batch_size, sample_size, prompt_embed_dim]
                    labels_sample = labels[:, i : i + self.sample_length]  # Shape: [batch_size, sample_size]
                    # print(p_embeds_sample.shape, labels_sample.shape)
                    # TODO: Add an option of random sampling of subset size (Exponential Distribution, clamp at 30-900)

                    cs = torch.cat((p_embeds_sample, labels_sample.unsqueeze(-1)), dim=-1).to(self.device)

                    zs, logprobs, z_means, z_vars = self.encoder(cs)
                    posteriors = torch.exp(logprobs)
                    preds = self.decoder(zs[:, :-1], p_embeds_sample[:, 1:]).squeeze(-1)
                    # print(preds.shape, labels_sample[:, 1:].shape)
                    loss = self.loss_fn(preds, labels_sample[:, 1:].float())
                    # print(posteriors[:, :-1])
                    # print(loss.shape)
                    # print(posteriors[:, :-1].shape)
                    loss = (loss * posteriors[:, :-1].sum(dim=-1)).mean()
                    if self.use_kl:
                        loss += self.kl_loss(z_means, z_vars) * self.kl_weight
                    total_loss += loss.item()

                    # Calculate accuracy
                    probabilities = torch.sigmoid(preds)
                    predicted_labels = (probabilities > 0.5).float()
                    correct_predictions = (predicted_labels == labels_sample[:, 1:].float()).float()
                    # print(correct_predictions.shape)
                    # print(torch.mean(correct_predictions, dim=0))
                    accuracy = correct_predictions.mean()
                    train_accuracies.append(accuracy)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # print(f'Loss: {loss}')
                    if self.train_on_subset:
                        if (i // self.sample_length) >= self.train_break_threshold:
                            break
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Time taken to run for loop training: {elapsed_time:.2f} seconds")

        # print(f'Training Loss: {loss}')
        print(f'Training Accuracy (Autoregressive): {sum(train_accuracies)/len(train_accuracies)}')

    def train_epoch_clustered(self, cluster):
        for cluster_x, cluster_y in cluster:
            cluster_dataloader = DataLoader(dataset=CustomDataset(cluster_x, cluster_y), batch_size=BATCH_SIZE, shuffle=True)
            self.train_epoch_autoregressive(cluster_dataloader)

    # def train_epoch_max_context(self, train_dataloader):
    #     torch.cuda.manual_seed(seed)
    #     total_loss = 0
    #     gen_indices = True
    #     train_accuracies = []
    #     for batch in tqdm(train_dataloader):
    #         # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
    #         p_embeds, labels = batch
    #         p_embeds = p_embeds.to(self.device)
    #         labels = labels.to(self.device)
    #         batch_size, num_total_question, q_dim = p_embeds.shape
    #         if gen_indices:
    #             gen_indices = False
    #             indices = torch.randperm(num_total_question)
    #         p_embeds = p_embeds[:, indices, :]
    #         labels = labels[:, indices]

    #         # if self.train_on_subset:
    #         #     start_time = time.time()
    #         #     p_embeds_sample = p_embeds[:, 0:self.sample_length*self.train_break_threshold, :]
    #         #     labels_sample = labels[:, 0:self.sample_length*self.train_break_threshold]
    #         #     print(p_embeds_sample.shape, labels_sample.shape)
    #         #     p_embeds_sample = p_embeds_sample.reshape(batch_size * self.train_break_threshold, self.sample_length, q_dim)
    #         #     labels_sample = labels_sample.reshape(batch_size * self.train_break_threshold, self.sample_length)

    #         #     cs = torch.cat((p_embeds_sample, labels_sample.unsqueeze(-1)), dim=-1).to(self.device)

    #         #     zs, logprobs, z_means, z_vars = self.encoder(cs)
    #         #     posteriors = torch.exp(logprobs)
    #         #     preds = self.decoder(zs[:, :-1], p_embeds_sample[:, 1:]).squeeze(-1)
    #         #     # print(f"posteriors: {posteriors.shape}")
    #         #     # print(f"preds: {preds.shape}, labels: {labels_sample.shape}")
    #         #     loss = self.loss_fn(preds[:, -1], labels_sample[:, -1].float())
    #         #     # print(loss, posteriors.shape)
    #         #     loss = (loss * posteriors[:, -1, :].sum(dim=-1)).mean()
    #         #     # print(loss)
    #         #     if self.use_kl:
    #         #         loss += self.kl_loss(z_means, z_vars) * self.kl_weight
    #         #     total_loss += loss.item()

    #         #     # Calculate accuracy
    #         #     probabilities = torch.sigmoid(preds)
    #         #     predicted_labels = (probabilities > 0.5).float()
    #         #     correct_predictions = (predicted_labels[:, -1] == labels_sample[:, -1].float()).float()
    #         #     accuracy = correct_predictions.mean()
    #         #     train_accuracies.append(accuracy)

    #         #     self.optimizer.zero_grad()
    #         #     loss.backward()
    #         #     self.optimizer.step()
    #         #     # print(f'Loss: {loss}')
    #         #     end_time = time.time()
    #         #     elapsed_time = end_time - start_time
    #         #     print(f"Time taken to run parallel training: {elapsed_time:.2f} seconds")

    #         for i in range(0, num_total_question, self.sample_length):
    #             start_time = time.time()
    #             p_embeds_sample = p_embeds[:, i : i + self.sample_length, :]  # Shape: [batch_size, sample_size, prompt_embed_dim]
    #             labels_sample = labels[:, i : i + self.sample_length]  # Shape: [batch_size, sample_size]
    #             # print(p_embeds_sample.shape, labels_sample.shape)
    #             # TODO: Add an option of random sampling of subset size (Exponential Distribution, clamp at 30-900)

    #             cs = torch.cat((p_embeds_sample, labels_sample.unsqueeze(-1)), dim=-1).to(self.device)

    #             zs, logprobs, z_means, z_vars = self.encoder(cs)
    #             posteriors = torch.exp(logprobs)
    #             preds = self.decoder(zs[:, :-1], p_embeds_sample[:, 1:]).squeeze(-1)
    #             # print(f"posteriors: {posteriors.shape}")
    #             # print(f"preds: {preds.shape}, labels: {labels_sample.shape}")
    #             loss = self.loss_fn(preds[:, -1], labels_sample[:, -1].float())
    #             # print(loss, posteriors.shape)
    #             loss = (loss * posteriors[:, -1, :].sum(dim=-1)).mean()
    #             # print(loss)
    #             if self.use_kl:
    #                 loss += self.kl_loss(z_means, z_vars) * self.kl_weight
    #             total_loss += loss.item()

    #             # Calculate accuracy
    #             probabilities = torch.sigmoid(preds)
    #             predicted_labels = (probabilities > 0.5).float()
    #             correct_predictions = (predicted_labels[:, -1] == labels_sample[:, -1].float()).float()
    #             accuracy = correct_predictions.mean()
    #             train_accuracies.append(accuracy)

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             # print(f'Loss: {loss}')
    #             if self.train_on_subset:
    #                 if (i // self.sample_length) >= self.train_break_threshold:
    #                     end_time = time.time()
    #                     elapsed_time = end_time - start_time
    #                     print(f"Time taken to run for loop training: {elapsed_time:.2f} seconds")
    #                     break
                
                
        # print(f'Training Loss: {loss}')
        # print(f'Training Accuracy (Max Context Length): {sum(train_accuracies)/len(train_accuracies)}')

    # def evaluate_autoregressive(self, train_val_dataloader, test_dataloader):
    #     # TODO: Change Logic
    #     self.encoder.eval()
    #     self.encoder.during_eval = True
    #     self.decoder.eval()

    #     p_embeds_test, labels_test = next(iter(test_dataloader))
    #     p_embeds_test = p_embeds_test.to(self.device)
    #     labels_test = labels_test.to(self.device)
    #     cs = torch.cat((p_embeds_test, labels_test.unsqueeze(-1)), dim=-1).to(self.device)

    #     zs, logprobs, z_means, z_vars = self.encoder(cs)
    #     posteriors = torch.exp(logprobs)
    #     preds = self.decoder(zs[:, :-1], p_embeds_test[:, 1:]).squeeze(-1)
        
    #     probabilities = torch.sigmoid(preds)
    #     predicted_labels = (probabilities > 0.5).float()
    #     correct_predictions = (predicted_labels == labels_test[:, 1:].float()).float()
    #     test_accuracy = correct_predictions.mean()
    #     # print(torch.mean(correct_predictions, dim=0))

    #     print(f'Test Accuracy (Autoregressive): {test_accuracy}')

    #     self.encoder.train()
    #     self.encoder.during_eval = False
    #     self.decoder.train()
            
    #     return test_accuracy

    def prepare_reference_set(self):
        train_val_p_embeds = []
        train_val_labels = []

        if self.ref_dataloader:
            train_val_set = self.ref_dataloader
            print("Start Preparing Eval Datasets Using Reference Set")
        else:
            train_val_set = self.train_dataloader
            print("Start Preparing Eval Datasets Using Random Sample from Training Set")

        for batch in train_val_set:
            p_embeds, labels = batch
            train_val_p_embeds.append(p_embeds)
            train_val_labels.append(labels)

        train_val_p_embeds = torch.cat(train_val_p_embeds, dim=0)
        train_val_labels = torch.cat(train_val_labels, dim=0)
        all_train_eval_p_embeds_subsets = []
        all_train_eval_labels_subsets = []
        for k in range(self.test_num):
            seed = self.test_seeds[k]
            torch.manual_seed(seed)
            # print(f"Evaluation Round {k} using seed {seed}")
            # Shuffle train_val_dataloader using seed
            # Get first self.sample_length entries (in question dimension) from train_val_dataloader
            # Perform Eval on the whole test_dataloader by loading the whole test_dataloader at once as a tensor

            num_samples = train_val_p_embeds.shape[1]
            indices = torch.randperm(num_samples)
            train_val_p_embeds = train_val_p_embeds[:, indices, :]
            train_val_labels = train_val_labels[:, indices]
            p_embeds_val = train_val_p_embeds[:, :self.sample_length, :]
            labels_val = train_val_labels[:, :self.sample_length]
            all_train_eval_p_embeds_subsets.append(p_embeds_val)
            all_train_eval_labels_subsets.append(labels_val)
        # print(f"train_eval_p_embeds shape: {train_val_p_embeds.shape}")
        # print(f"train_eval_labels shape: {train_val_labels.shape}")
        self.all_train_eval_p_embeds_subsets = all_train_eval_p_embeds_subsets
        self.all_train_eval_labels_subsets = all_train_eval_labels_subsets
        print([subset.shape for subset in self.all_train_eval_p_embeds_subsets])
        print([subset.shape for subset in self.all_train_eval_labels_subsets])

        p_embeds_test = []
        labels_test = []

        for batch in self.test_dataloader:
            p_embeds, labels = batch
            p_embeds_test.append(p_embeds)
            labels_test.append(labels)

        p_embeds_test = torch.cat(p_embeds_test, dim=0)
        labels_test = torch.cat(labels_test, dim=0)
        self.p_embeds_test = p_embeds_test
        self.labels_test = labels_test
        print(self.p_embeds_test.shape)
        print(self.labels_test.shape)
        print("Finish Preparing Eval Datasets")

    def evaluate_max_context(self):
        # TODO: Change Logic: Fix K seed and Test K times, each time shuffle using a seed and take first "sample_length" questions
        # TODO: And evaluate all questions left in test_dataloader
        self.encoder.eval()
        self.encoder.during_eval = True
        self.decoder.eval()

        # Collect all data from the train_val_dataloader and test_dataloader
        # train_val_p_embeds = []
        # train_val_labels = []

        # for batch in train_val_dataloader:
        #     p_embeds, labels = batch
        #     train_val_p_embeds.append(p_embeds)
        #     train_val_labels.append(labels)

        # train_val_p_embeds = torch.cat(train_val_p_embeds, dim=0)
        # train_val_labels = torch.cat(train_val_labels, dim=0)

        # all_train_eval_p_embeds_subsets = []
        # all_train_eval_labels_subsets = []

        # p_embeds_test = []
        # labels_test = []

        # for batch in test_dataloader:
        #     p_embeds, labels = batch
        #     p_embeds_test.append(p_embeds)
        #     labels_test.append(labels)

        # p_embeds_test = torch.cat(p_embeds_test, dim=0)
        # labels_test = torch.cat(labels_test, dim=0)
        # print(f"test_p_embeds shape: {p_embeds_test.shape}")
        # print(f"test_labels shape: {labels_test.shape}")
        
        test_accs = 0
        for k in range(self.test_num):
            # seed = self.test_seeds[k]
            # torch.manual_seed(seed)
            # print(f"Evaluation Round {k} using seed {seed}")
            # Shuffle train_val_dataloader using seed
            # Get first self.sample_length entries (in question dimension) from train_val_dataloader
            # Perform Eval on the whole test_dataloader by loading the whole test_dataloader at once as a tensor

            # num_samples = train_val_p_embeds.shape[1]
            # indices = torch.randperm(num_samples)
            # train_val_p_embeds = train_val_p_embeds[:, indices, :]
            # train_val_labels = train_val_labels[:, indices]
            # p_embeds_val = train_val_p_embeds[:, :self.sample_length, :]
            # labels_val = train_val_labels[:, :self.sample_length]
            p_embeds_val = self.all_train_eval_p_embeds_subsets[k].to(self.device)
            labels_val = self.all_train_eval_labels_subsets[k].to(self.device)

            cs = torch.cat((p_embeds_val, labels_val.unsqueeze(-1)), dim=-1).to(self.device)
            # print(cs.shape)
            zs, logprobs, z_means, z_vars = self.encoder(cs)

            p_embeds_test = self.p_embeds_test.to(self.device)
            labels_test = self.labels_test.to(self.device)
            batch_size, num_test_question, q_dim = p_embeds_test.shape
            # print(p_embeds_test.shape)
            posteriors = torch.exp(logprobs)
            preds = self.decoder(zs[:, -1, :].unsqueeze(1).repeat(1, num_test_question, 1), p_embeds_test).squeeze(-1)

            # Calculate accuracy
            probabilities = torch.sigmoid(preds)
            predicted_labels = (probabilities > 0.5).float()
            correct_predictions = (predicted_labels == labels_test.float()).float()
            test_accuracy = correct_predictions.mean()
            # print(f"Test Accuracy (Max Context Length) for round {k}: {test_accuracy}")
            test_accs += test_accuracy

        # total_loss = 0
        
        # Generate Z on Validation Set
        # p_embeds_val, labels_val = next(iter(train_val_dataloader))
        # p_embeds_val = p_embeds_val.to(self.device)
        # labels_val = labels_val.to(self.device)
        # cs = torch.cat((p_embeds_val, labels_val.unsqueeze(-1)), dim=-1).to(self.device)
        # print(cs.shape)
        # zs, logprobs, z_means, z_vars = self.encoder(cs)

        # p_embeds_test, labels_test = next(iter(test_dataloader))
        # p_embeds_test = p_embeds_test.to(self.device)
        # labels_test = labels_test.to(self.device)

        # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
        # batch_size, num_test_question, q_dim = p_embeds_test.shape
        # print(p_embeds_test.shape)
        # posteriors = torch.exp(logprobs)
        # # TODO: Check if this implementation is correct
        # preds = self.decoder(zs[:, -1, :].unsqueeze(1).repeat(1, num_test_question, 1), p_embeds_test).squeeze(-1)
        # loss = self.loss_fn(preds, labels_test.float())
        # loss = (loss * posteriors[:, :-1]).mean()
        # if self.use_kl:
        #     loss += self.kl_loss(z_means, z_vars) * self.kl_weight
        # total_loss += loss.item()
        
        # Calculate accuracy
        # probabilities = torch.sigmoid(preds)
        # predicted_labels = (probabilities > 0.5).float()
        # correct_predictions = (predicted_labels == labels_test.float()).float()
        # test_accuracy = correct_predictions.mean()

        # avg_loss = total_loss
        # avg_accuracy = accuracy
        # print(f'Test Loss: {total_loss}')
        print(f'Average Test Accuracy (Max Context Length) over {self.test_num} times: {test_accs/self.test_num}')

        self.encoder.train()
        self.encoder.during_eval = False
        self.decoder.train()
            
        return test_accs/self.test_num
    
    def prepare_clustered_reference_set(self):
        clustered_reference_set = {}
        for (index, (cluster_x, cluster_y)) in enumerate(self.train_clusters):
            one_cluster_ref_set = []
            for k in range(self.test_num):
                seed = self.test_seeds[k]
                torch.manual_seed(seed)

                indices = torch.randperm(cluster_x.shape[1])
                cluster_x = cluster_x[:, indices, :]
                cluster_y = cluster_y[:, indices]
                cluster_x_sample = cluster_x[:, :self.sample_length, :]
                cluster_y_sample = cluster_y[:, :self.sample_length]
                one_cluster_ref_set.append((cluster_x_sample, cluster_y_sample))
            clustered_reference_set[index] = one_cluster_ref_set
        self.clustered_reference_set = clustered_reference_set
        return
    
    def evaluate_clustered(self):
        self.encoder.eval()
        self.encoder.during_eval = True
        self.decoder.eval()
        
        clustered_test_accs = {i:0 for i in range(len(self.train_clusters))}

        for index in range(len(self.train_clusters)):
            test_cluster_x, test_cluster_y = self.test_clusters[index]
            # print(test_cluster_x.shape, test_cluster_y.shape)
            ref_cluster_set = self.clustered_reference_set[index]
            for k in range(self.test_num):
                ref_cluster_x_sample, ref_cluster_y_sample = ref_cluster_set[k]
                ref_cluster_x_sample = ref_cluster_x_sample.to(self.device)
                ref_cluster_y_sample = ref_cluster_y_sample.to(self.device)
                cs = torch.cat((ref_cluster_x_sample, ref_cluster_y_sample.unsqueeze(-1)), dim=-1).to(self.device)
                # print(cs.shape)
                zs, logprobs, z_means, z_vars = self.encoder(cs)

                test_cluster_x = test_cluster_x.to(self.device)
                test_cluster_y = test_cluster_y.to(self.device)
                batch_size, num_test_question, q_dim = test_cluster_x.shape
                # print(p_embeds_test.shape)
                posteriors = torch.exp(logprobs)
                preds = self.decoder(zs[:, -1, :].unsqueeze(1).repeat(1, num_test_question, 1), test_cluster_x).squeeze(-1)

                # Calculate accuracy
                probabilities = torch.sigmoid(preds)
                predicted_labels = (probabilities > 0.5).float()
                correct_predictions = (predicted_labels == test_cluster_y.float()).float()
                test_accuracy = correct_predictions.mean()
                # print(f"Test Accuracy (Max Context Length) for round {k}: {test_accuracy}")
                clustered_test_accs[index] += test_accuracy

        clustered_test_accs = {key: value / self.test_num for key, value in clustered_test_accs.items()}
        # print(clustered_test_accs)
        final_avg_test_acc = sum(clustered_test_accs.values())/len(clustered_test_accs.values())
        print(f'Average test accuracy over {self.test_num} times over all clusters: {final_avg_test_acc}')

        self.encoder.train()
        self.encoder.during_eval = False
        self.decoder.train()
            
        return final_avg_test_acc
    
if __name__ == "__main__":
    # Set seed for reproducibility
    SEED = 42 
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # If using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU setups
    
    assert torch.cuda.is_available(), "No GPU Available"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    BATCH_SIZE = 16
    # TODO: Try OpenAI's Ada Embedding (remember to cache)
    SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
    EMBEDDING_DIM = 768
    NUM_EPOCHS = 20

    BASE_MODEL_ONLY = True
    TRAIN_ON_SUBSET = False
    TRAIN_BREAK_THRESHOLD = 1
    SAMPLE_LENGTH = 200
    AR_TRAIN = True # Whether to train autoregressively or just using max context length
    AR_EVAL = False # Whether to evaluate autoregressively or just using max context length

    Z_DIM = 192 # Z_DIM choices: 32,64,96,128
    USE_KL = True
    KL_WEIGHT = 3 # Weight choices: 1,3,5,10
    ENCODER_USE_LINEAR = True
    ENCODER_USE_LAYERNORM = True
    DECODER_USE_LINEAR = False
    DECODER_USE_LAYERNORM = False
    DECODER_NORMALIZE = False
    USE_CONCAT = True
    USE_CLUSTERING = True
    NUM_CLUSTERS = 10
    LR = 3e-4

    print(f"Using sample_length={SAMPLE_LENGTH}, z_dim={Z_DIM}, kl_weight={KL_WEIGHT}, encoder_linear={ENCODER_USE_LINEAR}, encoder_layernorm={ENCODER_USE_LAYERNORM}, decoder_linear={DECODER_USE_LINEAR}, decoder_layernorm={DECODER_USE_LAYERNORM}, decoder_normalize={DECODER_NORMALIZE}")
    print("Start Initializing Dataset...")
    train_x, train_y, train_val_x, train_val_y, val_x, val_y, test_x, test_y = load_data(pwd=os.getcwd())
    print(train_x.shape)
    print(train_y.shape)

    if USE_CLUSTERING:
        print("Start Clustering")
        # Perform clustering
        train_clusters, kmeans_model = create_clusters(train_x, NUM_CLUSTERS)

        # Create cluster batches
        train_cluster_batches = create_cluster_batches(train_x, train_y, train_clusters)

        # Print shapes of each batch to verify
        # for i, (batch_x, batch_y) in enumerate(cluster_batches):
        #     print(f"Batch {i}:")
        #     print(f"  x shape: {batch_x.shape}")
        #     print(f"  y shape: {batch_y.shape}") 
        test_clusters = kmeans_model.predict(val_x[0])
        test_cluster_to_indices = {i: [] for i in range(NUM_CLUSTERS)}
        for idx, cluster in enumerate(test_clusters):
            test_cluster_to_indices[cluster].append(idx)
        # print(test_clusters)
        test_cluster_batches = create_cluster_batches(val_x, val_y, test_cluster_to_indices)
        print("Finish Clustering")
    else:
        train_cluster_batches = None
        kmeans_model = None
        test_cluster_batches = None

    train_dataset = CustomDataset(train_x, train_y)
    train_val_dataset = CustomDataset(train_val_x, train_val_y)
    val_dataset = CustomDataset(val_x, val_y)
    test_dataset = CustomDataset(test_x, test_y)
    print("Finish Initializing Dataset")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_val_dataloader = DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    encoder = Encoder(c_dim=EMBEDDING_DIM+1, z_dim=Z_DIM, linear=ENCODER_USE_LINEAR, layernorm=ENCODER_USE_LAYERNORM)
    decoder = Decoder(q_dim=EMBEDDING_DIM, z_dim=Z_DIM, use_concat=USE_CONCAT, linear=DECODER_USE_LINEAR, normalize=DECODER_NORMALIZE)
    trainer = Trainer(encoder, decoder, SAMPLE_LENGTH, 
                    train_dataloader=train_dataloader, test_dataloader=val_dataloader, ref_dataloader=train_val_dataloader,
                    lr=LR, use_kl=USE_KL, kl_weight = KL_WEIGHT, device=device, train_on_subset=TRAIN_ON_SUBSET,
                    train_break_threshold=TRAIN_BREAK_THRESHOLD, use_clustering=USE_CLUSTERING,
                    train_clusters=train_cluster_batches, kmeans_model=kmeans_model, test_clusters=test_cluster_batches)

    trainer.train(epochs=NUM_EPOCHS, ar_train=AR_TRAIN, ar_eval=AR_EVAL)