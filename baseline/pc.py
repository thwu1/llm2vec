import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):  # hidden_sizes is a list
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(input_size, hidden_sizes[0]))
        self.mlp.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            self.mlp.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class Encoder(nn.Module):  # Input a sequence of questions and output z
    def __init__(self, c_dim, z_dim):
        super().__init__()
        self.mean_network = MLP(input_size=c_dim, hidden_sizes=[512, 256], output_size=z_dim)
        self.variance_network = MLP(input_size=c_dim, hidden_sizes=[512, 256], output_size=z_dim)

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
    def __init__(self, q_dim, z_dim):
        super().__init__()
        self.network = MLP(
            input_size=z_dim + q_dim, hidden_sizes=[128, 16], output_size=1
        )

    def forward(self, zs, qs):
        # zs: [batch_size, len, z_dim]
        # qs: [batch_size, len, q_dim], batch_size = num_models
        # return [batch_size, len]
        x = torch.cat(
            (zs, qs), dim=-1
        )  # concat z and question, x: [batch_size, len, (z_dim + q_dim)]
        return self.network(x)  # Output 1D binary result 0,1

class CustomDataset(Dataset):
    # want "for batch in dataloader, and batch = (x,y)"
    # batch_size = num_model
    # x: (batch_size, num_total_questions, question_embedding_dim)
    # y: (batch_size, num_total_questions)
    def __init__(self, sentence_transformer):
        data = load_dataset("RZ412/mmlu_responses_1k_augmented")
        num_models = len(data['train'][0]['answers'])

        all_questions = data['train']['test_questions']
        model = SentenceTransformer(sentence_transformer)
        question_vectors = model.encode(all_questions, show_progress_bar=True)
        x = torch.tensor(question_vectors)
        x = x.unsqueeze(0).repeat(num_models, 1, 1)

        y = []
        for row in data['train']:
            model_order = [element['model'] for element in row['answers']]
            self.model_order = model_order
            option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            ref_answer = option_map[row['reference_answers']]
            correctness_result = [int(element['answer'] == ref_answer) for element in row['answers']]
            y.append(correctness_result)
        y = torch.tensor(y).transpose(0,1)

        self.data = x
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # Shape (num_questions, question_embedding_dim)
        y = self.labels[idx]  # Shape (num_models, num_questions)
        
        return x, y

class Trainer:  # batch-wise autoregressively input k question and get (k+1)_th questions' answer
    def __init__(self, encoder, decoder, subset_length, train_dataloader, test_dataloader=None, use_kl=True):
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.subset_length = subset_length
        self.use_kl = use_kl

    def train(self, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}: ")
            self.train_epoch(self.train_dataloader)

            if self.test_dataloader:
                eval_results = self.evaluate(self.test_dataloader)

    def kl_loss(self, z_means, z_vars):
        #TODO: KL Loss extremely large
        batch_size, subset_length, z_dim = z_means.shape
        prior = torch.distributions.Normal(torch.zeros(z_dim), torch.eye(z_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) 
                      for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior) for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def train_epoch(self, dataloader):
        # TODO: Within each batch can iterate through different subset samples
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        for batch in tqdm(dataloader):
            # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
            p_embeds, labels = batch
            batch_size, num_total_question, q_dim = p_embeds.shape
            # TODO: Add an option of random sampling of subset size (Exponential Distribution, clamp at 30-900)
            subset_indices = torch.randint(num_total_question, (batch_size, self.subset_length))
            # Shape: [batch_size, subset_length, prompt_embed_dim]
            p_embeds = torch.gather(p_embeds, 1, subset_indices.unsqueeze(-1).expand(-1, -1, q_dim))
            # Shape: [batch_size, subset_length]
            labels = torch.gather(labels, 1, subset_indices)
            cs = torch.cat((p_embeds, labels.unsqueeze(-1)), dim=-1)

            zs, logprobs, z_means, z_vars = self.encoder(cs)
            posteriors = torch.exp(logprobs) # TODO: Logprobs incorrect
            # print(posteriors)
            preds = self.decoder(zs[:, :-1], p_embeds[:, 1:]).squeeze(-1)
            loss = self.loss_fn(preds, labels[:, 1:].float())
            # print(loss)
            loss = (loss * posteriors[:, :-1]).mean()
            # print(loss)
            if self.use_kl:
                loss += self.kl_loss(z_means, z_vars)
                # print(loss)
            total_loss += loss.item()

            # Calculate accuracy
            probabilities = torch.sigmoid(preds)
            predicted_labels = (probabilities > 0.5).float()
            correct_predictions = (predicted_labels == labels[:, 1:].float()).float()
            accuracy = correct_predictions.mean()
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            num_batches += 1
            # print(f'Loss: {loss}')

        # print(f'Training Loss: {avg_loss}')
        print(f'Training Accuracy: {accuracy}')

    def evaluate(self, dataloader):
        # TODO: Add one more accuracy evaluation method (max sample and evaluate all questions left)
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                p_embeds, labels = batch
                batch_size, num_total_question, q_dim = p_embeds.shape

                subset_indices = torch.randint(num_total_question, (batch_size, self.subset_length))
                p_embeds = torch.gather(p_embeds, 1, subset_indices.unsqueeze(-1).expand(-1, -1, q_dim))
                labels = torch.gather(labels, 1, subset_indices)

                cs = torch.cat((p_embeds, labels.unsqueeze(-1)), dim=-1)
                zs, logprobs, z_means, z_vars = self.encoder(cs)

                preds = self.decoder(zs[:, :-1], p_embeds[:, 1:]).squeeze(-1)
                loss = self.loss_fn(preds, labels[:, 1:].float())
                loss = (loss * torch.exp(logprobs[:, :-1])).mean()  # Apply posterior as weights
                if self.use_kl:
                    loss += self.kl_loss(z_means, z_vars)

                # Calculate accuracy
                probabilities = torch.sigmoid(preds)
                predicted_labels = (probabilities > 0.5).float()
                correct_predictions = (predicted_labels == labels[:, 1:].float()).float()
                accuracy = correct_predictions.mean()
                total_accuracy += accuracy
                    
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
            # print(f'Test Loss: {avg_loss}')
            print(f'Test Accuracy: {avg_accuracy}')
            
            self.encoder.train()
            self.decoder.train()
            
            return avg_loss

def train_test_split(dataset, test_size=0.1, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(len(dataset) * (1 - test_size))
    
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, test_subset

BATCH_SIZE = 256
# TODO: Try OpenAI's Ada Embedding (remember to cache)
SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
Z_DIM = 64
SUBSET_LENGTH = 50
print("Start Initializing Dataset...")
dataset = CustomDataset(sentence_transformer=SENTENCE_TRANSFORMER)
print("Finish Initializing Dataset")
train_subset, test_subset = train_test_split(dataset, test_size=0.2)
train_dataloader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False)
encoder = Encoder(c_dim=EMBEDDING_DIM+1, z_dim=Z_DIM)
decoder = Decoder(q_dim=EMBEDDING_DIM, z_dim=Z_DIM)
# TODO: Try using KL Loss
trainer = Trainer(encoder, decoder, SUBSET_LENGTH, train_dataloader, test_dataloader, use_kl=True)

trainer.train(epochs=20)