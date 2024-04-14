import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):  # Hidden size is a list
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        original_shape = x.shape  # Save the original shape
        if len(original_shape) > 2:
            # Reshape to combine 'batch_size' and 'len' into a single dimension
            x = x.view(-1, original_shape[-1])

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        if (
            len(original_shape) > 2
        ):  # Reshape back to the original batch and num dimensions
            x = x.view(original_shape[0], original_shape[1], -1)

        return x


class Encoder(nn.Module):  # Input a sequence of questions and output z
    def __init__(self, q_dim=768, z_dim=128):
        super().__init__()
        self.mean_network = MLP(
            input_size=q_dim, hidden_sizes=[512, 256], output_size=z_dim
        )
        self.variance_network = MLP(
            input_size=q_dim, hidden_sizes=[512, 256], output_size=z_dim
        )

    def forward(
        self, qs
    ):  # Input multiple instances of context and output a latent representation of the task
        # Input qs: [batch_size, len, prompt_embed_dim], batch_size = num_models
        self.calculate_posterior(qs)
        return self.z  # Output Shape: [batch_size, len, z_dim]

    def sample_z(self):  # Sample from a multivariate gaussian
        # TODO: Rewrite logic and need to fit shape
        posteriors = [
            torch.distributions.Normal(m, torch.sqrt(s))
            for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))
        ]
        z = [d.rsample() for d in posteriors]
        self.z = torch.stack(z)

    def _product_of_gaussians(self, mus, sigmas_squared):
        # TODO: Shape Matching
        """compute mu, sigma of product of gaussians"""
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared

    def calculate_posterior(
        self, context
    ):  # Calculate posterior gaussian from gaussian factors given by c's
        """compute q(z|c) as a function of input context and sample new z from it"""
        # Input context: [batch_size(num_models), len(num_sample), prompt_embed_dim]
        mu = self.mean_network(context)  # Shape: [batch_size(num_models), len(num_sample), z_dim]
        sigma_squared = F.softplus(
            self.variance_network(context)
        )  # Shape: [batch_size(num_models), len(num_sample), z_dim]
        # TODO: Shape Matching
        z_params = [
            self._product_of_gaussians(m, s)
            for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
        ]  # Shape: tuple of 2 [batch_size(num_models), len(num_sample), z_dim]
        self.z_means = torch.stack([p[0] for p in z_params])
        self.z_vars = torch.stack([p[1] for p in z_params])
        self.sample_z()


class Decoder(nn.Module):  # Get input from encoder which is z and a new question, concat or project to same dim and dot product
    def __init__(self, q_dim=768, z_dim=128):
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
    def __init__(self):
        data = load_dataset("RZ412/mmlu_responses_1k_augmented")
        num_models = len(data['train'][0]['answers'])

        all_questions = data['train']['test_questions']
        model = SentenceTransformer('all-mpnet-base-v2')
        question_vectors = model.encode(all_questions)
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

class Trainer:  # autoregressively input k question and get k+1_th questions' answer
    def __init__(self, encoder, decoder, train_dataloader, test_dataloader=None):
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.use_kl = True

    def train(self, epochs=10):
        for epoch in range(len(epochs)):
            self.train_epoch(self.train_dataloader)

            if self.test_dataloader:
                eval_results = self.evaluate(self.test_dataloader)
                print(eval_results)

    def kl_loss(self): # TODO: Rewrite
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) 
                      for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior) for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def train_epoch(self, dataloader, subset_length=50):
        for batch in dataloader:
            # batch: ([batch_size, num_total_question, prompt_embed_dim], [batch_size, num_total_question])
            p_embeds, labels = batch

            batch_size, num_total_question, q_dim = p_embeds.shape
            #TODO: Need to make same indices for x and y
            indices_x = torch.randint(num_total_question, (batch_size, subset_length))
            # Shape: [batch_size, subset_length, prompt_embed_dim]
            p_embeds = torch.gather(p_embeds, 1, indices_x.unsqueeze(-1).expand(-1, -1, q_dim))
            indices_y = torch.randint(num_total_question, size=(batch_size, subset_length))
            labels = torch.gather(labels, 1, indices_y) # Shape: [batch_size, subset_length]

            zs, posteriors = self.encoder.sample_z(p_embeds, labels)
            preds = self.decoder(zs[:, :-1], p_embeds[:, 1:])
            loss = self.loss_fn(preds, labels[:, 1:])
            loss = (loss * posteriors[:, :-1]).mean()
            if self.use_kl:
                loss += self.kl_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, dataloader):
        for batch in dataloader:
            pass
