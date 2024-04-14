import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

pwd = os.getcwd()


def get_train_test():
    train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
    test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")

    train_data_group = [
        group for _, group in train_data.sort_values("model_id").groupby("model_id")
    ]
    test_data_group = [
        group for _, group in test_data.sort_values("model_id").groupby("model_id")
    ]

    return train_data_group, test_data_group


class gaussian_encoder(nn.Module):
    def __init__(self, input_size, output_size):  # How to setup?
        super(gaussian_encoder, self).__init__()
        # Need a mean and and variance network which can be just MLP

    def forward(self, x):
        return x


class InferenceNetwork(nn.Module):
    def __init__(self, input_size, output_size):  # How to train?
        super(InferenceNetwork, self).__init__()
        self.encoder = gaussian_encoder()

    def forward(self, x):
        return x


model = InferenceNetwork(input_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)  # What is the target?
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
