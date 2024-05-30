from ray import tune
from baseline_new.pc import load_data, CustomDataset, Encoder, Decoder, Trainer
from torch.utils.data import Dataset, DataLoader, Subset
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import numpy as np
import torch
import pandas as pd
import os
import random

# Set seed for reproducibility
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If using CUDA (PyTorch with GPU)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

def tune_pc(config, train_dataloader, val_dataloader, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_transformer = "all-mpnet-base-v2"
    embedding_dim = 768
    use_kl = True
    base_model_only = True
    train_on_subset = False
    lr = 1e-3
    num_epochs = 20
    batch_size = 128
    z_dim = config["z_dim"] # Z_DIM choices: 32,64,96,128
    kl_weight = config["kl_weight"] # Weight choices: 1,3,5,10
    use_linear = config["use_linear"]
    use_layernorm = config["use_layernorm"]
    sample_length = config["sample_length"] # Sample length choices: 50 or 100

    encoder = Encoder(c_dim=embedding_dim+1, z_dim=z_dim, linear=use_linear, layernorm=use_layernorm)
    decoder = Decoder(q_dim=embedding_dim, z_dim=z_dim)
    trainer = Trainer(encoder, decoder, sample_length, train_dataloader, val_dataloader, test_dataloader, 
                  lr=lr, use_kl=use_kl, kl_weight = kl_weight, device=device, train_on_subset=train_on_subset)

    max_accuracy = trainer.train(epochs=num_epochs, ar_train=True, ar_eval=False)
    return {"test_acc": max_accuracy}

search_space = {
    "z_dim": tune.choice([32, 64, 96, 128, 192, 256, 512]),
    "kl_weight": tune.choice([1, 3, 5, 10, 20]),
    "use_linear": tune.choice([True, False]),
    "use_layernorm": tune.choice([True, False]),
    "sample_length": tune.choice([50,100,200,500,1000]),
}

BASE_MODEL_ONLY = True
BATCH_SIZE = 128

(train_x, train_y, val_x, val_y, test_x, test_y) = load_data(pwd = os.getcwd())

train_dataset = CustomDataset(train_x, train_y)
val_dataset = CustomDataset(val_x, val_y)
test_dataset = CustomDataset(test_x, test_y)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

tune_pc_with_resources = tune.with_resources(tune_pc, {"gpu": 0.5})

tuner = tune.Tuner(
    tune.with_parameters(tune_pc_with_resources, train_dataloader=train_dataloader, 
                         val_dataloader=val_dataloader, test_dataloader=test_dataloader),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=200,
        # search_alg=OptunaSearch(metric="test_acc", mode="max"),
        scheduler=ASHAScheduler(metric="test_acc", mode="max"),
    )
)

results = tuner.fit()
print(results.get_best_result(metric="test_acc", mode="max"))
print("Best config:", results.get_best_result(metric="test_acc", mode="max").config)