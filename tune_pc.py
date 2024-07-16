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
from sklearn.cluster import KMeans

# Set seed for reproducibility
SEED = 42 
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# If using CUDA (PyTorch with GPU)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU setups

NUM_TESTS = 5
TEST_SEEDS = [random.randint(1, 1000) for i in range(NUM_TESTS)]

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

def tune_pc(config, train_x, train_y, test_x, test_y,
            train_dataloader, train_val_dataloader, val_dataloader, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_transformer = "all-mpnet-base-v2"
    embedding_dim = 768
    use_kl = True
    base_model_only = True
    train_on_subset = False
    lr = 3e-4
    num_epochs = 20
    batch_size = 16
    z_dim = config["z_dim"] # Z_DIM choices: 32,64,96,128
    use_concat = True
    kl_weight = config["kl_weight"] # Weight choices: 1,3,5,10
    use_linear = config["use_linear"]
    use_layernorm = True
    sample_length = config["sample_length"] # Sample length choices: 50 or 100
    num_clusters = config["num_clusters"]
    USE_CLUSTERING = True

    if USE_CLUSTERING:
        print("Start Clustering")
        # Perform clustering
        train_clusters, kmeans_model = create_clusters(train_x, num_clusters)

        # Create cluster batches
        train_cluster_batches = create_cluster_batches(train_x, train_y, train_clusters)

        # Print shapes of each batch to verify
        # for i, (batch_x, batch_y) in enumerate(cluster_batches):
        #     print(f"Batch {i}:")
        #     print(f"  x shape: {batch_x.shape}")
        #     print(f"  y shape: {batch_y.shape}") 
        test_clusters = kmeans_model.predict(test_x[0])
        test_cluster_to_indices = {i: [] for i in range(num_clusters)}
        for idx, cluster in enumerate(test_clusters):
            test_cluster_to_indices[cluster].append(idx)
        # print(test_clusters)
        test_cluster_batches = create_cluster_batches(test_x, test_y, test_cluster_to_indices)
        print("Finish Clustering")

    encoder = Encoder(c_dim=embedding_dim+1, z_dim=z_dim, linear=use_linear, layernorm=use_layernorm)
    decoder = Decoder(q_dim=embedding_dim, z_dim=z_dim, use_concat=use_concat)
    trainer = Trainer(encoder, decoder, sample_length, 
                train_dataloader=train_dataloader, test_dataloader=test_dataloader, ref_dataloader=train_val_dataloader,
                lr=lr, use_kl=use_kl, kl_weight = kl_weight, device=device, train_on_subset=train_on_subset,
                use_clustering=USE_CLUSTERING, 
                train_clusters=train_cluster_batches, kmeans_model=kmeans_model, test_clusters=test_cluster_batches)

    max_accuracy = trainer.train(epochs=num_epochs, ar_train=True, ar_eval=False)
    return {"test_acc": max_accuracy}

search_space = {
    "z_dim": tune.grid_search([192, 256]),
    "kl_weight": tune.grid_search([1, 5]),
    "use_linear": tune.grid_search([True]),
    "sample_length": tune.grid_search([100,200]),
    "num_clusters": tune.grid_search([6,10,25]),
}

BASE_MODEL_ONLY = True
BATCH_SIZE = 16

print("Start Initializing Dataset...")
train_x, train_y, train_val_x, train_val_y, val_x, val_y, test_x, test_y = load_data(pwd=os.getcwd())
train_dataset = CustomDataset(train_x, train_y)
train_val_dataset = CustomDataset(train_val_x, train_val_y)
val_dataset = CustomDataset(val_x, val_y)
test_dataset = CustomDataset(test_x, test_y)
print("Finish Initializing Dataset")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_val_dataloader = DataLoader(dataset=train_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

tune_pc_with_resources = tune.with_resources(tune_pc, {"gpu": 1})

tuner = tune.Tuner(
    tune.with_parameters(tune_pc_with_resources, train_x = train_x, train_y = train_y,
                         test_x = test_x, test_y = test_y,
                         train_dataloader=train_dataloader, train_val_dataloader = train_val_dataloader,
                         val_dataloader=val_dataloader, test_dataloader=test_dataloader),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        # num_samples=7,
        # search_alg=OptunaSearch(metric="test_acc", mode="max"),
        scheduler=ASHAScheduler(metric="test_acc", mode="max"),
    )
)

results = tuner.fit()
print(results.get_best_result(metric="test_acc", mode="max"))
print("Best config:", results.get_best_result(metric="test_acc", mode="max").config)