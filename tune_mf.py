from ray import tune
from baseline.mf import TextMF, train_recsys_rating, split_and_load
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import numpy as np
import torch
import pandas as pd
import os

pwd = os.getcwd()
global_train_data = pd.read_csv(f"{pwd}/data/mmlu_train.csv")
global_test_data = pd.read_csv(f"{pwd}/data/mmlu_test.csv")


def tune_mf(config, global_train_data, global_test_data):
    embedding_dim = config["embedding_dim"]
    batch_size = 128
    num_epochs = 50
    alpha = config["alpha"]
    (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    ) = split_and_load(
        batch_size=batch_size, subset_size=None, base_model_only=True, global_train_data=global_train_data, global_test_data=global_test_data
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TextMF(
        dim=embedding_dim,
        num_models=num_models,
        num_prompts=1000,  # TODO: Fix this
        num_classes=num_classes,
        alpha=alpha,
    ).to(device)

    train_recsys_rating(
        model,
        train_loader,
        test_loader,
        num_models,
        num_prompts,
        batch_size,
        num_epochs,
        devices=[device],
    )


search_space = {
    "embedding_dim": tune.choice([4 * k for k in range(32)]),
    "alpha": tune.choice([1, 0.1, 0.01, 0.005, 0.001, 0.2]),
}
tune_mf_with_resources = tune.with_resources(tune_mf, {"gpu": 1})

tuner = tune.Tuner(
    tune.with_parameters(tune_mf_with_resources, global_train_data=global_train_data, global_test_data=global_test_data),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=300,
        search_alg=OptunaSearch(metric="test_acc", mode="max"),
        # scheduler=ASHAScheduler(metric="test_acc", mode="max"),
    )
)

results = tuner.fit()
print(results.get_best_result(metric="test_acc", mode="max"))
print("Best config:", results.get_best_result(metric="test_acc", mode="max").config)
