from ray import tune
from baseline.mf import TextMF, train_recsys_rating, split_and_load
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import torch


def tune_mf(config):
    embedding_dim = config["embedding_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    alpha = config["alpha"]
    (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    ) = split_and_load(batch_size=batch_size, subset_size=None, base_model_only=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TextMF(
        dim=embedding_dim,
        num_models=num_models,
        num_prompts=num_prompts,
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
    "embedding_dim": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "num_epochs": tune.choice([10, 20, 30, 40]),
    "alpha": tune.choice([1, 0.1, 0.01, 0.005, 0.001, 0.2]),
}
tune_mf_with_resources = tune.with_resources(tune_mf, {"gpu": 0.5})

tuner = tune.Tuner(
    tune_mf_with_resources,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=300,
        scheduler=ASHAScheduler(metric="test_acc", mode="max"),
    ),
)

results = tuner.fit()
print(results.get_best_result(metric="test_acc", mode="max"))
print("Best config:", results.get_best_result(metric="test_acc", mode="max").config)
