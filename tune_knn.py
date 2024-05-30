from ray import tune
from baseline_new.knn import load_data, evaluate
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import os

def tune_knn(config, data):
    train_x, train_y, val_x, val_y, test_x, test_y = data
    info = evaluate(train_x, train_y, val_x, val_y, config["num_neighbors"])
    return info

train_x, train_y, val_x, val_y, test_x, test_y = load_data(pwd=os.getcwd())

search_space = {"num_neighbors": tune.grid_search(range(0, 150, 1))}
tune_knn_with_resources = tune.with_resources(tune_knn, {"cpu": 1})

tuner = tune.Tuner(
    tune.with_parameters(
        tune_knn_with_resources,
        data=[train_x, train_y, val_x, val_y, test_x, test_y],
    ),
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
)

results = tuner.fit()
print(results.get_best_result(metric="mean_accuracy", mode="max"))
print("Best config:", results.get_best_result(metric="mean_accuracy", mode="max").config)
