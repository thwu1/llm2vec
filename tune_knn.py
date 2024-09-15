from ray import tune
from baseline_new.knn import load_data, evaluate
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import os

def tune_knn(config, data):
    train_x, train_y, test_x, test_y = data
    info = evaluate(train_x, train_y, test_x, test_y, config["num_neighbors"])
    return info

TRAIN_X_PATH = f'data_new/new_train_x.pth'
TRAIN_Y_PATH = f'data_new/new_train_y.pth'
TEST_X_PATH = f'data_new/new_val_x.pth'
TEST_Y_PATH = f'data_new/new_val_y.pth'
print(f"Start Initializing Dataset...")
# train_x, train_y, val_x, val_y, test_x, test_y = save_data(path="../data_new/all_responses.pth")
train_x, train_y, test_x, test_y = load_data(TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH, TEST_Y_PATH)
print(f"Finish Initializing Dataset")

search_space = {"num_neighbors": tune.grid_search(range(1, 200, 1))}
tune_knn_with_resources = tune.with_resources(tune_knn, {"cpu": 1})

tuner = tune.Tuner(
    tune.with_parameters(
        tune_knn_with_resources,
        data=[train_x, train_y, test_x, test_y],
    ),
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
)

results = tuner.fit()
print(results.get_best_result(metric="mean_accuracy", mode="max"))
print("Best config:", results.get_best_result(metric="mean_accuracy", mode="max").config)
