from ray import tune
from baseline.knn import load_data, evaluate
from ray.tune.schedulers import ASHAScheduler
import numpy as np

BASE_MODEL_ONLY = True
model_order, train_prompt_order, val_prompt_order, test_prompt_order, train_x, train_y, val_x, val_y, test_x, test_y = load_data(
    base_model_only=BASE_MODEL_ONLY
)


def tune_knn(config, data):
    model_order, train_prompt_order, val_prompt_order, test_prompt_order, train_x, train_y, val_x, val_y, test_x, test_y = data
    info = evaluate(model_order, train_x, test_x, train_y, test_y, config["num_neighbors"])
    return info


search_space = {"num_neighbors": tune.grid_search(range(11, 201))}
tune_knn_with_resources = tune.with_resources(tune_knn, {"cpu": 1})

tuner = tune.Tuner(
    tune.with_parameters(
        tune_knn_with_resources,
        data=[model_order, train_prompt_order, val_prompt_order, test_prompt_order, train_x, train_y, val_x, val_y, test_x, test_y],
    ),
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
)

results = tuner.fit()
print(results.get_best_result(metric="mean_accuracy", mode="max"))
print("Best config:", results.get_best_result(metric="mean_accuracy", mode="max").config)
