from ray import tune
from baseline.knn import get_train_test, evaluate
from ray.tune.schedulers import ASHAScheduler
import numpy as np

train_ls, test_ls = get_train_test()


def tune_knn(config, data):
    train_ls, test_ls = data[0], data[1]
    info = evaluate(train_ls, test_ls, num_neighbors=config["num_neighbors"])
    return info


search_space = {"num_neighbors": tune.sample_from(lambda _: np.random.randint(1, 200))}
tune_knn_with_resources = tune.with_resources(tune_knn, {"cpu": 1})

tuner = tune.Tuner(
    tune.with_parameters(tune_knn_with_resources, data=[train_ls, test_ls]),
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=4, scheduler=ASHAScheduler(metric="mean_accuracy", mode="max")),
)

results = tuner.fit()
print(results.get_best_result(metric="mean_accuracy", mode="max"))
print("Best config:", results.get_best_result(metric="mean_accuracy", mode="max").config)
