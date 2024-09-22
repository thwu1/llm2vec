import wandb,json,random,argparse,ray,time,torch,os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm
from ray.train import Checkpoint

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

pwd = os.getcwd()

def split_and_load(global_train_data, global_test_data, batch_size=64, subset_size=None, base_model_only=False):
    # print(train_data.head())
    # print(test_data.head())

    if base_model_only:
        train_data = global_train_data[~global_train_data["model_name"].str.contains("vote|moe")].reset_index(drop=True)
        test_data = global_test_data[~global_test_data["model_name"].str.contains("vote|moe")].reset_index(drop=True)
    elif subset_size:
        model_subset = random.sample(list(test_data["model_name"]), subset_size)
        # print(model_subset[:10])
        train_data = global_train_data[global_train_data["model_name"].isin(model_subset)].reset_index(drop=True)
        test_data = global_test_data[global_test_data["model_name"].isin(model_subset)].reset_index(drop=True)
        # print(train_data.head())
        # print(test_data.head())

    max_category = max(train_data["category_id"].max(), test_data["category_id"].max())
    min_category = min(train_data["category_id"].min(), test_data["category_id"].min())
    num_categories = max_category - min_category + 1

    class CustomDataset(Dataset):
        def __init__(self, data):
            # print(data["model_id"])
            model_ids = torch.tensor(data["model_id"], dtype=torch.int64)
            # Get unique model IDs and their corresponding new indices
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            # Map original IDs to their ranks
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids

            # print("Original IDs:", model_ids)
            # print("Ranked IDs:", ranked_model_ids)

            # print(self.models)
            self.prompts = torch.tensor(data["prompt_id"], dtype=torch.int64)
            self.labels = torch.tensor(data["label"], dtype=torch.int64)
            self.categories = torch.tensor(data["category_id"], dtype=torch.int64)
            self.num_models = len(data["model_id"].unique())
            self.num_prompts = len(data["prompt_id"].unique())
            self.num_classes = len(data["label"].unique())
            print(f"number of models: {self.num_models}, number of prompts: {self.num_prompts}")

        def get_num_models(self):
            return self.num_models

        def get_num_prompts(self):
            return self.num_prompts

        def get_num_classes(self):
            return self.num_classes

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return (
                self.models[index],
                self.prompts[index],
                self.labels[index],
                self.categories[index],
            )

        def get_dataloaders(self, batch_size):
            return DataLoader(self, batch_size, shuffle=False)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    num_models = train_dataset.get_num_models()
    num_prompts = train_dataset.get_num_prompts() + test_dataset.get_num_prompts()
    num_classes = train_dataset.get_num_classes()

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = test_dataset.get_dataloaders(batch_size)

    MODEL_NAMES = list(np.unique(list(train_data["model_name"])))
    return (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    )


def create_router_dataloader(original_dataloader):
    # Step 1: Concatenate all batches from original dataloader into single tensors
    all_models = []
    all_prompts = []
    all_labels = []
    all_categories = []

    for batch in original_dataloader:
        models, prompts, labels, categories = batch
        all_models.append(models)
        all_prompts.append(prompts)
        all_labels.append(labels)
        all_categories.append(categories)

    all_models = torch.cat(all_models)
    all_prompts = torch.cat(all_prompts)
    all_labels = torch.cat(all_labels)
    all_categories = torch.cat(all_categories)

    # Step 2: Create dictionaries to store labels and categories
    label_dict = {}
    category_dict = {}

    # Step 3: Fill in the dictionaries
    for i in range(len(all_prompts)):
        prompt_id = int(all_prompts[i])
        model_id = int(all_models[i])
        label = int(all_labels[i])
        category = int(all_categories[i])
        
        if prompt_id not in label_dict:
            label_dict[prompt_id] = {}
        label_dict[prompt_id][model_id] = label
        
        if prompt_id not in category_dict:
            category_dict[prompt_id] = category

    # Step 4: Create the unique prompt and model lists
    unique_prompts = sorted(set(all_prompts.tolist()))
    unique_models = sorted(set(all_models.tolist()))
    model_num = len(unique_models)
    print(f"Model Num: {model_num}")

    # Step 5: Build the new dataloader content
    new_models = []
    new_prompts = []
    new_labels = []
    new_categories = []

    for prompt_id in unique_prompts:
        # Repeat the prompt ID for all models
        prompt_tensor = torch.tensor([prompt_id] * model_num)
        model_tensor = torch.tensor(unique_models)
        
        # Get labels for all models for the current prompt
        label_tensor = torch.tensor([label_dict[prompt_id][model_id] for model_id in unique_models])
        
        # Get the category for the current prompt and repeat it for all models
        category_tensor = torch.tensor([category_dict[prompt_id]] * model_num)
        
        new_prompts.append(prompt_tensor)
        new_models.append(model_tensor)
        new_labels.append(label_tensor)
        new_categories.append(category_tensor)

    # Step 6: Concatenate the new tensors
    new_prompts = torch.cat(new_prompts)
    new_models = torch.cat(new_models)
    new_labels = torch.cat(new_labels)
    new_categories = torch.cat(new_categories)

    # Step 7: Create the new DataLoader
    new_dataset = TensorDataset(new_prompts, new_models, new_labels, new_categories)
    new_dataloader = DataLoader(new_dataset, batch_size=model_num, shuffle=False)

    def compute_model_accuracy(label_dict, model_num=112):
        accuracy_dict = {}

        for model_id in range(model_num):
            correct_count = 0
            total_count = 0
            
            for prompt_id in label_dict:
                if model_id in label_dict[prompt_id]:
                    correct_count += label_dict[prompt_id][model_id]
                    total_count += 1

            if total_count > 0:
                accuracy = correct_count / total_count
            else:
                accuracy = 0.0

            accuracy_dict[model_id] = accuracy

        return accuracy_dict

    acc_dict = compute_model_accuracy(label_dict)
    return new_dataloader, label_dict, acc_dict
    
class TextMF(torch.nn.Module):
    def __init__(self, embedding_path, dim, num_models, num_prompts, text_dim=768, num_classes=2, alpha=0.05):
        super().__init__()
        self._name = "TextMF"
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        # embeddings = json.load(open(f"{pwd}/data/embeddings.json"))
        embeddings = torch.load(embedding_path)
        self.Q.weight.data.copy_(embeddings)
        self.text_proj = nn.Sequential(torch.nn.Linear(text_dim, dim))
        self.alpha = alpha

        # self.classifier = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, num_classes))
        self.classifier = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, model, prompt, category, test_mode=False):
        # print(model.shape)
        # print(self.P)
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)

        return self.classifier(p * q)

    def get_embedding(self):
        return (
            self.P.weight.detach().to("cpu").tolist(),
            self.Q.weight.detach().to("cpu").tolist(),
        )

    @torch.no_grad()
    def predict(self, model, prompt, category):
        logits = self.forward(model, prompt, category, test_mode=True)
        return torch.argmax(logits, dim=1)

def evaluator(net, test_iter, devices):
    net.eval()
    ls_fn = nn.CrossEntropyLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models, prompts, labels, categories in test_iter:
            # Assuming devices refer to potential GPU usage
            models = models.to(devices[0])  # Move data to the appropriate device
            # print(models.shape)
            # print(models)

            prompts = prompts.to(devices[0])
            labels = labels.to(devices[0])
            categories = categories.to(devices[0])

            logits = net(models, prompts, categories)
            # print(logits.shape)
            # print(logits[:50])
            # raise ValueError("STOP!")
            loss = ls_fn(logits, labels)  # Calculate the loss
            pred_labels = net.predict(models, prompts, categories)
            correct += (pred_labels == labels).sum().item()
            ls_list.append(loss.item())  # Store the sqrt of MSE (RMSE)
            num_samples += labels.shape[0]
    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples

def evaluator_router(net, test_iter, devices, acc_dict, model_num=112):
    net.eval()
    successful_num_routes = 0
    num_prompts = 0
    
    model_counts = [0] * model_num
    correctness_result = {}
    with torch.no_grad():
        for prompts, models, labels, categories in test_iter:
            prompts = prompts.to(devices[0])
            models = models.to(devices[0])
            labels = labels.to(devices[0])
            categories = categories.to(devices[0])

            logits = net(models, prompts, categories)
            logit_diff = (logits[:, 1] - logits[:, 0]).unsqueeze(1)
            max_index = torch.argmax(logit_diff)
            model_counts[max_index.item()] += 1
            successful_num_routes += int(labels[max_index] == 1)
            num_prompts += 1
            correctness_result[int(prompts[0])] = int(labels[max_index] == 1)

    # Calculate route accuracy
    route_acc = float(successful_num_routes / num_prompts)
    print(f"Route Accuracy: {route_acc}")

    # Calculate the highest accuracy baseline
    highest_accuracy = max(acc_dict.values())
    print(f"Highest Model Accuracy: {highest_accuracy}")

    # Calculate the weighted accuracy based on route_to
    weighted_acc_sum = 0
    for model_id, count in enumerate(model_counts):
        if count > 0:
            weighted_acc_sum += acc_dict[model_id] * count
    
    if sum(model_counts) > 0:
        weighted_accuracy = weighted_acc_sum / sum(model_counts)
    else:
        weighted_accuracy = 0

    print(f"Weighted Baseline Accuracy: {weighted_accuracy}")

    net.train()
    
    return "N/A", route_acc, correctness_result, model_counts

def train_recsys_rating(
    net,
    train_iter,
    test_iter,
    num_models,
    num_prompts,
    batch_size,
    num_epochs,
    loss=nn.CrossEntropyLoss(reduction="mean"),
    devices=["cuda"],
    evaluator=evaluator_router,
    **kwargs,
):
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    def train_loop():  # Inner function for one epoch of training
        net.train()  # Set the model to training mode
        train_loss_sum, n = 0.0, 0
        start_time = time.time()
        for idx, (models, prompts, labels, categorys) in enumerate(train_iter):
            # Assuming devices refer to potential GPU usage
            # print(models)
            models = models.to(devices[0])
            prompts = prompts.to(devices[0])
            labels = labels.to(devices[0])
            categorys = categorys.to(devices[0])

            output = net(models, prompts, categorys)
            ls = loss(output, labels)

            optimizer.zero_grad()  # Clear the gradients
            ls.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            train_loss_sum += ls.item() * labels.shape[0]
            n += labels.shape[0]

        return train_loss_sum / n, time.time() - start_time

    train_losses = []
    test_losses = []
    test_acces = []
    correctness_results = []
    model_counts_ls = []
    embeddings = []
    progress_bar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        train_loss, train_time = train_loop()
        train_losses.append(train_loss)
        info = {"train_loss": train_loss, "epoch": epoch}

        if evaluator:
            if evaluator == evaluator_router:
                test_ls, test_acc, correctness_result, model_counts = evaluator(net, test_iter, devices, acc_dict)
            else:
                test_ls, test_acc = evaluator(net, test_iter, devices)
            test_losses.append(test_ls)
            test_acces.append(test_acc)
            correctness_results.append(correctness_result)
            model_counts_ls.append(model_counts)
            info.update({"test_loss": test_ls, "test_acc": test_acc, "epoch": epoch})
        else:
            test_ls = None  # No evaluation

        embeddings.append(net.get_embedding()[0])

        ray.train.report({"test_acc": test_acc}, checkpoint=None)
        # wandb.log(info)

        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_ls, test_acc=test_acc)
        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_ls, test_acc=test_acc)
        progress_bar.update(1)

    progress_bar.close()
    max_index = test_acces.index(max(test_acces))
    best_correctness = correctness_results[max_index]
    best_model_counts = model_counts_ls[max_index]
    return max(test_acces), best_correctness, best_model_counts

if __name__ == "__main__":
    EMBED_DIM = 232
    ALPHA = 0.001
    TEST_MODE = True
    EMBEDDING_PATH = f"{pwd}/data_new/new_prompt_embeddings.pth"
    TRAIN_DATA_PATH = f"{pwd}/data_new/new_train_set.csv"
    # TRAIN_DATA_PATH = f"{pwd}/data_new/mf_embedding_test/for_paper/data/loo_gsm8k_train.csv"
    VAL_DATA_PATH = f"{pwd}/data_new/new_val_set.csv"
    TEST_DATA_PATH = f"{pwd}/data_new/new_test_set.csv"
    # TEST_DATA_PATH = f"{pwd}/data_new/mf_embedding_test/for_paper/data/loo_gsm8k_test.csv"
    SAVE_EMBEDDING = False
    SAVED_EMBEDDING_PATH = "data_new/mf_embedding_test/loo_truthfulqa_mathqa_embedding.pth"
    SAVE_CORRECTNESS = True
    SAVED_CORRECTNESS_PATH = "data_new/best_correctness_result.json"
    SAVED_MODEL_COUNT_PATH = "data_new/best_model_counts.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--base_model_only", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="noise level")
    args = parser.parse_args()

    print("Start Loading Dataset")
    global_train_data = pd.read_csv(TRAIN_DATA_PATH)
    if TEST_MODE:
        global_test_data = pd.read_csv(TEST_DATA_PATH)
    else:
        global_test_data = pd.read_csv(VAL_DATA_PATH)
    print("Finish Loading Dataset")

    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    subset_size = args.subset_size
    base_model_only = args.base_model_only
    alpha = args.alpha
    device = torch.device("cuda")

    (
        num_models,
        num_prompts,
        num_classes,
        num_categories,
        train_loader,
        test_loader,
        MODEL_NAMES,
    ) = split_and_load(global_train_data=global_train_data, global_test_data=global_test_data,
                       batch_size=batch_size, subset_size=subset_size, base_model_only=base_model_only,)

    # i = 0
    router_test_loader, label_dict, acc_dict = create_router_dataloader(test_loader)
    print(label_dict)
    print(acc_dict)
    # with open("data_new/label_dict.json", "w") as outfile: 
    #     json.dump(label_dict, outfile)
    # with open("data_new/acc_dict.json", "w") as outfile: 
    #     json.dump(acc_dict, outfile)
    # raise ValueError("ACC STOP!")
    # for prompts, models, labels, categories in router_test_loader:
    #     print(prompts.shape)
    #     print(prompts)
    #     print(models.shape)
    #     print(models)
    #     print(labels.shape)
    #     print(labels)
    #     i += 1
    #     if i >= 3:
    #         break

    # i = 0
    # for models, prompts, labels, categories in test_loader:
    #     model_num = 112
    #     all_models = torch.tensor(range(model_num))
    #     for prompt in list(prompts):
    #         prompt_tensor = torch.tensor([prompt]*model_num)
    #     # print(all_models)
    #     # print(prompt_tensor)
    #     print(models.shape)
    #     print(models)
    #     print(prompts.shape)
    #     print(prompts)
    #     print(labels.shape)
    #     print(labels)
    #     i += 1
    #     if i >= 3:
    #         raise ValueError("STOP!!!")

    mf = TextMF(
        embedding_path=EMBEDDING_PATH,
        dim=embedding_dim,
        num_models=num_models,
        num_prompts=35673, # TODO: fix this
        num_classes=num_classes,
        alpha=alpha,
    ).to(device)

    max_test_acc, best_correctness, best_model_counts = train_recsys_rating(
        mf,
        train_loader,
        router_test_loader, # test_loader or router_test_loader
        num_models,
        num_prompts,
        batch_size,
        num_epochs,
        devices=[device],
    )
    print(f"Embedding Dim: {embedding_dim}, Alpha: {alpha}")
    print(f"Max Test Accuracy: {max_test_acc}")
    if SAVE_CORRECTNESS:
        with open(SAVED_CORRECTNESS_PATH, "w") as outfile: 
            json.dump(best_correctness, outfile)
        with open(SAVED_MODEL_COUNT_PATH, "w") as outfile: 
            json.dump(best_model_counts, outfile)
    # print(mf.P.weight.shape)
    if SAVE_EMBEDDING:
        torch.save(mf.P.weight, SAVED_EMBEDDING_PATH)