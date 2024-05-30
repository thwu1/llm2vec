from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import torch
import json
import os
from tqdm import tqdm
import random
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

pwd = os.getcwd()
print(pwd)

def load_data(pwd):
    train_x = torch.load(f'{pwd}/data_new/new_train_x.pth')
    train_y = torch.load(f'{pwd}/data_new/new_train_y.pth')
    val_x = torch.load(f'{pwd}/data_new/new_val_x.pth')
    val_y = torch.load(f'{pwd}/data_new/new_val_y.pth')
    test_x = torch.load(f'{pwd}/data_new/new_test_x.pth')
    test_y = torch.load(f'{pwd}/data_new/new_test_y.pth')

    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, val_x, val_y, test_x, test_y

def evaluate(train_x, train_y, test_x, test_y, num_neighbors):
    accu = []
    num_model, q_embed_dim, num_train_questions = train_x.shape
    NUM_NEIGHBORS = num_neighbors
    for i in tqdm(range(num_model)):
        X_train, y_train = train_x[i, :, :].tolist(), train_y[i, :].tolist()
        X_test, y_test = test_x[i, :, :].tolist(), test_y[i, :].tolist()
        neigh = KNeighborsClassifier(n_neighbors=NUM_NEIGHBORS)
        neigh.fit(X_train, y_train)
        pred_y = neigh.predict(X_test).tolist()
        bool_ls = list(map(lambda x, y: int(x == y), pred_y, y_test))
        accu.append(sum(bool_ls) / len(y_test))
        # if i % 100 == 0:
        #     print(f"Mean Accuracy for {i} models:", sum(accu) / len(accu))

    # for acc, model in zip(accu, model_order):
    #     print(f"Model: {model}, Accuracy: {acc}")
        
    print(f"Mean Test Accuracy for {num_neighbors} neighbors:", sum(accu) / len(accu))
    return {"mean_accuracy": sum(accu) / len(accu)}

if __name__ == "__main__":
    print(f"Start Initializing Dataset...")
    # train_x, train_y, val_x, val_y, test_x, test_y = save_data(path="../data_new/all_responses.pth")
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(pwd=os.getcwd())
    print(f"Finish Initializing Dataset")
    
    NUM_NEIGHBORS = 132
    TEST_MODE = True
    
    if TEST_MODE:
        evaluate(train_x, train_y, test_x, test_y, NUM_NEIGHBORS)
    else:
        evaluate(train_x, train_y, val_x, val_y, NUM_NEIGHBORS)
    # neighbor_sizes = range(5,100,5)

    # for num_neighbors in neighbor_sizes:
    #     evaluate(train_x, train_y, val_x, val_y, num_neighbors)
