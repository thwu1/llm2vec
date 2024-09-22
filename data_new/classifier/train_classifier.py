import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score

model_attributes = pd.read_csv("model_attributes.csv", index_col = 0)
print(model_attributes)

def add_keyword_columns(df, keywords, column_name='model_name'):
    for keyword in keywords:
        df[keyword] = df[column_name].apply(lambda x: 1 if keyword.lower() in str(x).lower() else 0)
    
    return df

keywords = ['qwen', 'vicuna', 'luminex', 'math', 'code', 'llama', 
            'physic', 'bio', 'med', '7b', '13b', '70b']
labeled_df = add_keyword_columns(model_attributes, keywords)
print(labeled_df)

feature_tensor = torch.load("optimal_mf_model_embeddings.pth").cpu().detach().numpy()
def train_classifier(feature_tensor, df, label_column):
    """
    This function trains a logistic regression classifier using the feature tensor and a label column from the input DataFrame.

    Parameters:
    - feature_tensor (np.ndarray): A 2D numpy array (num_models, feature_dimension) representing model features.
    - df (pd.DataFrame): The DataFrame containing at least the label column.
    - label_column (str): The name of the label column in the DataFrame.

    Outputs:
    - Number of 1s and 0s in the label column.
    - Train and test accuracy of the logistic regression model.
    """

    # Extract labels from the DataFrame
    labels = df[label_column].values

    # Output the number of 1s and 0s in the label column
    num_ones = np.sum(labels == 1)
    num_zeros = np.sum(labels == 0)
    print(f"Number of 1s in the label column: {num_ones}")
    print(f"Number of 0s in the label column: {num_zeros}")
    if num_ones < 10:
        return
    print(f"Ratio: {num_zeros/(num_ones+num_zeros)}")

    # Split the data, ensuring both classes are present in the test set
    X_ones = feature_tensor[labels == 1]
    X_zeros = feature_tensor[labels == 0]
    y_ones = labels[labels == 1]
    y_zeros = labels[labels == 0]

    # Split both classes into train/test sets (80%-20% split)
    X_train_ones, X_test_ones, y_train_ones, y_test_ones = train_test_split(X_ones, y_ones, test_size=0.2, random_state=42)
    X_train_zeros, X_test_zeros, y_train_zeros, y_test_zeros = train_test_split(X_zeros, y_zeros, test_size=0.2, random_state=42)

    # Combine the train/test sets from both classes
    X_train = np.vstack([X_train_ones, X_train_zeros])
    X_test = np.vstack([X_test_ones, X_test_zeros])
    y_train = np.hstack([y_train_ones, y_train_zeros])
    y_test = np.hstack([y_test_ones, y_test_zeros])

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Output the train and test accuracy
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    return model

# Example usage:
# Assuming `feature_tensor` is a numpy array of shape (num_models, feature_dimension)
# and `df` is the DataFrame with a label column 'is_specialized'.
# for keyword in keywords:
#     print(f"Attribute: {keyword}")
#     model = train_classifier(feature_tensor, labeled_df, keyword)

def train_multiple_classifiers(features, df, label_column):
    # Extract the label column
    labels = df[label_column].values

    # Output the number of 1s and 0s in the label column
    num_ones = np.sum(labels == 1)
    num_zeros = np.sum(labels == 0)
    print(f"Number of 1s: {num_ones}")
    print(f"Number of 0s: {num_zeros}")
    
    # Perform stratified train-test split to ensure balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(np.sum(y_test == 1), np.sum(y_test == 0))
    # Initialize the classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "MLP Classifier": MLPClassifier(max_iter=500),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Support Vector Machine (SVM)": SVC()
    }

    # Iterate over the classifiers and report train/test accuracy
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)  # Train the classifier
        y_train_pred = clf.predict(X_train)  # Predict on the training set
        y_test_pred = clf.predict(X_test)  # Predict on the test set

        # Calculate train and test accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"\n{name}:")
        print(f"Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print(classification_report(y_test, y_test_pred))  # Shows precision, recall, F1

train_multiple_classifiers(feature_tensor, labeled_df, "7b")