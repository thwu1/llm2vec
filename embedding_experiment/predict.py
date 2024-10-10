import pandas as pd
import numpy as np
import torch, random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os,json
from scipy.stats import kendalltau


# Function to extract the base name
def get_base_name(column_name):
    return column_name.split('_')[0]

def correlation_significance(data, test_benchmarks, show_regression=False, test_benchmark_name=None,
                             kt_test=False):
    # Convert the NumPy array to a pandas DataFrame
    embeddings = pd.DataFrame(data.detach().cpu().numpy())

    embedding_vectors = {}

    # Iterate through each row in the order to populate the dictionary
    for index, row in order.iterrows():
        model_name = row['model_name']
        model_id = row['model_id']
        embedding = embeddings.iloc[model_id]
        embedding_vectors[model_name] = embedding

    # Extract embeddings and corresponding output vectors
    embeddings = np.array([embedding_vectors[name] for name in model_names])

    # Extract all columns except 'model_name' to construct the outputs matrix
    outputs = result_df[test_benchmarks].to_numpy()

    significance = 0
    mse_scores = []  # To store MSE scores for each iteration
    for j in range(100):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test, model_names_train, model_names_test = train_test_split(
            embeddings, outputs, model_names, test_size=0.5, random_state=j
        )

        # Initialize and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)
        # print(y_pred, y_test)
        
        # TODO: Plot y_pred against y_test
        if show_regression:
            # Scatter plot of y_pred vs y_test
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, color='blue', label='Predicted Accuracy')

            # Plot the y=x line
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='y = x')

            # Add labels and title
            plt.xlabel('Actual Accuracy', fontsize=20)
            plt.ylabel('Predicted Accuracy', fontsize=20)
            plt.title(f'Predicted vs. True Model Accuracy on {test_benchmark_name.upper()}', fontsize=20)
            plt.legend(fontsize=16)
            plt.grid(True)

            # Show the plot
            plt.show()
            plt.savefig(f"plots/regression_examples/regression_example_{test_benchmark_name}.png")
            break
        
        if kt_test:
            # Apply Kendall tau test to each pair of corresponding columns
            for i, column_name in enumerate(test_benchmarks):
                tau, p_value = kendalltau(y_test[:, i], y_pred[:, i])
                # print(f"Kendall's tau for {column_name}: {tau}")
                # print(f"P-value for {column_name}: {p_value}")

                # Interpretation
                if p_value < 0.05:  # Assuming a 5% significance level
                    significance += 1
                    # print(f"There is a significant trend between y_test and y_pred for {column_name}.")
                else:
                    significance += 0
                    # print(f"There is no significant trend between y_test and y_pred for {column_name}.")

        # Normalize y_pred and y_test using y_test's distribution statistics
        y_test_mean = np.mean(y_test, axis=0)
        y_test_std = np.std(y_test, axis=0)

        y_test_normalized = (y_test - y_test_mean) / y_test_std
        y_pred_normalized = (y_pred - y_test_mean) / y_test_std
        
        # Calculate MSE
        mse = mean_squared_error(y_test_normalized, y_pred_normalized)
        mse_scores.append(mse)
        # Print or log the MSE for this iteration
        # print(f"Iteration {j+1}, MSE: {mse:.4f}")

    # Calculate the average MSE over all iterations
    avg_mse = np.mean(mse_scores)
    # avg_mse = np.sum(mse_scores)

    # print(f"Average MSE over 100 iterations: {avg_mse:.5f}")
    if kt_test:
        return significance
    else:
        return avg_mse

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Read in model performance and order
    df = pd.read_csv('correctness_matrix.csv')
    order = pd.read_csv('model_order.csv')

    benchmarks = ['gpqa', 'mmlu', 'medmcqa', 'piqa', 'asdiv',
                        'truthfulqa', 'logiqa', 'gsm8k', 'mathqa', 'social']
    significance = {}

    # Apply the function to get base names and group
    base_names = df.columns.to_series().apply(get_base_name)
    grouped_columns = df.columns.to_series().groupby(base_names).apply(list)

    # Create an empty DataFrame for the results
    result_df = pd.DataFrame()
    result_df['model_name'] = df['model_name']

    model_names = result_df['model_name'].tolist()

    # Summarize the grouped columns (e.g., taking the mean)
    for base_name, columns in grouped_columns.items():
        if 'model_name' in columns:
            continue  # Skip the 'model_name' column
        # Compute the sum and average of the grouped columns
        result_df[base_name] = df[columns].sum(axis=1) / len(columns)

        # # Plotting the distribution
        # plt.hist(result_df[base_name], bins=50, alpha=0.7, edgecolor='black')
        # plt.title(f'{base_name}')
        # plt.xlabel('Model Accuracys')
        # plt.ylabel('Frequency')
        # plt.show()

    kt_test_results = {}
    for test_benchmark in benchmarks:
        if test_benchmark == "mathqa":
            not_removed_data = torch.load(os.path.join('embeddings', f"{test_benchmark}_embedding.pth"))
            correlation_significance(not_removed_data, [test_benchmark], show_regression=True, test_benchmark_name=test_benchmark)
            kt_test_results[test_benchmark] = correlation_significance(not_removed_data, [test_benchmark], kt_test=True)
    
    # for test_benchmark in benchmarks:
    #     for omitted_benchmark in benchmarks:
            
    #         if test_benchmark == omitted_benchmark:
    #             significance[str((test_benchmark, omitted_benchmark))] = 0
    #         else:
    #             not_removed_data = torch.load(os.path.join('embeddings', f"{test_benchmark}_embedding.pth"))
    #             try:
    #                 removed_data = torch.load(os.path.join('embeddings', f"{test_benchmark}_{omitted_benchmark}_embedding.pth"))
    #             except:
    #                 removed_data = torch.load(os.path.join('embeddings', f"{omitted_benchmark}_{test_benchmark}_embedding.pth"))
    #             significance[
    #                 str((test_benchmark, omitted_benchmark))
    #                 ] = correlation_significance(
    #                     removed_data, [test_benchmark]) - correlation_significance(not_removed_data, [test_benchmark])
    #         print(f"MSE Difference of removing {omitted_benchmark} on predicting {test_benchmark}: {significance[str((test_benchmark, omitted_benchmark))]:.6f}")
        

    # print(significance)
    # with open("mse_differences_result.json", 'w') as json_file:
    #     json.dump(significance, json_file, indent=2)
    # print(kt_test_results)
    # with open("kt_test_results.json", 'w') as json_file:
    #     json.dump(kt_test_results, json_file, indent=2)

