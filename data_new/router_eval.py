import pandas as pd
import json

def merge_prompt_data(csv_file, json_file, output_csv):
    csv_data = pd.read_csv(csv_file)
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    json_df = pd.DataFrame(list(json_data.items()), columns=['prompt_id', 'correctness'])
    json_df['prompt_id'] = json_df['prompt_id'].astype(int)
    # print(json_df)
    # print(csv_data)
    merged_df = pd.merge(csv_data, json_df, on='prompt_id', how='inner')

    return merged_df

def compute_by_benchmark_accuracy(label_dict, merged_csv):
    with open(label_dict, 'r') as f:
        label_dict = json.load(f)
    # Create a dictionary to hold benchmark accuracies
    accuracy_dict = {}

    # Ensure prompt_id in merged_csv is treated as int for matching purposes
    merged_csv['prompt_id'] = merged_csv['prompt_id'].astype(int)

    # Get the list of unique benchmark names and model IDs
    benchmarks = sorted(merged_csv['benchmark'].unique())
    model_ids = sorted(list(set([int(model_id) for result in label_dict.values() for model_id in result])))
    
    # Iterate over each benchmark
    for benchmark in benchmarks:
        # Filter merged_csv by benchmark
        benchmark_data = merged_csv[merged_csv['benchmark'] == benchmark]
        
        # Initialize a dictionary to store accuracy for each model on this benchmark
        benchmark_accuracy = {}
        
        # Iterate over each model
        for model_id in model_ids:
            # Collect all correctness results for the current model and benchmark
            correct_results = []
            
            # Iterate over all prompt_ids for this benchmark
            for prompt_id in list(benchmark_data['prompt_id'].unique()):
                # If the model has a correctness result for this prompt_id, add it to the list
                if str(prompt_id) in label_dict and str(model_id) in label_dict[str(prompt_id)]:
                    correct_results.append(label_dict[str(prompt_id)][str(model_id)])
            
            # Calculate the average accuracy for the model on this benchmark
            if correct_results:
                average_accuracy = sum(correct_results) / len(correct_results)
            else:
                average_accuracy = 0.0  # If no results, assume 0 accuracy
            
            # Store the result in the benchmark_accuracy dictionary
            benchmark_accuracy[model_id] = average_accuracy
        
        # Store the model accuracy for this benchmark in the final dictionary
        accuracy_dict[benchmark] = benchmark_accuracy
    
    return accuracy_dict

csv_file = 'prompt_id_to_benchmark.csv'  # CSV with columns: prompt_id, benchmark_name
json_file = 'best_correctness_result.json'     # JSON with structure: { "prompt_id": correctness_result }
output_csv = 'correctness_by_benchmark.csv'          # Output CSV with columns: prompt_id, benchmark_name, correctness
label_dict = 'label_dict.json'
model_counts = "best_model_counts.json"
with open(model_counts, 'r') as f:
    model_counts = json.load(f)

correctness_by_benchmark = merge_prompt_data(csv_file, json_file, output_csv)
print(correctness_by_benchmark)
router_benchmark_accuracy = correctness_by_benchmark.groupby('benchmark')['correctness'].mean().sort_index()
print(router_benchmark_accuracy)
acc_dict_by_benchmark = compute_by_benchmark_accuracy(label_dict, correctness_by_benchmark)
# print(acc_dict_by_benchmark)
for benchmark, model_accuracies in acc_dict_by_benchmark.items():
    router_accuracy = router_benchmark_accuracy[benchmark]
    highest_accuracy = max(model_accuracies.values())
    weighted_accuracy = sum([a*b for a,b in zip(model_accuracies.values(), model_counts)])/sum(model_counts)
    print(f"Benchmark: {benchmark} | Router Accuracy: {router_accuracy:.4f} | Single Best Accuracy: {highest_accuracy:.4f} | Weighted Accuracy: {weighted_accuracy:.4f}")
print(model_counts)