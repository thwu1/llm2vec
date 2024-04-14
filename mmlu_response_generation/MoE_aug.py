from datasets import load_dataset
import random
from collections import Counter
from itertools import combinations
import tqdm
import json


def extract_model_responses(dataset, model_name):
    extracted_responses = []

    for entry in dataset["train"]:
        model_answer = next(
            (answer for answer in entry["answers"] if answer["model"] == model_name),
            None,
        )

        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        if model_answer:
            response_info = {
                "exemplar_questions": entry["exemplar_questions"],
                "test_question": entry["test_questions"],
                "category": entry["subject"],
                "reference_answer": option_map[entry["reference_answers"]],
                "model_answer": model_answer["answer"],
            }
            extracted_responses.append(response_info)

    return extracted_responses


def calculate_model_accuracy(responses):
    correct_responses = sum(
        1
        for response in responses
        if response["model_answer"] == response["reference_answer"]
    )
    total_responses = len(responses)
    accuracy = correct_responses / total_responses if total_responses > 0 else 0
    return accuracy


def MoE_aggregation(all_model_responses, model_list):
    model_accuracies = {model: {} for model in model_list}

    # Iterate over each model to calculate accuracies
    for model in model_list:
        responses = all_model_responses[model]
        for response in responses:
            category = response["category"]
            if category not in model_accuracies[model]:
                category_responses = [r for r in responses if r["category"] == category]
                model_accuracies[model][category] = calculate_model_accuracy(
                    category_responses
                )

    # Determine the most accurate model per category
    best_models_per_category = {}
    for category in model_accuracies[model_list[0]].keys():
        best_model = max(model_list, key=lambda m: model_accuracies[m][category])
        best_models_per_category[category] = best_model

    # Aggregate responses using the best model for each category
    aggregated_responses = []
    for category, best_model in best_models_per_category.items():
        # print(f"Best model in answering {category} is {best_model}")
        best_model_responses = all_model_responses[best_model]
        for response in best_model_responses:
            if response["category"] == category:
                aggregated_responses.append(
                    {
                        "prompt": response["exemplar_questions"]
                        + f"\n\n"
                        + response["test_question"],
                        "output": [response["model_answer"]],
                    }
                )

    return aggregated_responses


def get_model_combos(model_list, size):
    combos = list(combinations(model_list, size))
    list_of_lists = [list(combo) for combo in combos]

    return list_of_lists


print("Start Loading Dataset...")
dataset = load_dataset("RZ412/mmlu_responses_1k")
print("Finish Loading Dataset")

print("Start Extracting All Model Responses...")
full_model_list = [entry["model"] for entry in dataset["train"][0]["answers"]]
all_model_responses = {
    model_name: extract_model_responses(dataset, model_name)
    for model_name in full_model_list
}
print("Finish Extracting All Model Responses")

combos = get_model_combos(full_model_list, 2)
print(f"Number of Combinations: {len(combos)}")

agg_model_responses = []
output_path = "/data/richard/llm2vec/mmlu_response_generation/outputs/"
for model_list in tqdm.tqdm(combos):
    combined_name = "+".join(model_list) + "_moe"
    agg_model_response = MoE_aggregation(all_model_responses, model_list)
    file_name = output_path + combined_name + "_mmlu_vllm.json"
    with open(file_name, "w") as file:
        json.dump(agg_model_response, file, indent=2)
