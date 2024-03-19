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


def vote_aggregation(all_model_responses, model_list):
    question_answers = {}

    for model in model_list:
        responses = all_model_responses[model]
        for response in responses:
            question = (
                response["exemplar_questions"] + f"\n\n" + response["test_question"]
            )
            answer = response["model_answer"]
            if question not in question_answers:
                question_answers[question] = []
            question_answers[question].append(answer)

    aggregated_responses = []
    for question, answers in question_answers.items():
        answer_counts = Counter(answers)
        top_answers = answer_counts.most_common()

        if len(top_answers) > 1 and top_answers[0][1] == top_answers[1][1]:
            majority_answer = random.choice(
                [answer[0] for answer in top_answers if answer[1] == top_answers[0][1]]
            )
        else:
            majority_answer = top_answers[0][0]

        aggregated_responses.append({"prompt": question, "output": [majority_answer]})

    responses = extract_model_responses(dataset, model_list[0])

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
    combined_name = "+".join(model_list) + "_vote"
    agg_model_response = vote_aggregation(all_model_responses, model_list)
    file_name = output_path + combined_name + "_mmlu_vllm.json"
    with open(file_name, "w") as file:
        json.dump(agg_model_response, file, indent=2)
