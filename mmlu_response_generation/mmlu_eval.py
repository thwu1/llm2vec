import json
import re
import datasets
import argparse

mmlu_dataset = datasets.load_dataset('hails/mmlu_no_train', 'all')
random_10k_sample = mmlu_dataset['test'].shuffle(seed=42).select(range(10000))
true_answers = [chr(65 + entry['answer']) for entry in random_10k_sample]

file_path = "/data/richard/llm2vec/llm2vec/mmlu_response_generation/outputs/Llama-2-7b-chat-hf_mmlu_vllm.json"
with open(file_path, 'r') as file:
    mmlu_log_prob = json.load(file)

letter_responses = [entry['output'][0] for entry in mmlu_log_prob]

acc = sum([true_answers[i] == letter_responses[i] for i in range(len(letter_responses))])/len(letter_responses)
print(f"MMLU Accuracy: {acc} for {len(letter_responses)} questions")