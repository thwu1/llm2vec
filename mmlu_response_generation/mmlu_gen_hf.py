from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
import numpy as np
from evaluation.utils import get_chat_template
import os
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--test", type=bool, default=False)
args = parser.parse_args()

#accelerator = Accelerator()

def get_model_output(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    last_token_logits = logits[0, -1, :]
    token_ids = tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D'])
    abcd_logits = last_token_logits[token_ids]
    abcd_probs = torch.softmax(abcd_logits, dim=0)
    abcd_log_probs = torch.log(abcd_probs)

    # for token, log_prob in zip(['A', 'B', 'C', 'D'], abcd_log_probs):
    #     print(f"Log probability for option {token}: {log_prob.item()}")

    option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    best_option = option_map[np.argmax(list(abcd_log_probs))]
    return best_option

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)
#model, tokenizer = accelerator.prepare(model, tokenizer)
print("Finish Loading Model and Tokenizer")

json_data = []
with open("mmlu_response_generation/mmlu_10k.json", "r") as file:
    for line in file:
        try:
            json_object = json.loads(line)
            json_data.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

if args.test:
    json_data = json_data[:1000]

print("Num of Prompts:", len(json_data))

def data_to_list(data):
    return [item["prompt"] for item in data]

chat_template = get_chat_template(args.model)
ls_prompt = [chat_template(prompt) for prompt in data_to_list(json_data)]
print(ls_prompt[0])

results = []
for prompt in tqdm.tqdm(ls_prompt):
    #with accelerator.autocast():
    results.append(get_model_output(model, tokenizer, prompt))

for data, output in zip(json_data, results):
    data["output"] = output

if not os.path.exists("/data/richard/llm2vec/llm2vec/mmlu_response_generation/outputs"):
    os.makedirs("/data/richard/llm2vec/llm2vec/mmlu_response_generation/outputs")
json.dump(json_data, open(f"/data/richard/llm2vec/llm2vec/mmlu_response_generation/outputs/{args.model.split('/')[-1]}_mmlu_hf.json", "w"), indent=2)

