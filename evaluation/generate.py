from vllm import LLM, SamplingParams
import argparse
import json
from evaluation.utils import get_chat_template
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--test", type=bool, default=False)
args = parser.parse_args()


sampling_params = SamplingParams(temperature=0.9, n=1, max_tokens=1024)
llm = LLM(model=args.model, tensor_parallel_size=args.parallel)

json_data = []
with open("data/chat_22k.json", "r") as file:
    for line in file:
        try:
            json_object = json.loads(line)
            json_data.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
print("Num of Prompts:", len(json_data))

if args.test:
    json_data = json_data[:100]

def data_to_list(data):
    return [item["prompt"] for item in data]

chat_template = get_chat_template(args.model)
ls_prompt = [chat_template(prompt) for prompt in data_to_list(json_data)]
print(ls_prompt[0])

outputs = llm.generate(prompts=ls_prompt, sampling_params=sampling_params)

def get_text(outputs):
    ls = []
    for output in outputs:
        ls.append([output.outputs[i].text for i in range(len(output.outputs))])
    return ls

outputs_text = get_text(outputs)
# print(outputs_text)
for data, output in zip(json_data, outputs_text):
    data["output"] = output

if not os.path.exists("evaluation/outputs"):
    os.makedirs("evaluation/outputs")
json.dump(json_data, open(f"evaluation/outputs/{args.model.split('/')[-1]}.json", "w"), indent=2)
