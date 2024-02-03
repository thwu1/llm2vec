from vllm import LLM, SamplingParams
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")


args = parser.parse_args()

llm = LLM(model=args.model, tensor_parrallel_size=1)

sampling_params = SamplingParams(temperature=0.7, top_k=40, top_p=0.95, num_samples=5)

data = json.load(open("data/mt-bench-first-round.json", "r"))


def data_to_list(data):
    return [item["prompt"] for item in data]
