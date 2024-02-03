import json
import copy

data = []
with open("/home/thw/llm2vec/data/mt-bench.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

first_round = []
second_round = []

for item in data:
    new_item = copy.deepcopy(item)
    if "reference" in new_item:
        new_item.pop("reference")
    new_item["prompt"] = new_item["turns"][0]
    new_item.pop("turns")
    first_round.append(new_item)

print(first_round)

json.dump(first_round, open("/home/thw/llm2vec/data/mt-bench-first-round.json", "w"), indent=2)
