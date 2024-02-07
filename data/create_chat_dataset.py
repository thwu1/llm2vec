from datasets import load_dataset

dataset = load_dataset("ThWu/cleaned_prompt_r_2")

# get 20k shuffled dataset
dataset = dataset["train"].shuffle(seed=42).select(range(22000))
# dataset = dataset.map(lambda x: {"prompt": x["conversations"][0], "question_id": x["id"]})

def modify_items(example, idx):
    example["prompt"] = example["conversations"][0]
    example["question_id"] = idx
    return example

dataset = dataset.map(modify_items, with_indices=True, remove_columns=["conversations"])

# dataset.push_to_hub("ThWu/Chat_22k")