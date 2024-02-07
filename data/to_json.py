from datasets import load_dataset

dataset = load_dataset("ThWu/Chat_22k", split="train")

dataset.to_json("data/chat_22k.json")