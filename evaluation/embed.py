from sentence_transformers import SentenceTransformer
import json
import os

embedder_name = "all-mpnet-base-v2"
embedder = SentenceTransformer(embedder_name)

file_paths = os.listdir("evaluation/outputs")
filtered_file_paths = [f"evaluation/outputs/{fp}" for fp in file_paths if "embed" not in fp and f"{fp[:-5]}_{embedder_name}_embed.json" not in file_paths]
print(filtered_file_paths)

for file_path in filtered_file_paths:
    json_data = json.load(open(file_path, "r"))

    def get_sentences(data):
        ls = []
        for item in data:
            for output in item["output"]:
                ls.append(item["prompt"] + output)
        return ls

    sentences = get_sentences(json_data)
    sentence_embeddings = embedder.encode(sentences)

    idx = 0
    for item in json_data:
        item["embedding"] = sentence_embeddings[idx : idx + len(item["output"])].tolist()
        idx += len(item["output"])

    assert idx == len(sentence_embeddings)

    json.dump(json_data, open(file_path.replace(".json", f"_{embedder_name}_embed.json"), "w"), indent=2)
