from sentence_transformers import SentenceTransformer
import json
import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="all-mpnet-base-v2")
    args = parser.parse_args()

    embedder_name = args.embedder
    embedder = SentenceTransformer(embedder_name)

    file_paths = os.listdir("evaluation/outputs")
    embed_file_paths = os.listdir(f"evaluation/outputs/{embedder_name}")
    if not os.path.exists(f"evaluation/outputs/{embedder_name}"):
        os.makedirs(f"evaluation/outputs/{embedder_name}")
    filtered_file_paths = [
        f"evaluation/outputs/{fp}" for fp in file_paths if fp.endswith(".json") and f"{fp[:-5]}_embed.json" not in embed_file_paths
    ]
    print("Files to be embedded:", filtered_file_paths)

    for file_path in tqdm(filtered_file_paths):
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

        json.dump(
            json_data,
            open(f"evaluation/outputs/{embedder_name}/" + file_path.split("/")[-1].replace(".json", f"_embed.json"), "w"),
            indent=2,
        )