import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Load the data
print("Start Loading Data")
data = pd.read_csv("all_responses.csv", index_col=0)
print("Finish Loading Data")
# Create a unique mapping of prompt IDs
prompt_id_mapping = {prompt: idx for idx, prompt in enumerate(data['question_text'].unique())}
data['prompt_id'] = data['question_text'].map(prompt_id_mapping)

# Ensure prompts are sorted by their prompt IDs
sorted_prompts = [prompt for prompt, idx in sorted(prompt_id_mapping.items(), key=lambda x: x[1])]

embedder_names = ["jxm/cde-small-v1", "nomic-ai/nomic-embed-text-v1-ablated", "sentence-transformers/all-mpnet-base-v2",
                  "sentence-transformers/msmarco-bert-co-condensor", "sentence-transformers/LaBSE"]
for embedder_name in embedder_names:
    # Initialize the SentenceTransformer model
    # embedder_name = 'all-mpnet-base-v2'
    print(f"Start Generating Embedding for {embedder_name}")
    model = SentenceTransformer(embedder_name, trust_remote_code=True)
    # continue
    # Compute embeddings for prompts in sorted order
    prompt_embeddings = model.encode(sorted_prompts, convert_to_tensor=True, show_progress_bar=True)

    # Convert embeddings to a PyTorch tensor
    prompt_embeddings_tensor = prompt_embeddings.clone().detach()
    print(prompt_embeddings_tensor.shape)

    # Save the embeddings to a .pth file
    output_path = f"new_prompt_embeddings_{embedder_name.split('/')[-1].replace('-', '_')}.pth"
    torch.save(prompt_embeddings_tensor, output_path)

    print(f"Prompt embeddings saved to {output_path}")