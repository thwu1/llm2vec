import json
import os
from tqdm import tqdm
import pandas as pd


directory = "../results"
response_df = []

# print(os.listdir("../results"))

for model_name in tqdm(os.listdir("../results")):
        
    model_dir = f"{directory}/{model_name}"
    if len(os.listdir(model_dir)) != 81:
        print(f"Error in Parsing {model_name}")
        continue

    for filename in os.listdir(model_dir):
        if not filename.endswith(".json") or "results" in filename:
            continue
        else:
            # print(filename)
            response_dir = f"{model_dir}/{filename}"
            with open(response_dir, 'r') as json_file:
                responses = json.load(json_file)
                dataset_name = filename.split("samples")[-1].split("2024")[0][1:-1]
                # print(dataset_name)

            for i in range(len(responses)):
                # print(responses[i]['doc_id'])
                question_id = dataset_name + "_" + str(responses[i]['doc_id'])
                # print(question_id)
                question_text = responses[i]['arguments'][0][0]
                # print(responses[i]['arguments'][0][1])
                # print("\n")
                if 'acc' in responses[i].keys():
                    # print(f"Correctness: {responses[i]['acc']}")
                    label = int(responses[i]['acc'])
                elif 'exact_match' in responses[i].keys():
                    # print(f"Correctness: {responses[i]['exact_match']}")
                    label = int(responses[i]['exact_match'])
                else:
                    raise(Exception)
                
                response_df.append({"model_name": model_name, "question_id": question_id, 
                                    "question_text": question_text, "correctness": label})
                # break

print(response_df[0:5])
response_df = pd.DataFrame(response_df)
response_df.to_csv("all_responses.csv")