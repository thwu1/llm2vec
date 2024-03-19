from vllm import LLM, SamplingParams
import argparse
import json
from evaluation.utils import get_chat_template
from transformers import AutoTokenizer

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, default="gpt2")
# parser.add_argument("--parallel", type=int, default=1)
# args = parser.parse_args()


def get_tokenizer_info(model_name, parallel, llm, sampling_params, k):
    """
    Extract the position of the target token (the answer option A/B/C/D)
    and the corresponding token ID for each of ABCD
    """
    num_indices = 100  # Display the last {num_indices} tokens

    json_data = []
    with open(
        f"/data/richard/llm2vec/mmlu_response_generation/mmlu_data/mmlu_10k_{k}_shot.json",
        "r",
    ) as file:
        for line in file:
            try:
                json_object = json.loads(line)
                json_data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    # print("Num of Prompts:", len(json_data))

    json_data = json_data[:1]

    def data_to_list(data):
        # return [item["prompt"] for item in data]
        return [item["prompt"] + "D" for item in data]

    chat_template = get_chat_template(model_name)
    ls_prompt = [chat_template(prompt) for prompt in data_to_list(json_data)]
    # print(ls_prompt[0])

    def get_option_index(llm, tokenizer, k):
        # Handle Special Cases due to Special Tokenizer
        # if model_name in [
        #     "Qwen/Qwen-14B-Chat",
        #     "mosaicml/mpt-30b-instruct",
        #     "mosaicml/mpt-7b-chat",
        #     "databricks/dolly-v2-12b",
        #     "tiiuae/falcon-40b-instruct",
        #     "deepseek-ai/deepseek-llm-67b-chat",
        #     "SUSTech/SUS-Chat-72B",
        #     "EleutherAI/pythia-12b",
        # ]:
        #     return "", -1

        if model_name in ["stabilityai/stablelm-tuned-alpha-7b"]:
            return "", -2

        test_output = llm.generate(
            prompts=ls_prompt[0], sampling_params=sampling_params
        )

        # print(test_output[0].prompt_logprobs[])
        def get_token(index):
            logprob = test_output[0].prompt_logprobs[index]
            token_id = next(iter(logprob))
            prompt_token = tokenizer.convert_ids_to_tokens(token_id)
            return prompt_token

        found_index = False
        for i in range(1, k + 1):
            prompt_token = get_token(-i)
            print(f"The word at {-i} index is: {prompt_token}")
            if prompt_token == "▁D":
                found_index = True
                target_index = -i
                print(f"Found target token at position {-i}")
                # Sanity Check
                left_index = -i - 3
                right_index = min((-i + 3), -1)
                neighborhood = ""
                for j in range(left_index, right_index + 1):
                    neighborhood += get_token(j)
                print(f"Sanity Check: {neighborhood}")
        if not found_index:
            # target_index = 0
            target_index = -1
            # raise ValueError("Target index not found!!!!")
            print("Warning: Target Index Not Found, Using Default -1")
        # print(test_output[0].prompt_logprobs[-5])
        return test_output, target_index

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 1: Check the tokens in the last k indices to find "_D"
    test_output, target_index = get_option_index(llm, tokenizer, k=num_indices)

    # Step 2: Get corresponding ids for A,B,C,D

    def get_letter_ids(data, tokenizer, index):
        if index == 0:
            return
        choices = ["A", "B", "C", "D"]
        four_option_data = [data[0]["prompt"] + letter for letter in choices]
        ls_prompt = [chat_template(prompt) for prompt in four_option_data]
        test_outputs = llm.generate(prompts=ls_prompt, sampling_params=sampling_params)
        option_map = {}
        option_list = []
        for letter, output in zip(choices, test_outputs):
            logprob = output.prompt_logprobs[index]
            token_id = next(iter(logprob))
            letter_token = tokenizer.convert_ids_to_tokens(token_id)
            # print(f"The word {letter} is: {token_id}")
            print(f"Sanity Check: The id {token_id} is word {letter_token}")
            option_map[token_id] = letter
            option_list.append(token_id)
        return option_list, option_map

    if target_index != 0:
        option_list, option_map = get_letter_ids(
            json_data, tokenizer, index=target_index
        )
        print(f"Target Index: {target_index}")
        print(f"Option Map: {option_map}")

    return target_index, option_list


# model_name = "project-baize/baize-v2-13b"
# parallel = 1
# sampling_params = SamplingParams(temperature=0.9, n=1, max_tokens=1024, prompt_logprobs=10)
# llm = LLM(model=model_name, tensor_parallel_size=parallel)

# get_tokenizer_info(model_name, parallel, llm)
