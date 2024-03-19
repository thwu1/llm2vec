from vllm import LLM, SamplingParams
from evaluation.utils import get_chat_template
from mmlu_response_generation.get_token_id import get_tokenizer_info
import argparse
import json
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--test", type=bool, default=False)
parser.add_argument("--store", type=bool, default=True)
args = parser.parse_args()

model_name = args.model
parallel = args.parallel
test = args.test
store = args.store

# NOTE: Need to change these settings for each model
sampling_params = SamplingParams(
    temperature=0.9, n=1, max_tokens=1024, prompt_logprobs=20
)
llm = LLM(
    model=model_name,
    tensor_parallel_size=parallel,
    trust_remote_code=True,
    swap_space=0,
    gpu_memory_utilization=0.85,
)
max_model_length = llm.llm_engine.model_config.max_model_len
# The desire number of shots to use for prompt (in case of prompt exceeding length limit), 5 shot for 4096, 2 shot for 2048
if max_model_length >= 4096:
    k = 5
elif max_model_length == 2048:
    k = 2
else:
    raise ValueError("LLM max model length not seen before")
print(f"Max model length: {max_model_length}, using {k}-shot prompting")
start_index = 0  # The desire starting index of the MMLU question data
end_index = 1000  # The desire end index of the MMLU question data

target_index, option_list = get_tokenizer_info(
    model_name, parallel, llm, sampling_params, k
)
# target_index = -11 # The index of the answer option (the "D" or in most cases the "_D")
# option_list = [330, 365, 334, 384] # The token IDs corresponding to the options ABCD

batch_size = 50  # If use multiple GPUs, can consider increase batch size for more optimal inference speed
output_dir = "/data/richard/llm2vec/mmlu_response_generation/outputs"
output_file = f"{output_dir}/{model_name.split('/')[-1]}_mmlu_vllm.json"


def process_question(data_path, start_index, end_index):
    """
    Methodology:

    Append a random answer to the test question and let LLM generate the log prob at that position
    Choose a large enough number of log prob to be displayed so that either ABCD in our options
    Or the model is extremely unconfident and in that case just choose the random answer we append
    which achieves a similar effect as random guessing

    """
    json_data = []
    with open(data_path, "r") as file:
        for line in file:
            try:
                json_object = json.loads(line)
                json_data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    if test:
        start_index = 0
        end_index = 50

    json_data = json_data[start_index:end_index]
    print("Number of Prompts:", len(json_data))

    def data_to_list(data):
        return [
            item["prompt"] + "D" for item in data
        ]  # Can be an arbitrary option between ABCD, here we use D

    chat_template = get_chat_template(model_name)
    ls_prompt = [chat_template(prompt) for prompt in data_to_list(json_data)]
    # print(ls_prompt[0])
    return json_data, ls_prompt


json_data, ls_prompt = process_question(
    data_path=f"mmlu_response_generation/mmlu_data/mmlu_10k_{k}_shot.json",
    start_index=start_index,
    end_index=end_index,
)


def get_text(outputs, target_index, option_list):
    ls = []
    option_map = dict(zip(option_list, "ABCD"))

    for output in outputs:
        try:
            prompt_logprobs = output.prompt_logprobs[target_index]
            # print(prompt_logprobs)
            prompt_logprobs = {
                key: prompt_logprobs[key]
                for key in option_list
                if key in prompt_logprobs
            }
            letter_option = option_map[max(prompt_logprobs, key=prompt_logprobs.get)]
            ls.append([letter_option for i in range(len(output.outputs))])
        except:  # For Debug Purpose
            # print(len(output.prompt_token_ids))
            raise ValueError("Error Detected")
    return ls


def store_result(llm_outputs, output_dir, output_file, batch_start_index, batch_size):
    """
    Store Result after for every batch
    """

    outputs_text = get_text(
        llm_outputs, target_index=target_index, option_list=option_list
    )
    for data, output in zip(
        json_data[batch_start_index : (batch_start_index + batch_size)],
        outputs_text,
    ):
        data["output"] = output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_file) and batch_start_index != 0:
        with open(output_file, "r") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.extend(
        json_data[batch_start_index : (batch_start_index + batch_size)]
    )

    with open(output_file, "w") as file:
        json.dump(existing_data, file, indent=2)


# Main Inference Loop

for i in tqdm.tqdm(range(0, len(ls_prompt), batch_size)):
    llm_outputs = llm.generate(
        prompts=ls_prompt[i : (i + batch_size)], sampling_params=sampling_params
    )
    if store:
        store_result(
            llm_outputs=llm_outputs,
            output_dir=output_dir,
            output_file=output_file,
            batch_start_index=i,
            batch_size=batch_size,
        )
    print(f"Finished Processing Question {i} to {i+batch_size}")
