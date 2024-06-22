import subprocess

model_infos = [
["WizardLM/WizardCoder-Python-34B-V1.0", 2, 2],
["TheBloke/WizardLM-13B-V1.2-GGUF", 2, 1],
["AiMavenAi/Prometheus-1.3", 4, 1],
["mistralai/Mixtral-8x7B-Instruct-v0.1", 4, 1],
["Qwen/Qwen-72B", 4, 1],
["SUSTech/SUS-Chat-72B", 4, 1],
["FelixChao/vicuna-7B-physics", 1, 2],
["AdaptLLM/medicine-LLM-13B", 2, 1],
["Qwen/Qwen1.5-0.5B-Chat", 1, 2],
["Qwen/Qwen1.5-7B-Chat", 1, 2],
["bigcode/octocoder", 2, 1],
["meta-llama/Meta-Llama-3-8B", 1, 2],
["meta-llama/Meta-Llama-3-8B-Instruct", 1, 2],
["meta-llama/Meta-Llama-Guard-2-8B", 1, 2],
["Qwen/Qwen1.5-4B-Chat", 1, 2],
["google/gemma-7b-it", 1, 2],
["h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b", 2, 1]
]

datasets = "mmlu,truthfulqa_mc1,social_iqa,piqa,medmcqa,mathqa,logiqa,gsm8k,gpqa,asdiv"
devices = "4,5,6,7"
messages = []
print(f"Evaluating {len(model_infos)} Models for this Run")

# Iterate over model names and run the command for each
for model_info in model_infos:

    model_name = model_info[0]
    tps = model_info[1]
    dps = model_info[2]
    print(f"START EVALUATING {model_name}...")
    print("\n \n \n")

    base_command = [
        "lm_eval", 
        "--model", "vllm",
        "--model_args", f"pretrained={model_name},tensor_parallel_size={tps},data_parallel_size={dps},dtype=auto,gpu_memory_utilization=0.8,max_model_len=2048,enforce_eager=True,trust_remote_code=True",
        "--tasks", datasets,
        "--batch_size", "auto:20",
        "--output_path", "results",
        "--log_samples",
        "--cache_requests", "true",
    ]
    full_command = f"export CUDA_VISIBLE_DEVICES={devices} && export NCCL_P2P_DISABLE=1 && " + " ".join(base_command)
    # print(full_command)

    try:
        # Execute the command in a shell environment
        result = subprocess.run(full_command, shell=True, check=True)
        message = f"Output for model {model_name} successfully recorded"
    except subprocess.CalledProcessError as e:
        message = f"Error for model {model_name}"
    
    print(message)
    print("\n \n \n")
    messages.append(message)

print("\n \n \n")
for message in messages:
    print(message)
