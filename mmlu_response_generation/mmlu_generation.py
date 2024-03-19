import subprocess

model_list = []

todo_list = []

messages = []
file_name = "output_log.txt"
with open(file_name, "a") as file:
    file.write("\n\n")
for model in model_list:
    try:
        print(f"Processing {model}")
        command = [
            "python",
            "mmlu_response_generation/mmlu_gen_vllm.py",
            "--model",
            model,
            "--parallel",
            "4",
        ]

        subprocess.run(command, check=True)
        message = f"SUCCESSFULLY finished running script for {model}"
        messages.append(message)
        print(message)
        print(f"------------------------------------------")
    except:
        message = f"ERROR in running script for {model}"
        messages.append(message)
        print(f"------------------------------------------")
        continue
    with open(file_name, "a") as file:
        file.write(message + "\n")

for message in messages:
    print(message)
