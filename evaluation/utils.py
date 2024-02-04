def get_chat_template(model):
    if "openchat" in model.lower() or "starling" in model.lower():
        return openchat_template
    elif "llama" in model.lower() or "mistral" in model.lower():
        return llama_template
    elif "yi" in model.lower():
            return yi_template
    elif "wizardlm" in model.lower():
        return wizard_template
    elif "tulu" in model.lower():
        return tulu_template
    elif "vicuna" in model.lower():
        return vicuna_template
    else:
        print(f"Model {model} not recognized, using default template")
        return default_template

def default_template(prompt):
    return prompt

def openchat_template(prompt):
    return f"GPT4 Correct User: {prompt}<|end_of_turn|> GPT4 Correct Assistant:"

def llama_template(prompt):
    return f"[INST] {prompt} [/INST]"

def yi_template(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def wizard_template(prompt):
    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"

def tulu_template(prompt):
    return f"<|user|>\n{prompt}\n<|assistant|>\n"

def vicuna_template(prompt):
    return f"USER: {prompt} ASSISTANT:"