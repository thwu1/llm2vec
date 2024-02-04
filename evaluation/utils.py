def get_chat_template(model):
    if "openchat" in model.lower() or "starling" in model.lower():
        return openchat_template
    elif "llama" in model.lower() or "mistral" in model.lower():
        return llama_template
    else:
        print(f"Model {model} not recognized, using default template")
        return default_template

def default_template(prompt):
    return prompt

def openchat_template(prompt):
    return f"GPT4 Correct User: {prompt}<|end_of_turn|> GPT4 Correct Assistant:"

def llama_template(prompt):
    return f"[INST] {prompt} [/INST]"