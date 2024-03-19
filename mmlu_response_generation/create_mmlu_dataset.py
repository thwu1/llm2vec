import datasets
import numpy as np

def format_subject(subject):
    """
    Taken from https://github.com/hendrycks/test/blob/master/evaluate.py
    """
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_mcq(entry, k = 5, include_answer=False):
    """
    Formats a dictionary entry into a multiple-choice question format.
    
    Parameters:
    - entry (dict): A dictionary containing the keys 'question', 'subject', 'choices', and 'answer'.
    - include_answer (bool, optional): Whether to include the correct answer in the output. Defaults to False.
    
    Returns:
    - str: A formatted string representing the multiple-choice question.
    """
    
    header = "The following are multiple choice questions (with answers) about{}.\n".format(format_subject(entry['subject']))
    
    question = entry['question']
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(entry['choices'])])
    
    formatted_mcq = f"{header}\n{question}\n\n{choices}"
    
    # Include the answer if required
    if include_answer:
        # Map the numerical answer to a letter
        answer_letter = chr(65 + entry['answer'])
        formatted_mcq += f"\n\nAnswer: {answer_letter}"
    else:
        formatted_mcq += f"\n\nAnswer: "
    
    return formatted_mcq

# Load Dataset and Randomly Select 10000 Samples
mmlu_dataset = datasets.load_dataset('hails/mmlu_no_train', 'all')
random_10k_sample = mmlu_dataset['test'].shuffle(seed=42).select(range(10000))

# Process Dataset
formatted_entries = [format_mcq(entry) for entry in random_10k_sample]
prompt_dataset = datasets.Dataset.from_dict({'prompt': formatted_entries})

# Save as Json Data
prompt_dataset.to_json('llm2vec/mmlu_response_generaton/mmlu_10k.json')