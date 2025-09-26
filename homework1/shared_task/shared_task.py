import os
import re
import json
import time
from typing import List

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from huggingface_hub import InferenceClient
from openai import OpenAI

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_NEW_TOKENS = 2048 # TODO
ENABLE_THINKING = False  # TODO

client = InferenceClient(
    model="http://babel-6-13:9020/v1" # qwen3-4b-it
)

grammar_algorithmic = {
    "type": "json",
    "value": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "weights": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["paths", "weights"],
    },
}


# Prompt Builders
# ----------------Algorithmic----------------
def prompt_algorithmic(ex): 
    return ex["prompt"]

# ----------------MMLU-Med----------------
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s
def format_example(example, include_answer=False):
    prompt = f"Question: {example['question']}\n Options:"
    these_choices = example["choices"]
    choices = ["A", "B", "C", "D"]

    for i in range(len(these_choices)):
        prompt += f"\n{choices[i]}. {these_choices[i]}"

    prompt += "\nAnswer:"   
    if include_answer:
        # for in-context learning
        prompt += f" {choices[example['answer']]}\n\n"
    return prompt
def prompt_mmlu_med(ex):
    # https://github.com/hendrycks/test/blob/master/evaluate.py
    prompt = f"The following is a multiple choice question (with answers) about {format_subject(ex['subject'])}.  Output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    return prompt + format_example(ex, include_answer=False)

# ----------------InfoBench----------------
def prompt_infobench(ex):
    return f"Instruction: {ex['instruction']}\nQuestion: {ex['input']}\nGeneration:"


# Task-specific Metrics
# ----------------Algorithmic----------------
def score_algorithmic(pred: str, gold: dict):
    pred = json.loads(pred)
    try:
        pred_pairs = {(tuple(pred['paths'][i]), pred['weights'][i]) for i in range(len(pred['paths']))}
    except:
        print("!! Failed to parse prediction:", pred)
        return None
    
    if len(pred_pairs) == 0:
        return 0.0

    gold_pairs = gold['paths']
    gold_pairs = {(tuple(d["path"]), d["weight"]) for d in gold_pairs}
    
    overlap = pred_pairs & gold_pairs
    return len(overlap) / len(pred)

# ----------------MMLU-Med----------------
def extract_answer(text):
    # remove the latex box, common for AIME
    text = re.sub(r'\$\\boxed\{([A-Za-z])\}\$', r'\1', text)

    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        pattern = r"option \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None

def convert_llm_response_to_solution(llm_response: str) -> str:
    # adapted from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
    return extract_answer(llm_response.replace('**', ''))

def score_mmlu_med(pred: str, gold: int):
    choices = ["A", "B", "C", "D"]
    predicted_solution = convert_llm_response_to_solution(pred)
    return choices[gold] == predicted_solution

# ----------------InfoBench----------------
SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"
def bool_ratio(bool_results: List[bool]) -> float:
    "Calculate true false ratio for eval results"
    count = {"true":0, "false":0}
    for entry in bool_results:
        if entry:
            count["true"] += 1
        else:
            count["false"] += 1
        
    return count['true']/sum(count.values())

def score_infobench(predicted_solution: str, example: str) -> float:
    # https://github.com/qinyiwei/InfoBench/blob/main/evaluation.py
    message = []
    answer = ""
    input_task = example['input']
    output = predicted_solution
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    for question in example["decomposed_questions"]:
        if len(message) == 0:
            if input_task:
                content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{question}\n"
        message.append({"role": "user", "content": content})
        # create a chat completion
        success = False
        early_stop = True
        while not success:
            try:
                # default config
                temperature = 1.0
                eval_model = "gpt-5-nano-2025-08-07"

                completion = client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=temperature,
                    )
                generation = completion.choices[0].message.content
                message.append(
                        {"role": "assistant", "content": generation})
                # check if generation is yes or no
                if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                    if generation.lower().startswith("yes"):
                        answer += "Yes\n"
                    else:
                        answer += "No\n"
                else:
                    if "YES" in generation and "NO" not in generation:
                        answer += "Yes\n"
                    elif "YES" not in generation and "NO" in generation:
                        answer += "No\n"
                    else:
                        for msg in message:
                            print(msg['content'])
                        print("NO YES or NO answer!" + generation)
                        answer += "None\n"
                        early_stop = True
                        break
                success = True
            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retry!")
                time.sleep(5)

            # when no answer occurs, break the loop and continue to next instance
            if early_stop:
                break

    answer = answer[:-1]
    # save eval results as List[bool]
    bool_results = []
    for i in answer.split('\n'):
        if i == "Yes":
            bool_results.append(True)
        elif i == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)

    return bool_ratio(bool_results)

# Models to evaluate
MODELS = [
    ("Qwen3-4B", "Qwen/Qwen3-4B", True),
    ("Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507", False),
    ("Qwen3-1.7B", "Qwen/Qwen3-1.7B", True),
]

# Decoding configs
DECODE = [
    ("default", {}),
    ("greedy", {"do_sample": False}),
    ("temp_0-25", {"do_sample": True, "temperature": 0.25, "top_p": 1.0}),
    ("temp_1-5",  {"do_sample": True, "temperature": 1.5,  "top_p": 1.0}),
    ("beam3", {"do_sample": False, "num_beams": 3}),
    ("beam25", {"do_sample": False, "num_beams": 25}),
    ("typical", {"do_sample": True, "typical_p": 0.9, "top_p": 1.0, "temperature": 1.0}),
    # ("bad_case", {"do_sample": True, "temperature": 8.0, "top_p": 0.8}),
]

TASKS = [
    ("graph_dev", "dev_test", prompt_algorithmic, score_algorithmic, grammar_algorithmic),
    ("infobench", "dev_test", prompt_infobench, score_infobench, None),
    ("mmlu_med",  "dev_test", prompt_mmlu_med,  score_mmlu_med, None),
]

def main():
    rows = []

    for model_name, hf_id, thinking_mode in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, 
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        for task, split, build_prompt_func, calculate_score_func, res_format in TASKS:
            print(f"\n=== Model: {model_name} | Task: {task} ===")
            dataset = load_dataset("vashistht/11763_datasets", task, split=split)
            # dataset = dataset.select(range(5))  # debugging

            for mode_name, co in DECODE:
                metrics_sum, n = 0, 0
                for ex in dataset:
                    # Build input
                    prompt = build_prompt_func(ex)
                    messages = [{"role": "user", "content": prompt}]
                    
                    if thinking_mode:
                        if co:
                            co['chat_template_kwargs'] = {'enable_thinking': ENABLE_THINKING}
                        else:
                            co = {'chat_template_kwargs': {'enable_thinking': ENABLE_THINKING}}
                    print("extra_body config:", co)
                    response = client.chat_completion(
                        model=hf_id,
                        messages=messages,
                        max_tokens=MAX_NEW_TOKENS,
                        response_format=res_format,
                        extra_body=co if co else {},
                    )
                    txt = response.choices[0].message.content
                    print(response)
                    print(txt)
                    # if thinking_mode:
                    #     prompt = tokenizer.apply_chat_template(
                    #         messages,
                    #         tokenize=False,
                    #         add_generation_prompt=True,
                    #         enable_thinking=ENABLE_THINKING,
                    #     )
                    # else:
                    #     prompt = tokenizer.apply_chat_template(
                    #         messages,
                    #         tokenize=False,
                    #         add_generation_prompt=True,
                    #     )
                    # model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
                    # generated_ids = model.generate(
                    #     **model_inputs,
                    #     max_new_tokens=MAX_NEW_TOKENS,
                    #     **co,
                    # )
                                       
                    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
                    # # parsing thinking content
                    # try:
                    #     # rindex finding 151668 (</think>)
                    #     index = len(output_ids) - output_ids[::-1].index(151668)
                    # except ValueError:
                    #     index = 0

                    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                    # print('-'*20)
                    # print("thinking content:", thinking_content)
                    # print("content:", content)
                    # print('-'*20)
                    # txt = content
                    
                    if task == "graph_dev":
                        m = calculate_score_func(txt, ex["solution"])
                    elif task == "mmlu_med":
                        m = calculate_score_func(txt, ex["answer"])
                    elif task == "infobench":
                        m = calculate_score_func(txt, ex)

                    if m is not None:
                        metrics_sum += m
                        n += 1
                    else:
                        metrics_sum += 0
                        n += 1

                row = {"model": model_name, "hf_id": hf_id, "task": task, "split": split, "decode": mode_name}
                row['score'] = metrics_sum / max(1, n)
                rows.append(row)
                print(row)
                print('='*20)

        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        df.to_csv(f'shared_task_{task}_{model_name}_result.csv', index=False)
        print(f"\nResults saved!")
        for t in df.task.unique():
            print("\n==", t, "==")
            print(df[df.task==t].to_string(index=False))


if __name__ == "__main__":
    main()