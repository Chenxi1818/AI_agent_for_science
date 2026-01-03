import argparse
import time
import re
import numpy as np
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
from inference_auth_token import get_access_token

# It's critical to import the necessary classes from the openai library
from openai import OpenAI, APITimeoutError, APIConnectionError, APIError

'''parses command-line arguments
for python part1.py -model <model name> -n <num of questions>, 
'''
parser = argparse.ArgumentParser(description="Agent with ReAct Run a math reasonging benchmark with specified reasonging model/LLM") # the purpose of your command line

parser.add_argument(
    "-model", "--model",
    type=str,
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    help="LLM to use"
)
parser.add_argument(
    "-n", "--num_questions",
    type=int,
    default=10,
    help="Number of questions to evaluate against"
)
parser.add_argument('--seed', 
    type=int, 
    default=99, 
    help="Random seed for dataset shuffling (default: 99)")
parser.add_argument(
    "-t", "--timeout",
    type=int,
    default=30,
    help="Timeout in seconds"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="openai/gsm8k",
    help="dataset to use"
)
args = parser.parse_args()


def _clean_and_convert(num_str):
    """Helper to clean and convert a string to a float."""
    if num_str is None:
        return None
    try:
        # Remove commas (e.g., "1,000")
        cleaned_str = num_str.replace(',', '')
        return float(cleaned_str)
    except ValueError:
        return None

def extract_ground_truth(true_answer):
    """Parses the #### answer from the GSM8K dataset."""
    # This regex looks for #### ... and captures the number, e.g. "1,000"
    match = re.search(r"####\s*([\d,.-]+)", true_answer)
    if match:
        return _clean_and_convert(match.group(1))
    return None

def parse_model_answer(model_answer):
    """Parses the #### answer from the GSM8K dataset."""
    # This regex looks for #### ... and captures the number, e.g. "1,000"
    match = re.search(r"####\s*([\d,.-]+)", model_answer)
    if match:
        return _clean_and_convert(match.group(1))
    return None

def build_prompt(question):
    """Creates the prompt with the required output format."""
    return (
        f"Question: {question}\n\nAnswer: Put your final numerical answer in the format: #### NUMBER"
    )

def query_model(prompt, model, timeout):
    '''invoke a reasoning model'''
        # Get your access token
    access_token = get_access_token()

    client = OpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    )

    try:
        # start timer
        start_time = time.time()

        # override options just for this one call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        # stop timer
        end_time = time.time()
        latency = end_time - start_time

        # get content and token count
        response_content = response.choices[0].message.content
        # Use word count as proxy
        token_count = len(response_content.split())

        return {
            "answer_text": response_content,
            "latency": latency,
            "tokens": token_count
        }

    # --- Improved Error Handling ---
    except APITimeoutError:
        print(f"Warning: Request timed out after {timeout} seconds.", file=sys.stderr)
        return None
    except APIError as e:
        print(f"Warning: API Error (Status {e.status_code}): {e.message}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None

# invoke command-line arguments

model = args.model
n_ques = args.num_questions
seed = args.seed
timeout = args.timeout
data_path = args.dataset_path

# load 'gsm8k'dataset from huggingface
# pip install huggingface datasets
# hf auth login # login the huggingface using User Access Tokens

import datasets
from datasets import load_dataset
data_path
dataset = load_dataset(data_path, "main", split="train")

print(f"Loading {dataset} split {dataset.split}")
print(f"Selecting {n_ques} random questions (seed={seed})...")

# shuffle and select the subset
subset = dataset.shuffle(seed=seed).select(range(n_ques))

# store value from evaluation harness
latencies = [] # store latency, response time
tokens = [] # store output tokens
correct_count = 0 # count the correct reasoning examples
num_evaluated = 0 # count the evaluation process

print(f"Starting evaluation for model: {model}")
# Use tqdm for a nice progress bar
for example in tqdm(subset, total=n_ques):
    question = example['question']
    true_text = example['answer']

    true_value = extract_ground_truth(true_text)

    if true_value is None: # need to set true_value 0?? nono, if none, predict none, then right, otherwise wrong.
        print(f"Warning: Could not parse true answer for question: {question[:50]}...")
        continue
    
    # build the prompt
    prompt = build_prompt(question)

    # query the reasoning model
    response = query_model(prompt, model, timeout)

    if response is None:
        print(f"Skipping question due to API error. The question: {question[:50]}")

    # collect metrics
    latencies.append(response['latency'])
    tokens.append(response['tokens'])

    # parse model's answer
    parsed_answer = parse_model_answer(response['answer_text'])

    # compare and count
    if parsed_answer is not None and np.isclose(parsed_answer, true_value):
        correct_count += 1

    num_evaluated += 1

if num_evaluated == 0:
    print('No questions were successfully evaluated.')


# --- Calculate and Print Final Report ---
accuracy = (correct_count / num_evaluated) * 100

avg_latency = np.mean(latencies)
std_latency = np.std(latencies)

avg_tokens = np.mean(tokens)
std_tokens = np.std(tokens)

print("\n---" + "-"*30)
print(f"EVALUATION REPORT: {args.model}")
print(f"Total Questions Evaluated: {num_evaluated}")
print(f"Overall Accuracy: {accuracy:.0f}%")
print("\n--- Latency (Time) ---")
print(f"  Average:   {avg_latency:.2f} seconds")
print(f"  Std. Dev:  {std_latency:.2f} seconds")
print("\n--- Output Tokens (Words) ---")
print(f"  Average:   {avg_tokens:.2f} tokens")
print(f"  Std. Dev:  {std_tokens:.2f} tokens")
print("---" + "-"*30 + "\n")