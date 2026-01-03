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

# 10 diverse samples from GSM8K with manually written CoT reasoning (from the hugging face).
# change to simple answer
manual_samples_list = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "In April Natalia sold 48 clips.\nIn May Natalia sold 24.\n72 altogether in April and May.\n#### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "50 minutes equal to 5/6 hour.\n Weng earns $10.\n#### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "Betty has $50.\nParents give $15, grandparents give $30.\nTotal money is $95.\nNeeds $5 more.\n#### 5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "Read 24 pages today.\nTotal read is 36 pages.\n84 pages remaining.\nShe should read 42 pages tomorrow.\n#### 42"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "Writes 12 pages per week.\nWrites 624 pages per year.\n#### 624"
    },
    {
        "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
        "answer": "18 purple flowers.\n28 yellow and purple flowers total.\n7 green flowers.\nTotal of 35 flowers in the garden.\n#### 35"
    },
    {
        "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
        "answer": "32 slices from large pizzas.\n16 slices from small pizzas.\nEats 48 pieces in total.\n#### 48"
    },
    {
        "question": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
        "answer": "Weight becomes 6 pounds after brownies.\nWeight becomes 8 pounds after more jelly beans.\nFinal weight is 16 pounds.\n#### 16"
    },
    {
        "question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
        "answer": "Total spent without shoes is $143.\nTotal spent with shoes is $184.\nThe shoes cost $41.\n#### 41"
    },
    {
        "question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
        "answer": "Regular pay per day is $144.\nOvertime pay per day is $54.\nTotal daily pay is $198.\nTotal 5-day earning is $990.\n#### 990"
    }
]

'''parses command-line arguments
'''
parser = argparse.ArgumentParser(description="Run a math reasonging benchmark with specified reasonging model/LLM") # the purpose of your command line

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
parser.add_argument(
    "-s", "--samples",
    type=int,
    default=1,
    help="samples to use in the chain of thought"
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

def build_prompt(question,n_samples,manual_samples_list):
    """Creates the prompt with the required output format."""
        # extract the samples from manual list
    sample_to_use = manual_samples_list[:n_samples] 
    ''' 
    return [{
            "question": "A class has 28 students. The ratio of boys to girls is 3:4. How many girls are in the class?",
            "answer": "The ratio of boys to girls is 3:4. This means there are 3+4=7 parts in total. The total number of students is 28. To find the number of students per part, we divide 28 / 7 = 4 students per part. The number of girls is 4 parts, so there are 4 * 4 = 16 girls. #### 16"
        }]
    '''
    prompt = ""
    for sample in sample_to_use:
        # #The "\n\n" is critical to properly separate examples for the model
        prompt += f"Question: {sample['question']}\n\nAnswer: {sample['answer']}\n\n"

    prompt += f"Question: {question}\n\nAnswer: Please refer to the reasoning pattern and put your final numerical answer in the format: #### NUMBER"

    return prompt


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
n_samples = args.samples



# load 'gsm8k'dataset from huggingface
# pip install huggingface datasets
# hf auth login # login the huggingface using User Access Tokens

import datasets
from datasets import load_dataset
data_path
dataset = load_dataset(data_path, "main", split="train")

overlap_ques = {sample['question'] for sample in manual_samples_list}
eval_dataset = dataset.filter(lambda example: example['question'] not in overlap_ques)
# explain: lambda example: example['question'] not in cot_questions
# example = {'question': '...', 'answer': '...'} in dataset
# If the function returns True (meaning the question is not one of your CoT examples), it keeps that row.

print(f"Loading {dataset} split {dataset.split}")
print(f"Selecting {n_ques} random questions (seed={seed})...")

# shuffle and select the subset
subset = eval_dataset.shuffle(seed=seed).select(range(n_ques))

# store value from evaluation harness
latencies = [] # store latency, response time
tokens = [] # store output tokens
correct_count = 0 # count the correct reasoning examples
num_evaluated = 0 # count the evaluation process

print(f"Starting evaluation for model: {model} with {n_samples} CoT samples.")
# Use tqdm for a nice progress bar
for example in tqdm(subset, total=n_ques):
    question = example['question']
    true_text = example['answer']

    true_value = extract_ground_truth(true_text)

    if true_value is None: # need to set true_value 0?? nono, if none, predict none, then right, otherwise wrong.
        print(f"Warning: Could not parse true answer for question: {question[:50]}...")
        continue
    
    # build the prompt
    prompt = build_prompt(question,n_samples,manual_samples_list)

    # query the reasoning model
    response = query_model(prompt, model, timeout)

    if response is None:
        print(f"Skipping question due to API error. The question: {question[:50]}")
        continue

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
print(f"EVALUATION REPORT: {args.model} ({args.samples}-shot)")
print(f"Total Questions Evaluated: {num_evaluated}")
print(f"Overall Accuracy: {accuracy:.0f}%")
print("\n--- Latency (Time) ---")
print(f"  Average:   {avg_latency:.2f} seconds")
print(f"  Std. Dev:  {std_latency:.2f} seconds")
print("\n--- Output Tokens (Words) ---")
print(f"  Average:   {avg_tokens:.2f} tokens")
print(f"  Std. Dev:  {std_tokens:.2f} tokens")
print("---" + "-"*30 + "\n")