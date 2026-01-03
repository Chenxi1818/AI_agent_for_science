
import argparse
def parse_arguments():
    """Parses command-line arguments and returns them."""
    parser = argparse.ArgumentParser(description="Run a math reasoning benchmark with a ReAct Agent")
    parser.add_argument("-model", "--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="LLM to use")
    parser.add_argument("-n", "--num_questions", type=int, default=10, help="Number of questions to evaluate against")
    parser.add_argument('--seed', type=int, default=99, help="Random seed for dataset shuffling")
    parser.add_argument("-t", "--timeout", type=float, default=60.0, help="Timeout in seconds for API request")
    parser.add_argument("--dataset_path", type=str, default="openai/gsm8k", help="Dataset to use")
    return parser.parse_args()


args = parse_arguments()
args
args.model

#-------------------------------------------------
# 1. load llm (brain): understand natural language
#-------------------------------------------------

from langchain_openai import ChatOpenAI
from inference_auth_token import get_access_token

# get llm api
access_token = get_access_token()

llm = ChatOpenAI(
    model=args.model,
    api_key=access_token,
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0
)

#-----------------------------------------------------
# 2. define tools that llm can use to finish the task
#-----------------------------------------------------

from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Use this tool for mathematical calculations. It can handle addition (+),
    subtraction (-), multiplication (*), and division (/). The input should
    be a string representing a valid mathematical expression.
    Example: '100 / 2' or '5 + 15'.
    """
    try:
        # WARNING: eval() is used for simplicity. In a production environment,
        # this is a security risk. A safer parser should be used.
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: Invalid expression. Could not evaluate '{expression}'. Reason: {e}"

#-----------------------------------------------    
# 3. create ReAct agent that can use the tool
#------------------------------------------------

from langchain.agents import create_agent
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are an assistant capable of performing simple mathematical calculations by using calculator tool."
)

# extract number in the string, and turn it into value
import re
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

def extract_number(answer):
    """Parses the #### answer from the GSM8K dataset."""
    # This regex looks for #### ... and captures the number, e.g. "1,000"
    match = re.search(r"####\s*([\d,.-]+)", answer)
    if match:
        return _clean_and_convert(match.group(1))
    return None

#------------------------------------------
# 4. invoke ReAct agent to answer question
#------------------------------------------

import time

from tqdm import tqdm

import numpy as np

# import dataset from hugging face
import datasets
from datasets import load_dataset

dataset = load_dataset(args.dataset_path, "main", split="train")

# shuffle and select the subset
subset = dataset.shuffle(seed=args.seed).select(range(args.num_questions))


print(f"Loading {dataset} split {dataset.split}")
print(f"Selecting {args.num_questions} random questions (seed={args.seed})...")


# store value from evaluation harness
latencies = [] # store latency, response time
tokens = [] # store output tokens
correct_count = 0 # count the correct reasoning examples
num_evaluated = 0 # count the evaluation process

for example in tqdm(subset, total=args.num_questions):
    question = example['question']
    true_answer = example['answer']
    
    true_answer_value = extract_number(true_answer)
    
    start_time = time.time()

    result = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": f"Question: {question}\n\nAnswer: You can use tool to answer the question if needed and put your final numerical answer in the format: #### NUMBER"
        }]
        })

    end_time = time.time()

    latency = end_time - start_time
    latencies.append(latency)

    messages = result['messages'] # extract the message from the agent output

    final_response = messages[-1].content # response, return string

    usage_data = messages[-1].usage_metadata
    output_tokens = usage_data['output_tokens'] # n of output tokens
    tokens.append(output_tokens)

    response_value = extract_number(final_response)

    if response_value is not None and np.isclose(response_value, true_answer_value):
        correct_count += 1 

    num_evaluated += 1

if num_evaluated == 0:
    print('No questions were successfully evaluated.')

#------------------------------------------
# Calculate and Print Final Report 
#------------------------------------------

accuracy = (correct_count / num_evaluated) * 100

avg_latency = np.mean(latencies)
std_latency = np.std(latencies)

avg_tokens = np.mean(tokens)
std_tokens = np.std(tokens)

print("\n--- Final Evaluation Report ---")
print(f"Model: {args.model}")
print(f"Questions evaluated: {num_evaluated}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average latency: {avg_latency:.2f}s")
print(f"latency std: {std_latency:.2f}")
print(f"Average output tokens: {avg_tokens:.2f}")
print(f"output tokens std: {std_tokens:.2f}")
