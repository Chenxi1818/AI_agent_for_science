#%% Imports
from langchain.tools import tool
import pickle, json
from langchain_openai import ChatOpenAI
from inference_auth_token import get_access_token
from langchain.agents import create_agent

import biopsykit as bp

import pandas as pd

from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

#%% ------------------- Define Tools -------------------

@tool("load_data", description="Loads the IMU sleep data. Use this for loading and previewing the data.")
def load_data(file_path: str) -> str:
    """Load IMU sleep data."""
        
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
        print("Load data...")
               
        return data.to_json


@tool("store_results", description="Store the prediction results. Use this to store the results in a JSON file")
def store_results(file_path: str, results: str) -> str:
    """Store computed results to a JSON file."""
    print("Store results...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return f"Results stored at {file_path}"


#%% ------------------- Python Execution Tool -------------------
python_repl = PythonREPL()
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use `python_repl(python code)` to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

#%% ------------------- Subagents -------------------

access_token = get_access_token()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=access_token,
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# -- Generate Agent: generates code --
GenerateCodeAgent = create_agent(
    model=llm,
    system_prompt="You are a code generation agent specialized in generating code based on Github repositories.",
)

@tool("GenerateCodeAgent", description="Generate sleep detection code.")
def call_GenerateCodeAgent(query: str) -> str:
    result = GenerateCodeAgent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# -- Detection Agent: executes verified code --
DetectSleepAgent = create_agent(
    model=llm,
    tools=[repl_tool],
    system_prompt="You are a code execution agent specialized in running validated Python code for sleep data detection. You should use tool `repl_tool` to run the validated Python code and return the results.",
)

@tool("DetectSleepAgent", description="Execute verified Python sleep detection code.")
def call_DetectSleepAgent(query: str) -> str:
    result = DetectSleepAgent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# -- Check Code Agent: validates generated code --
CheckCodeAgent = create_agent(
    model=llm,
    system_prompt="You are a code review agent. Validate Python code correctness for sleep detection based on the BioPsyKit package repository.",
)

@tool("CheckCodeAgent", description="Validate generated Python code correctness.")
def call_CheckCodeAgent(query: str) -> str:
    result = CheckCodeAgent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

#%% ------------------- Main Data Analyst Agent -------------------

SYSTEM_PROMPT = """
You are a Data Analyst Agent. You can use tools and subagents if needed.

Domain knowledge:
- You can use the function sleep_processing_pipeline.predict_pipeline_acceleration() in biopsykit to compute the sleep endpoints. biopsykit is a Python package for the analysis of biopsychological data.

Some tips:
- The units of the index in the data are not in microseconds, so you need to convert them into microseconds. Useful command is `data.index = data.index.as_unit('us')`
- The sampling rate is 204.8
- The source of biopsykit package is: https://github.com/mad-lab-fau/BioPsyKit
"""

main_agent = create_agent(
    model=llm,
    tools=[load_data, call_GenerateCodeAgent, call_CheckCodeAgent, call_DetectSleepAgent, store_results],
    system_prompt=SYSTEM_PROMPT,
)

USER_PROMPT = """
You should use tools when needed. I want you to:
1. Use tool `load_data('sleep_imu_data/sleep_data.pkl')` to read the dataset.
2. Based on data, compute the following sleep endpoints and store them in a dictionary format: 
- time of falling asleep("sleep_onset")
- time of awakening("wake_onset")
- total duration spent sleeping("total_sleep_duration")
3. When you write the Python code, you should call `CheckCodeAgent` to check if the generated code is right.
4. If the generated code is right, you should call `DetectSleepAgent` to run the code. You should be aware of the unit of total_sleep_duration should be minutes. The timestamps should be in a human-readable format
5. In the end, use tool `store_results('pred_results/imu_pred.json', results_dict)` to store the results in a JSON file.


Please think step by step. First use tool `load_data(...)` to load data. Then convert the unit of row index into microseconds. Then call subagents to help you finish the task. Finally, store the results in a JSON file by using tool `store_results(...)`. 
"""

if __name__ == "__main__":
    print("Invoking main agent...")
    response = main_agent.invoke({"messages": [{"role": "user", "content": USER_PROMPT}]})
    final_output = response["messages"][-1].content
    print("\n=== Final Output ===\n")
    print(final_output)

# %%
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

