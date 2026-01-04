#%%

from langchain.tools import tool
import pickle, json
from langchain_openai import ChatOpenAI
from inference_auth_token import get_access_token
from langchain.agents import create_agent

import biopsykit as bp


#%% 
# ---------- simple prompt without domain_knowledge----------

# --- define tool
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


# --- create agent 

from langchain_openai import ChatOpenAI
from inference_auth_token import get_access_token

# get llm api
access_token = get_access_token()

llm = ChatOpenAI(
    model='openai/gpt-oss-20b',
    api_key=access_token,
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0
)    

USER_PROMPT = """
You should use tools when needed. I want you to:
1. Use tool `load_data('sleep_imu_data/sleep_data.pkl')` to read the dataset.
2. Based on data, compute the following sleep endpoints and store them in a dictionary format: 
- time of falling asleep("sleep_onset")
- time of awakening("wake_onset")
- total duration spent sleeping ("total_sleep_duration")
3. In the end, use tool `store_results('pred_results/imu_pred.json', results_dict)` to store the results in a JSON file.
"""

from langchain.agents import create_agent
agent = create_agent(
    model=llm,
    tools=[load_data,store_results]
)    

# invoke agent
response = agent.invoke(
    {"messages":[{"role": "user",
                  "content": USER_PROMPT}]}
)        

response['messages'][-1].content
# %%
# ---------- simple prompt with domain_knowledge----------

SYSTEM_PROMPT = """
You are a Data Analyst Agent. You can use tools if needed.

Domain knowledge:
- You can use the function sleep_processing_pipeline.predict_pipeline_acceleration() in biopsykit to compute the sleep endpoints. biopsykit is a Python package for the analysis of biopsychological data.
"""

@tool
def load_data(file_path: str) -> str:
    """Load IMU sleep data."""
        
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
        print("Load data...")
        return data.to_json
 
@tool
def store_results(file_path: str, results: dict) -> str:
    """Store computed results to JSON."""
    print("Store results...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return f"Results stored at {file_path}"

agent = create_agent(
    model=llm,
    tools=[load_data,store_results],
    system_prompt=SYSTEM_PROMPT
)    

# same user prompt as before
USER_PROMPT = """
You should use tools when needed. I want you to:
1. Use tool `load_data('sleep_imu_data/sleep_data.pkl')` to read the dataset.
2. Based on data, compute the following sleep endpoints and store them in a dictionary format: 
- time of falling asleep("sleep_onset")
- time of awakening("wake_onset")
- total duration spent sleeping ("total_sleep_duration")
3. In the end, use tool `store_results('pred_results/imu_pred.json', results_dict)` to store the results in a JSON file.
"""

response = agent.invoke(
    {"messages":[{"role": "user",
                  "content": USER_PROMPT}]}
)        

response['messages'][-1].content
#%%
# ---------- add tips and chain of though -------

SYSTEM_PROMPT = """
You are a Data Analyst Agent. You can use tools if needed.

Domain knowledge:
- You can use the function sleep_processing_pipeline.predict_pipeline_acceleration() in biopsykit to compute the sleep endpoints. biopsykit is a Python package for the analysis of biopsychological data.

Some tips:
- The units of the index in the data are not in microseconds, so you need to convert them into microseconds. Useful command is `data.index = data.index.as_unit('us')`
- The sampling rate is 204.8
- The source of biopsykit package is: https://github.com/mad-lab-fau/BioPsyKit
"""

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

agent = create_agent(
    model=llm,
    tools=[load_data,store_results],
    system_prompt=SYSTEM_PROMPT
)    

USER_PROMPT = """
You should use tools when needed. I want you to:
1. Use tool `load_data('sleep_imu_data/sleep_data.pkl')` to read the dataset.
2. Based on data, compute the following sleep endpoints and store them in a dictionary format: 
- time of falling asleep("sleep_onset")
- time of awakening("wake_onset")
- total duration spent sleeping ("total_sleep_duration")
3. In the end, use tool `store_results('pred_results/imu_pred.json', results_dict)` to store the results in a JSON file.

Please think step by step. First use tool `load_data(...)` to load data. Then convert the unit of row index into microseconds. Then perform detection using function sleep_processing_pipeline.predict_pipeline_acceleration() in biopsykit Python package. Store the results in a JSON file by using tool `store_results(...)`. 
"""

response = agent.invoke(
    {"messages":[{"role": "user",
                  "content": USER_PROMPT}]}
)        

response['messages'][-1].content
