[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/I14rQV2K)
# A3 - HPC Agents with Academy

## Tasks
1. Star the Academy [Github repository](https://github.com/academy-agents/academy)
1. Create a Battleship Player Agent
1. Create a Tournament Agent
1. Deploy Agents to Midway
1. Invoke Parsl within Coordinator Agent

## Deliverables
- player.py
- simple_tournament.py
- run_tournament.py
- run_distributed.py
- parsl_tournament.py

*Warning:
Please use python3.11 for this assignment to avoid compatibility issues on Midway!!!*

Much of the challenge of this assignment will be getting familiar with tools to use HPC resources, specifically Academy, Globus Compute, and Parsl. Please use the docs extensively for development and debugging.

## Details

### Task 1: Star the Academy Github Repository
Please!

Also install the starter code:

```
git clone <your_repo>
cd <your_repo>
python -m venv venv
. ./venv/bin/activate
pip install -e .
```

### Task 2: Create a Battleship Player Agent
In the starter code for the assignment you will find a package with a Battleship game board and a simple Coordinator Agent. Your task is to implement a simple player agent in `player.py`.  The player agent defines 4 methods:
 - new_game - indicates that the Agent should start a new game, and returns to the coordinator a Board with all the Ships placed
 - get_move - returns a (x,y) coordinate of guess where an opposing ship is located
 - notify_result - called to notify the agent of the result of its own last move
 - notify_move - called to notify the agent of a guess the opponent made

The player agent should, at a minimum, store the history of their own guesses for the current game as part of its state so it does not guess the same coordinate multiple times. Implement more advanced strategies as you wish.

We provide a basic coordinator agent that pits two agents against each other.
To run the code:
```
python academy_battleship/run.py 
```
This will start the game. Use the commands `game`, `stat` and `exit` to interact with the coordinator.

### Task 3: Create a Tournament Agent
The coordinator agent in the provided stater code simple runs a series of games between two players. We will now create an agent to run an ELO style tournament between agents.
In `simple_tournament.py`, the tournament agent will implement the following methods:

- register(player: Handle[Player]) - adds a player to the tournament
- run - a control “loop” that, in a loop, randomly selects two registered players, runs a game between them, and updates their win percentage. (Alternatively, you could try to use ELO scores!)
- report - returns a dictionary of AgentIds to win percentage for the currently registered agents

You do not have to re-implement the logic to run one game, please copy and paste from the `Coordinator` starter code.

Write a script `run_tournament.py` that launches the tournament agent and several instance of your agent. We provide *incomplete* starter code for this.


### Task 4: Deploy Agents to Midway
For this assignment, you should have access to Midway, a computing cluster on campus. To access midway, you can use ssh:
```
ssh <CNetID>@midway3.rcc.uchicago.edu
```

**Deploy Globus Compute Endpoint:**

First create a virtual environment on midway
```
python -m venv venv; . ./venv/bin/activate
```
Then install Globus Compute Endpoint:

```
pip install pip install globus-compute-endpoint
```

*You do not have to change the default configuration of the endpoint.* This configuration will run agents on the login node, which, for the purposes of this assignment, is what we want. In Part 5, we will use Parsl to run tasks on the compute nodes.
Clone and install your repository in the same virtual environment.
Then start the endpoint.

**Launch Remote Agents:**

In your local environment (i.e. not on Midway), copy `run_tournament.py` to `distributed_tournament.py`. Modify the script to launch agents using the `from globus_compute_sdk import Executor as GCExecutor`, and communicate using the cloud hosted exchange at `https://exchange.academy-agents.org`.

### Task 5: Use Parsl to Parallelize Tasks
Create a copy of the tournament agent in `parsl_tournament.py`. 
Modify the run function of the tournament to execute up to n games in parallel. 
You should modify the “run_game” function to be a Parsl `@python_app` decorated function. 
The control loop should now maintain a list of `n` futures of running games, wait for a game to be completed using `asyncio.as_completed`, and launch a new game when a running one completes. 
Use `asyncio.wrap_future` to turn Parsl futures into `awaitable`s. 
We also need to configure how Parsl runs tasks. 
In the `Agent.agent_on_startup()` method, load the Parsl configuration for Midway (provided in the starter code). 
This tells Parsl how to allocate compute resources and run tasks.

Note: When testing your code, the first results of the tournament will take longer than they did in the prior version because the Agent is waiting for nodes to be allocated. On Midway, this can take up to hours. *For testing, use parsl.load() with no arguments to load a local configuration and launch the agent with n=2 to run with low parallelism on the login node*. Once it works with the local configuration, use the Midway configuration. 
