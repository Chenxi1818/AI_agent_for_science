[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/g_FKUsS6)
# A4 — ScienceAgentBench

[ScienceAgentBench](https://arxiv.org/abs/2410.05080) is a recent attempt to create a benchmark to evaluate the capabilities of scientific assistants. We will use this benchmark to experiment with designing and evaluating multi-agent systems.

## Tasks
1. Manually complete a benchmark
1. One-shot complete a benchmark
1. Multi-Agent system construction
1. (Bonus) Human in the loop

## Deliverables
 - one_shot.py or one_shot_prompts/ and one_shot_logs/
 - multi_agent.py
 - a4_report.pdf (or a4_report.md)

## Details

### Task 1: Manually complete a benchmark
Choose a task from [ScienceAgentBench](https://huggingface.co/datasets/osunlp/ScienceAgentBench) and attempt to complete it manually. Note: these are challenging tasks. Look for a task where you have some expertise that seems reasonable with the resources and time that you have. In your report document the steps that you took, and resources you used (i.e. wikipedia, documentation, python) and any code that you write. Also keep note of how long it took to complete these tasks. Limit the time you spend on this portion to 1.5 hours. If you do not finish within that time, note how far you got, and make a plan for what you would do to complete the rest of the task.

### Task 2: One shot completion
Use a reasoning model (i.e. gpt-oss) to complete the task with a single prompt. Compare the performance if you give the agent the domain knowledge or not. Furthermore, try engineering a system prompt that helps the agent make more progress on the prompt (this could be a role, a chain of thought, some extra information about the data that you saw when you completed the task yourself, or something else). If the LLM failed to complete the task, analyze where in the answer and the reasoning trace the LLM made a mistake.

If you complete this using the API, submit your python code. If you complete this using a Chat interface, submit the prompts that you came up with. Also submit the reasoning traces from the instance that (in your opinion) go closest to completing the task.

### Task 3: Create a multi-agent system.
We have looked at serveral tools to create multi-agent systems (i.e AutoGen, LangChain, Academy). Choose one of these tools (or one we did not mention) and build a multi-agent system to solve **only the single task** that you had chosen. The criteria here is not success, but should be progress over the one-shot solution. If in Task 2, you found a Task + LLM combination that solved the task in a single shot, use a weaker model that cannot complete the task in a single shot, and show that using the same model, your multi-agent system makes more progress.

There are several ways that you could approach this --- different agents with different abilities, a critiquer loop, role playing different experts, etc. This is intentionally vaugue because their are many ways to improve over the one shot solution.

Submit your solution as `multi_agent.py`. If your multi agent solution was still not able to complete the task, analyze where the system failed. Include logs of the interaction between agents in the repo.

## (Bonus) Task 4: Human in the loop
Complete a different task in the dataset (ideally a task a peer attempted in part 1) with the assistance of *any agent or LLM you choose*. Again document the steps and the time needed to complete the task. If enough people complete this portion, we will be able to get a sense --- for this class --- if and how much LLM assistants helped humans and visa-versa how much having a human in the loop helped the LLMs.