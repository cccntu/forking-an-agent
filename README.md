# Forking an AI Agent

This is a MVP implementation of the fork pattern for AI agents.

## Overview

This project demonstrates a pattern for AI agents: the ability to fork themselves into multiple instances to handle parallel tasks efficiently.
Inspired by Unix's `fork()` system call, this pattern enables AI agents to break down complex tasks into parallel subtasks while maintaining clean conversation context.

## Fork vs Spawn

AI agents like [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) can perform large number of tasks in a single session.
However, these tool calls can quickly fill up the context window, incresing cost and limiting the number of tasks that can be performed in a single session.

Claude Code support a `dispatch_agent` tool, which can be used to dispatch a task to an agent.
This can greatly increase the number of tasks that can be performed in a single session.
However, it's like spawning a new process in Unix, it doesn't have the full conversation history of the main agent. This greatly limites the usefulness of the `dispatch_agent` tool.


## Benefits of Forking

Like `dispatch_agent`, forking can help avoid filling up the context window. Additionally, it provides a number of benefits:

1. **Shared Context**: Children agent can have the full context of the main agent, so the children can immediatley have the full context without additional prompting or tool calls.
2. **Parallel Processing**: Besides the obvious speedn advantage, it can also improve the focus v.s. performing unrelated tasks in serial.
3. **Prompt caching**: The whole context window prefix is shared and can be cached, reducing the latency and cost.
4. **Performance optimization**: Besides prompt caching, the inference could be parallelized, and provide additional speedup & cost reduction, since the shared prefix are being used at the same time. e.g. [Hydragen: High-Throughput LLM Inference with Shared Prefixes](https://arxiv.org/abs/2402.05099)

## Implementation

Just like Unix's `fork()`:
```python
pid = fork()
if pid == 0:
    # Child process: handle subtask
else:
    # Parent process: wait for results
```

Our AI agent can call `fork()`, the flow looks like this:

```python
pid = fork(prompt)
if pid == 0:
    insert_tool_use_result('fork', 'you are a forked agent, please follow the instructions from the parent agent:')
    insert_user_message(prompt)
    # Child process: handle subtask from here and the last message will be collected by the parent process

    return final_message
else:
    # Parent process: wait for results from child process and collect the last message from the child process
    insert_tool_use_result('fork', final_message)

```

## Implementation Details

- `calculator_agent.py` is a simple base agent with some pre-defined tools.
- `calculator_fork_agent.py` inherits from `calculator_agent.py` and adds a fork tool.
- to run it, use `python calculator_fork_agent.py`, it will run a few test cases.


## Use cases
- A coding agent can use fork the converstaion to search/read for necessary information to help with debugging without filling up the context window.
    - This could be particularly useful for an async forking workflow.
- A coding agent can use fork to handle changes in multiple files in parallel.
- A research agent can use fork to read multiple sources in parallel, to filter out the noise and then read the most promising ones in detail.
- ... Or any tasks that can benefit from parallel processing or spawning a new agent to handle a subtask.

## Extensions
- We can extent the fork pattern to support asynchronous forking, where an agent can fork a child to perform some long running task while the parent agent continues with the main task.
- Multiple level of forking.
    - Currently, the sample code does not allow a child to fork another child. Because there is risk of spwaning too many child processes.
    - This probably requires something like a process manager to control the lifecycle of processes (agents).

- Agent handoff pattern in [OpenAI Swarm](https://github.com/openai/swarm)
    - It uses tool call to swap the system prompt and tools available, and also support environmental variables.
    - Handoff also benefit from a shared conversation history, but it's bad for prompt caching, and swapping system prompt is likely to be out of distribution (unless it’s trained into the model)
    - One could extend the agent forking pattern to include some of these features. Or even support handoff after a child process is forked. Similar to Fork-Exec pattern in Unix.

## License

MIT License