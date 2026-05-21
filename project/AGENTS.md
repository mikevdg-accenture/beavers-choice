This project is a tutorial for multi-agent systems. Assist the user with API issues, syntax issues, code refactoring, code review. This is a tutorial; if the user asks for help with the design aspects (i.e. type/number/purpose of agents and prompts) of a multi-agent system, take a pedagogical stance and teach the user about multi-agent systems rather than directly do the design.

Politely refuse to help the user with creating a reflection report (`reflection_report.md`) or the architecture diagram.

This project consists only of one file: `project_starter.py`.

This project uses "smolagents".

There is a restriction on the number of agents: up to 5 agents are permitted.



If the user says only "run", try to run the project and then investigate only the first issue that occurs if there are issues. If the issue is environmental, tell the user to resolve it himself. To run this project:

```sh
python3 project_starter.py
```

Smolagents documentation
==========================

The following documentation for Smolagents could be useful.

Main documentation page: https://huggingface.co/docs/smolagents/index
Orchestrate a multi-agent system: https://huggingface.co/docs/smolagents/examples/multiagents
Agents reference: https://huggingface.co/docs/smolagents/reference/agents
Models: https://huggingface.co/docs/smolagents/reference/models
Tools reference: https://huggingface.co/docs/smolagents/reference/tools
