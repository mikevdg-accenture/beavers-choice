# Reflection Report

This project has been an illuminating experience working with AI agents, and the difficulty of getting them to co-ordinate with each other. It has been akin to herding cats, except the cats have the intelligence of 3-year-old children, and are tripping on hallucinogenics. 

My main learning experience has been how profoundly difficult it is to get an AI agent to reliably pass simple information from one system to another without it forgetting parts of the input, or inventing data from thin air when deprived of proper input by a parent agent. The ramifications of this are that I am now exceptionally wary of trusting AI with any system that relies on accuracy, such as, for example, a simple stock management system.

## Workflow explanation

For this project, an Inventory agent and Sales agent were added. For coordination and interaction with the customer, an over-arching Orchestration agent was added. Later, an Item Matching agent was added to work around problems with item matching, and a sanisation agent as a result of project feedback on the first submission.

The Orchestration agent is provided with no tools and must delegate to other agents. Each agent is provided with an adequate description which is provided to the Orchestration agent by the `smolagents` framework. 

For it's workflow, the Orchestration agent is effectively left to its own devices. For this pedagogical project, this is effective enough, but difficulties during testing have revealed the value of establishing a formal state machine to guide agents. 

## Decision-making process

I attempted to not hard-code the agent's thought process, as I was going by the assumption that customer requests might not simply be incoming orders but might also be queries about products, stock levels, just having a chat (as alluded to with the "mood" field) or asking very general questions about how a problem they have might be solved. Providing the Orchestration agent with context about it's role and abilities allowed it to be flexible.

## Test results analysis

I have spent far too long tuning the prompts for this exercise. One issue that appeared early was that the requested items often don't appear in our inventory, or have items in our inventory that are similar. For these, I spent a long time trying to get the agents to return suggestions back to the customer.

Agents also hallucinate and forget items. I have decided to accept my fate here. Avoiding this requires a stricter framework build around the agents to verify and validate their decisions at each stage.

I am yet to be convinced that AIs are reliable enough to use for handling orders. Each test run of this project yielded new and unpredictable results. This has reinforced the importance of thorough prototyping before committing to any large AI project.

This agent framework is also far too slow to be used in a realistic scenario. 

## Strengths of multi-agent approach

There are many reasons for using a multi-agent approach:

* A multi-agent approach allows the context window to be managed by spreading it across multiple agents.
* Within a multi-agent system, smaller models with smaller system prompts could be used for simpler tasks to reduce token usage and improve performance. Conversely, larger models with more complex system prompts and higher hardware requirements could be used for more complex tasks.
* Specialised, fine-tuned agents could be used for custom tasks such as SQL generation.
* Agents allow for parallel processing, which enables an agentic system to scale.
* Agents can be used to verify each other's decisions, which reduces hallucinations.

However, my opinion is that more than one agent is not required for this particular problem. A multiple-agent solution has only been implemented for the learning experience.

## Suggestions for improvement

Some ideas I have for improving this project are:

* Requests would be classified as "order", "query" or "other" on arrival by a classification agent. This allows orders to be handled by a more rigid framework with verification.
* For "order", before handing the request to the Orchestrator, an agent should pre-process the request to map the customer's names for products to the products that actually exist in our database. This would be verified by a verification agent before continuing.
* Rather than fullfilling orders using a tool, an "order conversion" agent would convert orders to JSON which is then processed. This prevents an agent from processing the same order twice or not at all.
