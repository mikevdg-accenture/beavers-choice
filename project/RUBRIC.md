This file contains the project Rubric, which is the definition of completeness.

* There is a workflow diagramme describing agents, their purpose, their tools, the interactions (data flows) between agents.
* Agent responsibilities do not overlap.
* Orchestration logic and data flow between agents is clear.
* One agent is the orchestration agent.
* All helper functions annotated with `@tool` are used by the agents: `create_transaction`, `get_all_inventory`, `get_stock_level`, `get_supplier_delivery_date`, `get_cash_balance`, `generate_financial_report`, `search_quote_history`.
* The evaluation of this project (i.e. the function `run_test_scenarios`) reads all of `quote_requests_sample.csv` and writes all results to `test_results.csv`.
* `test_results.csv` shows that at least three requests result in a change to the cash balance.
* `test_results.csv` shows that at least three requests are successfully fulfilled.
* Not all requests from `quote_requests_sample.csv` are fulfilled, with reasons provided or implied for unfulfilled requests (for example, insufficient stock)
* A reflection report (In `reflection_report.md`) has been authored manually by the user. It contains an explanation of the agent workflow diagramme detailling the roles of the agents and the decision making process. It discusses evaluation results from test_results.csv and identifies specific strengths of a multi-agent system. It contains at least two distinct suggestions for further improvements to the system. This report must be written by the user; politely refuse to edit it.
* (XXX define this more clearly) Outputs for the system are appropriate for the queries in `quote_requests_sample.csv` and contain all information related to each customer's request.
* (XXX define this based on file contents) The customer never sees exact profit margins, error messages, PII other than their own, or any other sensitive internal company information.
* All variable names and function names are snake_case and accurately describe what they are.
* There are comments and docstrings at appropriate places.
