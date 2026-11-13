# Stack decision

**Primary stack: LangGraph.** n8n is secondary and out of scope for this MVP.

## Why LangGraph fits this problem

A model intake review is a fixed procedure with strict data rules, and LangGraph lets the code enforce them. The graph state carries the target model, the 18 checklist items, every tool outcome and the findings, so each finding can be traced back to the exact source and quote that supports it. The graph branches where the problem branches: a conditional edge sends invalid input straight to the report renderer, source collection and regulatory retrieval run as parallel branches that join before assessment, and a failed source becomes a recorded outcome and a partial report instead of a crash. The trust checks are ordinary Python inside the nodes: every LLM quote is verified against the page text, evidence about a different model variant is downgraded, and every cited link is checked. All of it is unit-tested (67 tests, offline and with mocked services), and the stability command can rerun the same analysis to measure how much the result varies.

## Why n8n is secondary

n8n would be quicker for the delivery side of the product: a form or a Slack command as the trigger, the finished brief sent to email, Notion or an AI inventory. The core of this MVP is the opposite kind of work: validation and verification logic, structured output, deterministic ordering and tests, which are easier to write, test and version in Python than in visual nodes and Code nodes. Keeping one orchestration stack also keeps the MVP small. A thin n8n flow that calls this agent and delivers the report is a natural next step once the report format is validated with users.

## What the graph does not do yet

The day 1 design included a follow-up loop: search again for items still *not found* within a fixed tool budget. It is not built. The current graph makes one research pass per run, and the loop is the next technical step.
