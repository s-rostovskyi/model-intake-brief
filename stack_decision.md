# Stack decision

LangGraph is the primary stack because each checklist item needs a bounded research loop that can search, assess evidence and revisit gaps. The graph state tracks the model version, evidence, source checks and remaining tool budget so both open-weight and hosted models can use the same workflow. Conditional edges make it possible to test follow-up search and partial results when a source fails. n8n would be useful for future intake notifications and report delivery, and we are deferring it to keep one orchestration stack for this MVP.
