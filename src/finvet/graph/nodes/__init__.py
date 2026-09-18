"""Graph nodes for the verification pipeline.

Each node is a function that takes VerificationState and returns a dict
of updates to apply to the state. Nodes are connected by LangGraph.

The graph registers 12 nodes. At most 9 execute for any one claim -- the router
picks a single domain agent, and the two HITL nodes only run below the confidence
threshold.

1.  input_guardrails    - Validate and sanitize input
2.  claim_parser        - Parse claim into structured fields
3.  period_resolver     - Resolve time periods to dates (SEC claims only)
4.  sec_agent           - ReAct verification against SEC EDGAR
5.  market_agent        - ReAct verification against Finnhub
6.  news_agent          - ReAct verification against Tavily
7.  reject_handler      - Terminal path for unsafe / non-financial claims
8.  confidence_adjuster - Confidence adjustment on the agent verdict
9.  output_guardrails   - Confidence threshold + output safety -> HITL routing
10. hitl_checkpoint     - INTERRUPT point for human review
11. apply_hitl_decision - Apply the reviewer's decision
12. response_generator  - Format final response
"""

from .input_guardrails import input_guardrails
from .claim_parser import claim_parser
from .period_resolver import period_resolver
from .domain_agents import (
    run_sec_agent,
    run_market_agent,
    run_news_agent,
)
from .output_guardrails import output_guardrails
from .response_generator import response_generator

__all__ = [
    # Core pipeline nodes
    "input_guardrails",
    "claim_parser",
    "period_resolver",
    # Domain agent nodes
    "run_sec_agent",
    "run_market_agent",
    "run_news_agent",
    # Output
    "output_guardrails",
    "response_generator",
]
