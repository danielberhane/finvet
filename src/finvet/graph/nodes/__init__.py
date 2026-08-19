"""Graph nodes for the verification pipeline.

Each node is a function that takes VerificationState and returns a dict
of updates to apply to the state. Nodes are connected by LangGraph.

Node Order (9-node architecture):
1. input_guardrails - Validate and sanitize input
2. claim_parser - Parse claim into 6 fields
3. period_resolver - Resolve time periods to dates
4. domain_agents - Run SEC/Market/News agent based on claim_type
5. consensus - Combine evidence and determine verdict
6. output_guardrails - Check if HITL is needed
7. response_generator - Format final response
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
