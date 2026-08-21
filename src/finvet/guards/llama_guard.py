"""Llama Guard 3 provider via Ollama REST API.

Calls /api/generate with the Llama Guard prompt template.
Input classifies safety only (S1-S5, S7-S13) — advice-seeking is a
verifiability judgement the parser owns, not a safety block. The OUTPUT
template keeps the S6 "specialized advice" category so investment
recommendations in an agent's response route to human review.

Graceful degradation: if Ollama is down, returns safe=True with a
'llama_guard_unavailable' flag. Safety hardens but never blocks
when infrastructure fails.
"""

import time

import httpx

from ..config.settings import settings
from ..utils.logging import get_logger
from .provider import GuardResult

logger = get_logger(__name__)

# Input safety taxonomy: full Llama Guard 3 categories MINUS S6 (advice), which
# the parser owns as a verifiability rejection rather than a safety block.
_INPUT_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations
according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {user_message}

<END CONVERSATION>

Provide your safety assessment for the User message in the above conversation.
First line must be 'safe' or 'unsafe'.
If unsafe, a second line must include a comma-separated list of violated categories.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

_OUTPUT_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'Agent' messages in conversations
according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S6: Specialized Advice - The response provides specific investment recommendations \
(buy/sell/hold), portfolio allocation advice, or personalized financial guidance. \
Factual verification results, data citations, and confidence scores are SAFE.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {original_claim}

Agent: {agent_response}

<END CONVERSATION>

Provide your safety assessment for the Agent message in the above conversation.
First line must be 'safe' or 'unsafe'.
If unsafe, a second line must include a comma-separated list of violated categories.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


class LlamaGuardProvider:
    """Semantic safety classification via Llama Guard 3 on Ollama."""

    def classify_input(self, text: str) -> GuardResult:
        prompt = _INPUT_PROMPT_TEMPLATE.format(user_message=text)
        return self._classify(prompt, context_label="input")

    def classify_output(self, response_text: str, original_claim: str) -> GuardResult:
        prompt = _OUTPUT_PROMPT_TEMPLATE.format(
            original_claim=original_claim,
            agent_response=response_text,
        )
        return self._classify(prompt, context_label="output")

    def _classify(self, prompt: str, context_label: str) -> GuardResult:
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                f"{settings.llama_guard_url}/api/generate",
                json={
                    "model": settings.llama_guard_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=settings.llama_guard_timeout_s,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            safe, categories = _parse_response(raw)
            return GuardResult(
                safe=safe,
                categories=categories,
                violation_type="LLAMA_GUARD_UNSAFE" if not safe else None,
                provider="llama_guard",
                latency_ms=_ms(t0),
            )
        except Exception as e:
            logger.warning(f"Llama Guard unavailable ({context_label}): {e}")
            return GuardResult(
                safe=True,
                flags=["llama_guard_unavailable"],
                provider="llama_guard",
                latency_ms=_ms(t0),
            )


def _parse_response(raw: str) -> tuple[bool, list[str]]:
    """Parse Llama Guard response: first line 'safe'/'unsafe', second line categories."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return True, []
    first = lines[0].lower()
    if first == "safe":
        return True, []
    if first == "unsafe":
        categories = []
        if len(lines) > 1:
            categories = [c.strip() for c in lines[1].split(",") if c.strip()]
        return False, categories
    return True, []


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
