# LLM context layer.
#
# Submodules:
# - polza_client.py: OpenAI-compatible client for Polza.ai
# - brief.py: pre-cycle market brief (TTL-cached, hourly)
# - explainer.py: async post-trade explanations
#
# The LLM layer NEVER blocks the hot trading path. All calls fail-soft and
# return cached/default values when the API is unavailable.

from moex_agent.llm.polza_client import PolzaClient, LLMResponse
from moex_agent.llm.brief import BriefService, MarketBrief, default_brief
from moex_agent.llm.explainer import ExplainerService
from moex_agent.llm.news import NewsService, compact_headlines

__all__ = [
    "PolzaClient",
    "LLMResponse",
    "BriefService",
    "MarketBrief",
    "default_brief",
    "ExplainerService",
    "NewsService",
    "compact_headlines",
]
