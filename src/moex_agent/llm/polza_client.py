"""Polza.ai OpenAI-compatible chat client.

Polza aggregates many LLMs through a single OpenAI-compatible API at
`https://api.polza.ai/v1/chat/completions`. We pick the model per call,
so the agent can use cheap-and-fast for high-frequency tasks
(post-trade explanations) and a premium model for daily retrospectives.

Design contracts:
- LLM is OUTSIDE the hot trading path. This client never raises on
  network/parse errors — it returns None and lets the caller fall back
  to a cached value or simply skip the LLM step.
- Hard timeout (default 15s) — Polza usually answers in 1-3s.
- Exponential backoff retry on 5xx and connection errors only; never
  retry on 4xx (configuration error).
- JSON mode is requested whenever we ask for structured output, so the
  caller can `json.loads()` the response without prompt-engineering
  parsers.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.polza.ai"


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def as_json(self) -> dict[str, Any] | None:
        """Parse content as JSON. Returns None if not parseable."""
        if not self.content:
            return None
        text = self.content.strip()
        # Tolerate ```json fences from chatty models.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip("\n")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract the first balanced {...} object — handles models
            # that wrap the JSON in reasoning prose or stray text.
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning("llm response was not valid JSON", extra={"extra": {"snippet": text[:200]}})
            return None


class PolzaClient:
    """Thin OpenAI-compatible client for Polza.ai.

    Usage:
        client = PolzaClient(api_key=..., base_url="https://api.polza.ai")
        resp = client.chat(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
            max_tokens=500,
            json_mode=True,
        )
        if resp is not None:
            data = resp.as_json()
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.5,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.local = self.base_url != DEFAULT_BASE_URL.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._session.headers.update({"Content-Type": "application/json"})

    def enabled(self) -> bool:
        # Свой эндпоинт (Ollama, LM Studio, vLLM) работает без ключа.
        return bool(self.api_key) or self.local

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse | None:
        # NOTE: do NOT send reasoning_effort here. Polza's proxy rejects it
        # with 4xx for some upstream models, producing silent no_response in
        # production. If we need GPT-5 thinking control later, validate
        # against the live API first.
        if not self.api_key and not self.local:
            logger.debug("llm client called without api key — skipping")
            return None
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if json_mode:
            # Polza is OpenAI-compatible; OpenAI uses response_format.type='json_object'.
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/v1/chat/completions"
        last_err: str | None = None
        for attempt in range(1, self.max_retries + 2):  # initial + retries
            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)
                if response.status_code >= 500:
                    last_err = f"HTTP {response.status_code}: {response.text[:300]}"
                    self._sleep_backoff(attempt)
                    continue
                if response.status_code >= 400:
                    # 4xx is configuration error — bail without retry.
                    logger.warning(
                        "polza 4xx",
                        extra={"extra": {"status": response.status_code, "body": response.text[:400], "model": model}},
                    )
                    return None
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = str(message.get("content") or "").strip()
                # Some reasoning models (deepseek-v4-flash) intermittently return
                # an EMPTY content (token budget eaten by hidden reasoning). Treat
                # empty as transient and retry, instead of handing a guaranteed
                # parse failure (invalid_json) upstream.
                if not content and attempt <= self.max_retries:
                    last_err = "empty content from model"
                    self._sleep_backoff(attempt)
                    continue
                usage = data.get("usage") or {}
                return LLMResponse(
                    content=content,
                    model=str(data.get("model") or model),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                self._sleep_backoff(attempt)
                continue
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("polza unexpected error", extra={"extra": {"error": last_err, "model": model}})
                return None
        logger.warning(
            "polza request exhausted retries",
            extra={"extra": {"model": model, "last_error": last_err}},
        )
        return None

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt > self.max_retries:
            return
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        time.sleep(delay)
