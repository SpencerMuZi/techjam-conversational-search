"""Small, injectable DeepSeek Chat Completions adapter (stdlib only)."""
from __future__ import annotations

import json
import math
import os
import time
from http.client import HTTPException
from dataclasses import dataclass, field
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Safe-to-display provider/protocol error; never contains response bodies."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    timeout: float = 30.0
    max_tokens: int = 1600
    max_steps: int = 8
    max_retries: int = 1
    fallback: bool = True

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("LLM base URL must be an HTTPS URL without credentials, query or fragment")
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 120:
            raise ValueError("LLM timeout must be in (0, 120] seconds")
        for name, maximum in (("max_steps", 20), ("max_tokens", 8192), ("max_retries", 3)):
            value = getattr(self, name)
            minimum = 0 if name == "max_retries" else 1
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid LLM {name}")
        if not self.model.strip():
            raise ValueError("LLM model must not be empty")

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            timeout=float(os.environ.get("SHOPPING_COPILOT_LLM_TIMEOUT", "30")),
            max_tokens=int(os.environ.get("SHOPPING_COPILOT_LLM_MAX_TOKENS", "1600")),
            max_steps=int(os.environ.get("SHOPPING_COPILOT_MAX_STEPS", "8")),
            fallback=os.environ.get("SHOPPING_COPILOT_LLM_FALLBACK", "true").lower() in {"true", "1", "yes"},
        )


@dataclass(frozen=True)
class LLMReply:
    message: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...


class DeepSeekClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        if not self.config.api_key.strip():
            raise LLMError("DEEPSEEK_API_KEY is not configured")
        payload = {
            "model": self.config.model, "messages": messages, "tools": tools,
            "tool_choice": "required", "stream": False,
            "max_tokens": self.config.max_tokens,
            # ReAct uses observation-driven tool selection; it does not require
            # exposing or storing a model's private reasoning_content.
            "thinking": {"type": "disabled"},
        }
        request = Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"},
            method="POST",
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                with urlopen(request, timeout=self.config.timeout) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise LLMError("LLM response exceeds size limit")
                return self._parse(raw)
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if (status == 429 or 500 <= status < 600) and attempt < self.config.max_retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise LLMError(f"LLM HTTP {status}") from None
            except (URLError, TimeoutError, OSError, HTTPException):
                if attempt < self.config.max_retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise LLMError("LLM request failed or timed out") from None
        raise LLMError("LLM request failed")

    @staticmethod
    def _parse(raw: bytes) -> LLMReply:
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            if choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise ValueError("Incomplete response")
            message = choice["message"]
            if message.get("content") is not None and not isinstance(message["content"], str):
                raise ValueError("Invalid message content")
            calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not isinstance(calls, list) or len(calls) > 4:
                raise ValueError("Invalid assistant message")
            identifiers = set()
            for call in calls:
                if (call.get("type") != "function" or not isinstance(call.get("id"), str)
                        or not call["id"] or call["id"] in identifiers
                        or not isinstance(call["function"]["name"], str)
                        or not isinstance(call["function"]["arguments"], str)):
                    raise ValueError("Invalid tool call")
                identifiers.add(call["id"])
            usage = data.get("usage") or {}
            counts = [usage.get(key, 0) for key in ("prompt_tokens", "completion_tokens")]
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError("Invalid usage")
            # Deliberately retain only protocol fields, never reasoning_content.
            clean = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                clean["tool_calls"] = calls
            return LLMReply(clean, *counts)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("Malformed or incomplete LLM response") from None
