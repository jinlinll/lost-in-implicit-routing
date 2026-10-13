#!/usr/bin/env python3
"""Tencent Cloud TokenPlan OpenAI-compatible LLM client.

TokenPlan is exposed to OpenClaw as an OpenAI-compatible provider using
`api: openai-completions`, a `baseUrl`, an `apiKey`, and model IDs.  This
module implements the same wire protocol for the standalone SkillsBench agent:
POST {baseUrl}/chat/completions with `Authorization: Bearer <key>`.

Secrets are loaded from environment variables or the local `.env` file, but are
never logged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

TOKENPLAN_DEFAULT_BASE_URL = "https://api.lkeap.cloud.tencent.com/plan/v3"
TOKENPLAN_BASE_ENV_NAMES = (
    "TENCENT_TOKENPLAN_BASE_URL",
    "TENCENT_TOKEN_PLAN_BASE_URL",
    "TENCENT_TOKENPLAN_BASEURL",
    "Tencent_TokenPlan_Base_Url",
    "TOKENPLAN_BASE_URL",
    "TOKEN_PLAN_BASE_URL",
)
TOKENPLAN_KEY_ENV_NAMES = (
    "TENCENT_TOKENPLAN_KEY",
    "TENCENT_TOKEN_PLAN_KEY",
    "TENCENT_TOKENPLAN_API_KEY",
    "Tencent_TokenPlan_Key",
    "TOKENPLAN_API_KEY",
    "TOKEN_PLAN_API_KEY",
)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def read_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Read a simple KEY=VALUE .env file without requiring python-dotenv."""
    env_path = path or (Path(__file__).resolve().parent / ".env")
    if not env_path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if key:
            values[key] = value
    return values


def _get_first_env(names: tuple[str, ...], env_file: Optional[Path] = None) -> str:
    file_values = read_env_file(env_file)
    for name in names:
        value = os.environ.get(name)
        if value:
            return _strip_quotes(value)
        value = file_values.get(name)
        if value:
            return _strip_quotes(value)
    return ""


def get_default_tokenplan_api_base(env_file: Optional[Path] = None) -> str:
    """Return TokenPlan base URL from env/.env, or the documented default."""
    return (_get_first_env(TOKENPLAN_BASE_ENV_NAMES, env_file) or TOKENPLAN_DEFAULT_BASE_URL).rstrip("/")


def get_default_tokenplan_api_key(env_file: Optional[Path] = None) -> str:
    """Return TokenPlan API key from env/.env. Empty means not configured."""
    return _get_first_env(TOKENPLAN_KEY_ENV_NAMES, env_file)


def build_chat_completions_url(base_url: str) -> str:
    """Build a chat completions endpoint from an OpenAI-compatible base URL."""
    base = (base_url or TOKENPLAN_DEFAULT_BASE_URL).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class TokenPlanLLMClient:
    """OpenClaw-style TokenPlan client compatible with SimpleLLMClient."""

    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.logger = logger or logging.getLogger("experiment.llm")

    def _api_base(self) -> str:
        return (getattr(self.config, "api_base", "") or get_default_tokenplan_api_base()).rstrip("/")

    def _api_key(self) -> str:
        return getattr(self.config, "api_key", "") or get_default_tokenplan_api_key()

    def _payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
        }
        temperature = getattr(self.config, "temperature", None)
        top_p = getattr(self.config, "top_p", None)
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    def call(self, messages: List[Dict[str, str]], max_retries: int = 3, retry_delay: float = 30.0) -> str:
        """Call TokenPlan /chat/completions and return response content."""
        url = build_chat_completions_url(self._api_base())
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "Tencent TokenPlan API key is not configured. Set Tencent_TokenPlan_Key "
                "or TENCENT_TOKENPLAN_API_KEY in .env/environment."
            )

        payload = self._payload(messages)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        self.logger.info(f"[TokenPlan] Calling {url} with model={self.config.model_name}")
        self.logger.info(
            f"[TokenPlan] Message count: {len(messages)}, "
            f"last message length: {len(messages[-1].get('content', ''))} chars"
        )

        response: Optional[requests.Response] = None
        last_error: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"[TokenPlan] Sending request (attempt {attempt}/{max_retries})...")
                response = requests.post(url, headers=headers, json=payload, timeout=300)
                self.logger.info(f"[TokenPlan] Response status: {response.status_code}")

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait_time = float(retry_after) if retry_after else retry_delay * attempt
                    except ValueError:
                        wait_time = retry_delay * attempt
                    last_error = RuntimeError(f"TokenPlan rate limit 429: {response.text}")
                    self.logger.warning(
                        f"[TokenPlan] Rate limited 429 (attempt {attempt}/{max_retries}): "
                        f"{response.text[:500]}"
                    )
                    if attempt < max_retries:
                        time.sleep(wait_time)
                        continue
                    raise last_error

                if response.status_code >= 500:
                    last_error = RuntimeError(f"TokenPlan API error {response.status_code}: {response.text}")
                    self.logger.warning(
                        f"[TokenPlan] Server error {response.status_code} "
                        f"(attempt {attempt}/{max_retries}): {response.text[:500]}"
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    raise last_error

                if response.status_code != 200:
                    self.logger.error(f"[TokenPlan] Client error {response.status_code}: {response.text[:500]}")
                    raise RuntimeError(f"TokenPlan API error {response.status_code}: {response.text}")

                try:
                    data = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = RuntimeError(f"TokenPlan returned invalid JSON: {exc}")
                    self.logger.warning(
                        f"[TokenPlan] Invalid JSON response (attempt {attempt}/{max_retries}): {exc}"
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    raise last_error

                # A small number of HTTP-200 responses contain an empty message.
                # Treat those as transient provider failures, not fatal agent runs.
                self._track_usage(data)
                try:
                    content = self._extract_content(data)
                except RuntimeError as exc:
                    last_error = exc
                    retryable_empty = (
                        "missing content" in str(exc).lower()
                        or "missing choices" in str(exc).lower()
                    )
                    if retryable_empty and attempt < max_retries:
                        self.logger.warning(
                            f"[TokenPlan] Empty/incomplete 200 response "
                            f"(attempt {attempt}/{max_retries}); retrying: {exc}"
                        )
                        time.sleep(retry_delay)
                        continue
                    raise

                if attempt > 1:
                    self.logger.info(f"[TokenPlan] Request succeeded on attempt {attempt}")
                self.logger.info(f"[TokenPlan] Response length: {len(content)} chars")
                return content

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = exc
                self.logger.warning(f"[TokenPlan] Connection error (attempt {attempt}/{max_retries}): {exc}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                raise

        if response is None:
            raise RuntimeError(f"TokenPlan request failed: {last_error}")

        raise RuntimeError(f"TokenPlan request failed: {last_error}")

    def _track_usage(self, data: Dict[str, Any]) -> None:
        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)
        if usage:
            self.logger.info(f"[TokenPlan] Tokens: {input_tokens} in, {output_tokens} out")

    def _extract_content(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"TokenPlan response missing choices: {json.dumps(data, ensure_ascii=False)[:500]}")

        message = choices[0].get("message") or {}
        content = message.get("content")

        if content is None:
            refusal = message.get("refusal", "")
            if refusal:
                raise RuntimeError(f"Model refused to respond: {refusal}")
            content = message.get("reasoning_content")

        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "".join(text_parts)

        if not isinstance(content, str) or content == "":
            raise RuntimeError(f"TokenPlan response missing content. Message keys: {list(message.keys())}")
        return content
