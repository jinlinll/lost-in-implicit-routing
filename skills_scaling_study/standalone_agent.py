#!/usr/bin/env python3
"""
Standalone Agent for SkillsBench tasks.

This agent is independent of harbor and litellm, using direct HTTP calls to the LLM API.
It implements the same logic as harbor's HarborTerminus2WithSkills agent.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import time
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import requests

from tokenplan_client import (
    TokenPlanLLMClient,
    get_default_tokenplan_api_base,
    get_default_tokenplan_api_key,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - dynamic RAG requires numpy at runtime
    np = None

DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QWEN_EMBEDDING_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_EMBEDDING_API_KEY = os.environ.get("QWEN_EMBEDDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")

# Import skill preselection module
from skill_preselect import (
    SkillPreselector, 
    NaivePreselector, 
    PrecomputedRAGPreselector,
    create_preselector,
    SkillInfo,
)

# Docker SDK (optional, for SDK backend)
try:
    import docker
    from docker.models.containers import Container
except ImportError:
    docker = None
    Container = None


# =============================================================================
# Docker Resource Limits Configuration
# =============================================================================
# 限制 Docker 容器资源使用，防止服务器过载导致无法连接
# Local default budget: Docker Desktop currently exposes 10 CPUs and 8GB RAM.
# Keep headroom for Docker Desktop and the host.  Larger machines can override
# these values without editing code.
DOCKER_TOTAL_CPUS = int(os.environ.get("SKILLS_DOCKER_TOTAL_CPUS", "8"))
DOCKER_TOTAL_MEMORY_GB = float(os.environ.get("SKILLS_DOCKER_TOTAL_MEMORY_GB", "6"))
DOCKER_MEMORY_SWAP_MULTIPLIER = float(
    os.environ.get("SKILLS_DOCKER_MEMORY_SWAP_MULTIPLIER", "1")
)

def get_docker_resource_args(max_parallel: int = 4) -> List[str]:
    """
    返回 Docker 资源限制的命令行参数（每个容器的限制）
    
    Args:
        max_parallel: 最大并发容器数，用于计算每个容器的资源配额
    
    Returns:
        Docker run 命令的资源限制参数列表
    """
    # 动态计算每个容器的资源配额
    cpus_per_container = max(1, DOCKER_TOTAL_CPUS // max_parallel)
    # Allocate in MiB so a 6 GiB budget can be divided across four workers as
    # 1536 MiB each.  Integer GiB division previously reduced this to 1 GiB
    # per container and made the four-way follow-up unnecessarily OOM-prone.
    memory_per_container_mb = max(
        1024,
        int(DOCKER_TOTAL_MEMORY_GB * 1024) // max_parallel,
    )
    memory_swap_per_container_mb = max(
        memory_per_container_mb,
        int(memory_per_container_mb * DOCKER_MEMORY_SWAP_MULTIPLIER),
    )
    
    return [
        "--cpus", str(cpus_per_container),
        "--memory", f"{memory_per_container_mb}m",
        "--memory-swap", f"{memory_swap_per_container_mb}m",
    ]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Command:
    """A command to execute in the terminal."""
    keystrokes: str
    duration_sec: float = 1.0


@dataclass
class ParseResult:
    """Result of parsing LLM response."""
    commands: List[Command]
    is_task_complete: bool
    error: str = ""
    warning: str = ""


@dataclass
class SkillMetadata:
    """Metadata for a skill."""
    name: str
    description: str
    location: str


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model_name: str
    api_base: str = field(default_factory=get_default_tokenplan_api_base)
    api_key: str = field(default_factory=get_default_tokenplan_api_key)
    llm_provider: str = "tokenplan"  # "tokenplan" (default) or "openai-compatible"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_episodes: int = 100
    response_format: str = "json"  # "json" or "xml"
    max_output_bytes: int = 10000
    max_context_turns: int = 3  # Keep system prompt + last N turns when context is too long
    
    # Skill pre-selection settings
    preselect_strategy: str = "naive"  # "naive", "llm", "rag", "hybrid-rag-llm", "multi-query-rag-llm", or "dynamic-rag"
    preselect_max_skills: int = 5  # Max skills to select (ignored for "naive")
    preselect_model_name: Optional[str] = None  # LLM model for "llm" strategy (defaults to main model)
    preselect_embedding_model: str = DEFAULT_QWEN_EMBEDDING_MODEL  # Qwen/DashScope embedding model for "rag" strategy
    rag_precomputed_dir: str = ""  # Path to pre-computed RAG results (for "rag"/"hybrid-rag-llm" strategies)
    hybrid_rag_retrieval_k: int = 100  # RAG candidates passed to LLM reranker for hybrid-rag-llm


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    episodes: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error: Optional[str] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    loaded_skills: List[Dict[str, Any]] = field(default_factory=list)
    # loaded_skills: [{"episode": 2, "skill": "mesh-analysis"}, ...]
    
    # Pre-selection info
    preselect_strategy: str = "naive"
    preselected_skills: List[str] = field(default_factory=list)  # Names of skills pre-selected into prompt
    preselect_input_tokens: int = 0  # Tokens used for pre-selection (LLM strategy)
    preselect_output_tokens: int = 0


# =============================================================================
# Model Context Window Registry
# =============================================================================

# Known context window sizes (in tokens) for common models.
# Used for proactive context management to avoid hitting API limits.
# Conservative estimates — actual limits may be slightly higher.
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    # GPT-5 series
    "gpt-5": 400_000,
    "gpt-5-chat": 128_000,   # Chat variant: 128k in, 16k out, no function call
    "gpt-5-mini": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5.1-chat": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "gpt-5-nano": 400_000,
    # GPT-4 series
    "gpt-4o": 128_000,
    "gpt-4o-2024-11-20": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # OpenAI o-series (reasoning)
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-pro": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # ── Anthropic / Claude ──────────────────────────────────────────────────
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-4-5-sonnet-20250929": 200_000,
    "claude-4-sonnet-20250514": 200_000,
    "claude-3-7-sonnet-20250219": 200_000,
    "claude-opus-4-5-20251101": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    # ── Google Gemini ───────────────────────────────────────────────────────
    "gemini-3.1-pro": 1_048_576,   # 1024K
    "gemini-3.1-flash-lite": 1_048_576,  # 1024K
    "gemini-3-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "gemini-2.5-flash-image": 65_000,
    "gemini-3.1-flash-image": 65_000,   # Nano Banana 2
    "gemini-3-pro-image": 65_000,      # Nano Banana Pro
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    # ── Meta Llama ──────────────────────────────────────────────────────────
    "llama-4-maverick": 1_000_000,
    "llama-4-scout": 512_000,
    "llama-3.3-70b": 128_000,
    "llama-3.1-405b": 128_000,
    "llama-3.1-70b": 128_000,
    "llama-3.1-8b": 128_000,
    # ── Mistral ─────────────────────────────────────────────────────────────
    "mistral-large": 128_000,
    "mistral-medium": 32_000,
    "mistral-small": 32_000,
    # ── DeepSeek ────────────────────────────────────────────────────────────
    "deepseek-v3.2": 64_000,
    "deepseek-v3.1-terminus": 64_000,
    "deepseek-v3-local-II": 64_000,
    "deepseek-r1-local-II": 64_000,
    "deepseek-r1-local-III": 64_000,
    "deepseek-chat": 64_000,
    "deepseek-r1": 64_000,
    "deepseek-v3": 64_000,
    "deepseek-ocr": 8_000,
    # ── Qwen ─────────────────────────────────────────────────────────────────
    "qwen3-235b-a22b-2507-fp8": 128_000,
    "qwen3-235b-a22b-thinking-2507-fp8": 128_000,
    "qwen3-vl-235b-a22b-thinking": 128_000,
    "qwen3-vl-235b-a22b-instruct": 128_000,
    "qwen3.5-397b-a17b": 64_000,
    "qwen3.5-35b-a3b": 64_000,
    "qwen3-30b-a3b-instruct-2507": 64_000,
    "qwen3-30b-a3b-thinking-2507": 64_000,
    "qwen3-omni-30b-a3b-thinking": 64_000,
    "qwen3-omni-30b-a3b-instruct": 64_000,
    "qwen3-235b-a22b": 128_000,
    "qwen-max": 128_000,
    "qwen-plus": 128_000,
    "qwen-turbo": 128_000,
    # ── Kimi ─────────────────────────────────────────────────────────────────
    "kimi-k2.5": 128_000,
    "kimi-k2-instruct-local": 64_000,
    "kimi-k2-instruct-0905-local": 128_000,
    # ── GLM (智谱) ───────────────────────────────────────────────────────────
    "glm-5": 128_000,
    "glm-4.7": 128_000,
    # ── MiniMax ──────────────────────────────────────────────────────────────
    "minimax-m2.5": 128_000,
    # ── ARC Video ────────────────────────────────────────────────────────────
    # arc-video-7b-v1.1 / v2: no context window info available
}

# Default context window if model is not in the registry
DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model_name: str) -> int:
    """
    Get the context window size for a model.
    
    Uses exact match first, then tries substring/prefix matching
    for model variants (e.g., "gpt-4o-2024-05-13" matches "gpt-4o").
    
    Returns:
        Context window size in tokens
    """
    # Exact match
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]
    
    # Try prefix matching (longest match first for specificity)
    candidates = sorted(MODEL_CONTEXT_WINDOWS.keys(), key=len, reverse=True)
    for known_name in candidates:
        if model_name.startswith(known_name):
            return MODEL_CONTEXT_WINDOWS[known_name]
    
    # Try substring matching (e.g., model_name contains "gpt-4o")
    for known_name in candidates:
        if known_name in model_name:
            return MODEL_CONTEXT_WINDOWS[known_name]
    
    return DEFAULT_CONTEXT_WINDOW


def estimate_token_count(messages: List[Dict[str, str]]) -> int:
    """
    Rough estimate of token count for a list of messages.
    
    Uses the approximation of ~4 characters per token (conservative).
    This avoids requiring tiktoken as a dependency.
    """
    total_chars = 0
    for msg in messages:
        # Each message has ~4 tokens overhead (role, formatting)
        total_chars += 16  # ~4 tokens * 4 chars/token
        content = msg.get("content", "")
        total_chars += len(content)
    return total_chars // 4


def trim_messages_to_fit(
    messages: List[Dict[str, str]],
    max_tokens: int,
    max_context_turns: int = 3,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, str]]:
    """
    Trim message history to fit within the context window.
    
    Strategy:
    1. Always keep the first message (system prompt / initial instruction).
    2. Keep the last `max_context_turns` conversation turns (each turn = 
       1 assistant message + 1 user message = 2 messages).
    3. If still over the limit, progressively reduce turns.
    4. If even 1 turn + system prompt exceeds the limit, keep only the
       system prompt and the last user message (best effort).
    
    Args:
        messages: Full message history
        max_tokens: Maximum context window in tokens
        max_context_turns: Maximum number of recent turns to keep
        logger: Optional logger for reporting trimming actions
    
    Returns:
        Trimmed list of messages that fits within the context window
    """
    # Leave 20% headroom for the model's response
    effective_limit = int(max_tokens * 0.80)
    
    current_estimate = estimate_token_count(messages)
    
    if current_estimate <= effective_limit:
        return messages  # No trimming needed
    
    if logger:
        logger.info(f"[Context] Estimated tokens: ~{current_estimate}, limit: {effective_limit} "
                     f"(window: {max_tokens}). Trimming needed.")
    
    # The first message is always the system prompt / initial instruction
    system_msg = [messages[0]]
    remaining = messages[1:]  # The conversation turns after the system prompt
    
    # Each "turn" is (assistant, user) — 2 messages per turn
    # Try keeping max_context_turns turns, then reduce if still too large
    for n_turns in range(max_context_turns, 0, -1):
        n_messages_to_keep = n_turns * 2
        if len(remaining) <= n_messages_to_keep:
            trimmed = system_msg + remaining
        else:
            trimmed = system_msg + remaining[-n_messages_to_keep:]
        
        est = estimate_token_count(trimmed)
        if est <= effective_limit:
            if logger:
                dropped = len(messages) - len(trimmed)
                logger.info(f"[Context] Trimmed to {n_turns} recent turns "
                            f"(dropped {dropped} messages, ~{est} tokens)")
            return trimmed
    
    # Last resort: system prompt + last message only
    trimmed = system_msg + [remaining[-1]] if remaining else system_msg
    if logger:
        dropped = len(messages) - len(trimmed)
        est = estimate_token_count(trimmed)
        logger.warning(f"[Context] Aggressive trim: system prompt + last message only "
                       f"(dropped {dropped} messages, ~{est} tokens)")
    return trimmed


# =============================================================================
# LLM Client (Direct HTTP, no litellm)
# =============================================================================

class SimpleLLMClient:
    """Simple LLM client using direct HTTP requests."""
    
    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.logger = logger or logging.getLogger("experiment.llm")
    
    def call(self, messages: List[Dict[str, str]], max_retries: int = 3, retry_delay: float = 30.0) -> str:
        """Call the LLM API and return the response content.
        
        Retries on 5xx errors, 429 rate limits, and connection errors
        (up to max_retries times).
        """
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        self.logger.info(f"[LLM] Calling {url} with model={self.config.model_name}")
        self.logger.info(f"[LLM] Message count: {len(messages)}, last message length: {len(messages[-1]['content'])} chars")
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"[LLM] Sending request (attempt {attempt}/{max_retries})...")
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=300  # 5 minute timeout
                )
                
                self.logger.info(f"[LLM] Response status: {response.status_code}")
                
                if response.status_code == 429:
                    # Rate limit — retryable with longer backoff
                    error_text = response.text[:500]
                    # Try to extract retry-after header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = retry_delay
                    else:
                        wait_time = retry_delay * attempt  # Exponential-ish backoff
                    self.logger.warning(f"[LLM] ⚠️ Rate limited 429 (attempt {attempt}/{max_retries}): {error_text}")
                    last_error = RuntimeError(f"API rate limit 429: {response.text}")
                    if attempt < max_retries:
                        self.logger.warning(f"[LLM] ⏳ Rate limited, waiting {wait_time:.0f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    self.logger.error(f"[LLM] ❌ Rate limit persisted after {max_retries} retries, giving up.")
                    raise last_error
                
                if response.status_code >= 500:
                    # Server error — retryable
                    error_text = response.text[:500]
                    self.logger.warning(f"[LLM] ⚠️ Server error {response.status_code} (attempt {attempt}/{max_retries}): {error_text}")
                    last_error = RuntimeError(f"API error {response.status_code}: {response.text}")
                    if attempt < max_retries:
                        self.logger.warning(f"[LLM] ⏳ Server error, waiting {retry_delay:.0f}s before retry...")
                        time.sleep(retry_delay)
                        continue
                    self.logger.error(f"[LLM] ❌ Server error persisted after {max_retries} retries, giving up.")
                    raise last_error
                
                if response.status_code != 200:
                    # Client error (4xx, not 429) — not retryable
                    self.logger.error(f"[LLM] ❌ Client error {response.status_code}: {response.text[:500]}")
                    raise RuntimeError(f"API error {response.status_code}: {response.text}")
                
                # Success — break out of retry loop
                if attempt > 1:
                    self.logger.info(f"[LLM] ✓ Request succeeded on attempt {attempt}")
                break
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Network error — retryable
                self.logger.warning(f"[LLM] ⚠️ Connection error (attempt {attempt}/{max_retries}): {e}")
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(f"[LLM] ⏳ Connection error, waiting {retry_delay:.0f}s before retry...")
                    time.sleep(retry_delay)
                    continue
                self.logger.error(f"[LLM] ❌ Connection error persisted after {max_retries} retries, giving up.")
                raise
        
        data = response.json()
        
        # Track token usage
        if "usage" in data:
            self.total_input_tokens += data["usage"].get("prompt_tokens", 0)
            self.total_output_tokens += data["usage"].get("completion_tokens", 0)
            self.logger.info(f"[LLM] Tokens: {data['usage'].get('prompt_tokens', 0)} in, {data['usage'].get('completion_tokens', 0)} out")
        
        message = data["choices"][0]["message"]
        content = message.get("content")
        
        # Handle missing or null content (e.g. refusal, tool_calls-only response)
        if content is None:
            refusal = message.get("refusal", "")
            if refusal:
                self.logger.warning(f"[LLM] Model refused: {refusal}")
                raise RuntimeError(f"Model refused to respond: {refusal}")
            
            # Handle Kimi's reasoning_content field (sometimes content is in reasoning_content instead)
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                self.logger.info(f"[LLM] Using reasoning_content as content (Kimi model behavior)")
                content = reasoning_content
            else:
                # Log the full message for debugging
                self.logger.warning(f"[LLM] Response message has no 'content'. Keys: {list(message.keys())}")
                self.logger.warning(f"[LLM] Full message: {json.dumps(message, ensure_ascii=False)[:500]}")
                raise RuntimeError(f"LLM response missing 'content'. Message keys: {list(message.keys())}")
        
        # Handle multimodal response format where content is a list
        # Some models (like Gemini) return content as a list of text chunks
        if isinstance(content, list):
            # Extract text parts from the content list and join without separators
            # Using "" instead of "\n" to avoid inserting control characters into JSON
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "".join(text_parts)  # Join without newlines to preserve JSON integrity
            self.logger.info(f"[LLM] Converted list content ({len(data['choices'][0]['message']['content'])} items) to string")
        
        self.logger.info(f"[LLM] Response length: {len(content)} chars")
        return content


# =============================================================================
# Response Parsers
# =============================================================================

class JSONResponseParser:
    """Parser for JSON format responses."""
    
    required_fields = ["analysis", "plan", "commands"]
    
    def parse_response(self, response: str) -> ParseResult:
        """Parse a JSON response from the LLM."""
        warnings = []
        
        # Extract JSON content
        json_content = self._extract_json(response)
        if not json_content:
            return ParseResult([], False, "No valid JSON found in response")
        
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            return ParseResult([], False, f"Invalid JSON: {e}")
        
        # Validate structure
        if not isinstance(data, dict):
            return ParseResult([], False, "Response must be a JSON object")
        
        missing = [f for f in self.required_fields if f not in data]
        if missing:
            return ParseResult([], False, f"Missing required fields: {', '.join(missing)}")
        
        # Check if task is complete
        is_complete = data.get("task_complete", False)
        if isinstance(is_complete, str):
            is_complete = is_complete.lower() in ("true", "1", "yes")
        
        # Parse commands
        commands = []
        commands_data = data.get("commands", [])
        if not isinstance(commands_data, list):
            return ParseResult([], False, "Field 'commands' must be an array")
        
        for i, cmd in enumerate(commands_data):
            if not isinstance(cmd, dict):
                return ParseResult([], False, f"Command {i+1} must be an object")
            
            if "keystrokes" not in cmd:
                return ParseResult([], False, f"Command {i+1} missing 'keystrokes' field")
            
            keystrokes = cmd["keystrokes"]
            duration = cmd.get("duration", 1.0)
            
            if not isinstance(duration, (int, float)):
                warnings.append(f"Command {i+1}: invalid duration, using default 1.0")
                duration = 1.0
            
            commands.append(Command(keystrokes=keystrokes, duration_sec=float(duration)))
        
        warning_str = "- " + "\n- ".join(warnings) if warnings else ""
        return ParseResult(commands, is_complete, "", warning_str)
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON object from response text."""
        brace_count = 0
        start_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response):
            if escape_next:
                escape_next = False
                continue
            
            if char == "\\":
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == "{":
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        return response[start_idx:i+1]
        
        return ""


class XMLResponseParser:
    """Parser for XML format responses."""
    
    required_sections = ["analysis", "plan", "commands"]
    
    def parse_response(self, response: str) -> ParseResult:
        """Parse an XML response from the LLM."""
        warnings = []
        
        # Extract <response> content
        response_match = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
        if not response_match:
            # Try without closing tag
            response_match = re.search(r"<response>(.*)", response, re.DOTALL)
            if not response_match:
                return ParseResult([], False, "No <response> tag found")
            warnings.append("Missing </response> closing tag")
        
        content = response_match.group(1).strip()
        
        # Check if task is complete
        complete_match = re.search(r"<task_complete>\s*true\s*</task_complete>", content, re.IGNORECASE)
        is_complete = bool(complete_match)
        
        # Extract commands
        commands_match = re.search(r"<commands>(.*?)</commands>", content, re.DOTALL)
        if not commands_match:
            if is_complete:
                return ParseResult([], True)
            return ParseResult([], False, "Missing <commands> section")
        
        commands_content = commands_match.group(1)
        
        # Parse keystrokes
        commands = []
        keystrokes_pattern = re.compile(r'<keystrokes([^>]*)>(.*?)</keystrokes>', re.DOTALL)
        
        for i, match in enumerate(keystrokes_pattern.finditer(commands_content)):
            attrs_str = match.group(1)
            keystrokes = match.group(2)
            
            # Parse duration attribute
            duration_match = re.search(r'duration\s*=\s*["\']([^"\']*)["\']', attrs_str)
            if duration_match:
                try:
                    duration = float(duration_match.group(1))
                except ValueError:
                    warnings.append(f"Command {i+1}: invalid duration, using default 1.0")
                    duration = 1.0
            else:
                warnings.append(f"Command {i+1}: missing duration, using default 1.0")
                duration = 1.0
            
            commands.append(Command(keystrokes=keystrokes, duration_sec=duration))
        
        warning_str = "- " + "\n- ".join(warnings) if warnings else ""
        return ParseResult(commands, is_complete, "", warning_str)


# =============================================================================
# Docker Environment (Abstract Base)
# =============================================================================

class DockerEnvironmentBase:
    """Base class for Docker environment implementations."""
    
    def __init__(
        self,
        image_name: str,
        container_name: str,
        work_dir: Optional[str] = "/root",
        logs_dir: Optional[Path] = None,
    ):
        self.image_name = image_name
        self.container_name = container_name
        self.work_dir = work_dir
        self.logs_dir = logs_dir
    
    def start(self) -> None:
        """Start the Docker container."""
        raise NotImplementedError
    
    def stop(self) -> None:
        """Stop and remove the container."""
        raise NotImplementedError
    
    def exec(self, command: str, timeout_sec: float = 900) -> Tuple[int, str, str]:
        """Execute a command in the container."""
        raise NotImplementedError
    
    def copy_to_container(self, local_path: Path, container_path: str) -> None:
        """Copy a file or directory to the container."""
        raise NotImplementedError

    def was_oom_killed(self) -> bool:
        """Return whether Docker recorded an OOM kill during this run."""
        raise NotImplementedError


# =============================================================================
# Docker Environment - CLI Backend (Default)
# =============================================================================

class DockerEnvironmentCLI(DockerEnvironmentBase):
    """Docker environment using docker CLI commands (default backend)."""
    
    def __init__(
        self,
        image_name: str,
        container_name: str,
        work_dir: Optional[str] = "/root",
        logs_dir: Optional[Path] = None,
        max_parallel: int = 4,  # 用于计算 Docker 资源限制
    ):
        super().__init__(image_name, container_name, work_dir, logs_dir)
        self._started = False
        self._max_parallel = max_parallel
    
    def _run_docker_cmd(
        self, 
        args: List[str], 
        timeout: Optional[float] = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a docker command."""
        cmd = ["docker"] + args
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace undecodable bytes with U+FFFD instead of crashing
            timeout=timeout,
            check=check,
        )
    
    def start(self) -> None:
        """Start the Docker container."""
        # Remove existing container if it exists
        self._run_docker_cmd(["rm", "-f", self.container_name], timeout=30)
        
        # Start new container with resource limits
        # 资源限制防止服务器过载导致无法连接（根据并发数动态计算）
        run_args = [
            "run",
            "--name", self.container_name,
            "-d",  # detached
            "-t",  # allocate pseudo-TTY
            # Override stale ENV values in cached task images. Tencent's PyPI
            # mirror returned several hash-mismatched wheels in this run.
            "-e", "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
            "-e", "PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn",
        ]
        if self.work_dir:
            run_args.extend(["-w", self.work_dir])
        run_args.extend([
            *get_docker_resource_args(self._max_parallel),  # 添加 CPU 和内存限制
            self.image_name,
            "sleep", "infinity",
        ])
        result = self._run_docker_cmd(run_args, timeout=60)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        
        self._started = True
        
        # Wait for container to be ready
        time.sleep(1)
    
    def stop(self) -> None:
        """Stop and remove the container."""
        if self._started:
            self._run_docker_cmd(["rm", "-f", self.container_name], timeout=30)
            self._started = False
    
    def exec(self, command: str, timeout_sec: float = 900) -> Tuple[int, str, str]:
        """
        Execute a command in the container.
        
        Returns:
            Tuple of (return_code, stdout, stderr)
            On timeout: returns (124, "", "TIMEOUT: command exceeded {timeout_sec}s ...")
        """
        if not self._started:
            raise RuntimeError("Container not started")

        # Enforce the deadline *inside* the container.  A timeout applied only
        # to the host-side ``docker exec`` client leaves the command running in
        # the container (for example, a stale pip download), so a later agent
        # retry can race it.  GNU timeout owns a process group and terminates
        # the whole command tree; the host deadline is only a fallback.
        inner_timeout_sec = max(1, int(timeout_sec))
        wrapped_command = (
            f"timeout --signal=TERM --kill-after=5s {inner_timeout_sec}s "
            f"bash -c {shlex.quote(command)}"
        )

        try:
            exec_args = ["exec"]
            if self.work_dir:
                exec_args.extend(["-w", self.work_dir])
            exec_args.extend([self.container_name, "bash", "-c", wrapped_command])
            result = self._run_docker_cmd(exec_args, timeout=timeout_sec + 15)
            return result.returncode, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired:
            # Don't crash — return timeout info so the LLM can adapt.
            # Use exit code 124 (same as GNU timeout convention).
            cmd_preview = command[:200]
            timeout_msg = (
                f"TIMEOUT: Command exceeded {timeout_sec:.0f}s time limit and was killed.\n"
                f"Command: {cmd_preview}\n"
                f"Please adjust your plan accordingly."
            )
            return 124, "", timeout_msg
    
    def copy_to_container(self, local_path: Path, container_path: str) -> None:
        """Copy a file or directory to the container."""
        if not self._started:
            raise RuntimeError("Container not started")
        
        # Ensure the destination directory exists
        self.exec(f"mkdir -p '{container_path}'")
        
        # Use docker cp to copy files
        # docker cp copies the contents when source ends with /. 
        # Otherwise it copies the directory itself
        src = str(local_path)
        if local_path.is_dir():
            src = f"{src}/."
        
        result = self._run_docker_cmd(
            ["cp", src, f"{self.container_name}:{container_path}"],
            timeout=120,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to copy to container: {result.stderr}")

    def was_oom_killed(self) -> bool:
        """Check Docker's sticky per-container OOMKilled state."""
        if not self._started:
            return False
        result = self._run_docker_cmd(
            ["inspect", "--format", "{{.State.OOMKilled}}", self.container_name],
            timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"


# =============================================================================
# Docker Environment - SDK Backend (Optional)
# =============================================================================

class DockerEnvironmentSDK(DockerEnvironmentBase):
    """Docker environment using Docker SDK (requires docker package)."""
    
    def __init__(
        self,
        image_name: str,
        container_name: str,
        work_dir: Optional[str] = "/root",
        logs_dir: Optional[Path] = None,
    ):
        if docker is None:
            raise RuntimeError("Docker SDK not installed. Run: pip install docker")
        
        super().__init__(image_name, container_name, work_dir, logs_dir)
        self.container: Optional[Container] = None
        self.client = docker.from_env()
    
    def start(self) -> None:
        """Start the Docker container."""
        # Remove existing container if it exists
        try:
            existing = self.client.containers.get(self.container_name)
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass
        
        # Start new container
        run_kwargs = {
            "name": self.container_name,
            "detach": True,
            "tty": True,
            "command": "sleep infinity",
            # Keep the SDK backend consistent with the CLI backend, including
            # when an already-built task image contains the old mirror ENV.
            "environment": {
                "PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple",
                "PIP_TRUSTED_HOST": "pypi.tuna.tsinghua.edu.cn",
            },
        }
        if self.work_dir:
            run_kwargs["working_dir"] = self.work_dir
        self.container = self.client.containers.run(self.image_name, **run_kwargs)
        
        # Wait for container to be ready
        time.sleep(1)
    
    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def was_oom_killed(self) -> bool:
        """Check Docker's sticky per-container OOMKilled state."""
        if self.container is None:
            return False
        self.container.reload()
        return bool((self.container.attrs.get("State") or {}).get("OOMKilled", False))
    
    def exec(self, command: str, timeout_sec: float = 900) -> Tuple[int, str, str]:
        """
        Execute a command in the container.
        
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if not self.container:
            raise RuntimeError("Container not started")
        
        # As with the CLI backend, enforce the deadline inside the container so
        # a timed-out SDK call cannot leave a detached child process running.
        inner_timeout_sec = max(1, int(timeout_sec))
        wrapped_command = (
            f"timeout --signal=TERM --kill-after=5s {inner_timeout_sec}s "
            f"bash -c {shlex.quote(command)}"
        )

        # Execute command
        exec_kwargs = {"demux": True}  # Separate stdout/stderr
        if self.work_dir:
            exec_kwargs["workdir"] = self.work_dir
        result = self.container.exec_run(["bash", "-c", wrapped_command], **exec_kwargs)
        
        exit_code = result.exit_code
        stdout = result.output[0].decode("utf-8", errors="replace") if result.output[0] else ""
        stderr = result.output[1].decode("utf-8", errors="replace") if result.output[1] else ""
        
        return exit_code, stdout, stderr
    
    def copy_to_container(self, local_path: Path, container_path: str) -> None:
        """Copy a file or directory to the container."""
        if not self.container:
            raise RuntimeError("Container not started")
        
        import tarfile
        import io
        
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if local_path.is_file():
                tar.add(local_path, arcname=local_path.name)
            else:
                for item in local_path.rglob("*"):
                    tar.add(item, arcname=item.relative_to(local_path))
        
        tar_stream.seek(0)
        self.container.put_archive(container_path, tar_stream.read())


# =============================================================================
# Docker Environment Factory
# =============================================================================

def DockerEnvironment(
    image_name: str,
    container_name: str,
    work_dir: Optional[str] = "/root",
    logs_dir: Optional[Path] = None,
    backend: str = "cli",  # "cli" or "sdk"
    max_parallel: int = 4,  # 用于计算 Docker 资源限制
) -> DockerEnvironmentBase:
    """
    Factory function to create a Docker environment.
    
    Args:
        image_name: Docker image name
        container_name: Name for the container
        work_dir: Working directory inside container
        logs_dir: Optional directory for logs
        backend: "cli" (default) or "sdk"
        max_parallel: 最大并发容器数，用于计算每个容器的资源配额
    
    Returns:
        DockerEnvironmentBase instance
    """
    if backend == "sdk":
        return DockerEnvironmentSDK(
            image_name=image_name,
            container_name=container_name,
            work_dir=work_dir,
            logs_dir=logs_dir,
        )
    else:  # default to CLI
        return DockerEnvironmentCLI(
            image_name=image_name,
            container_name=container_name,
            work_dir=work_dir,
            logs_dir=logs_dir,
            max_parallel=max_parallel,
        )


# =============================================================================
# Skill Loader
# =============================================================================

class SkillLoader:
    """
    Loads skills from the container filesystem following harbor's Agent Skills spec.
    
    Skills are discovered by scanning skill_dirs for subdirectories containing SKILL.md.
    The SKILL.md file must have YAML frontmatter with 'name' and 'description' fields.
    """
    
    def __init__(self, environment: DockerEnvironmentBase):
        self.env = environment
        self._metadata: List[SkillMetadata] = []
    
    def build_index(self, skill_dirs: Optional[List[str]] = None) -> None:
        """
        Build an index of available skills by scanning skill directories.
        
        Following harbor's approach:
        - Uses 'find' to locate SKILL.md files
        - Parses YAML frontmatter for name and description
        - Records the full path to the SKILL.md file (for cat-to-load pattern)
        
        Optimized: reads all SKILL.md frontmatters in a single docker exec
        using find + awk to extract only the YAML frontmatter (between --- markers).
        
        Args:
            skill_dirs: List of directories to scan for skills
        """
        if not skill_dirs:
            skill_dirs = []
        
        self._metadata = []
        
        for skill_dir in skill_dirs:
            # Single docker exec: find all SKILL.md files and extract their
            # YAML frontmatter (content between the first two --- lines).
            # Uses awk to precisely extract only the frontmatter, avoiding
            # issues with head -N truncating long frontmatters.
            cmd = (
                f"find '{skill_dir}' -mindepth 2 -maxdepth 2 -name SKILL.md -type f 2>/dev/null | sort | "
                f"while IFS= read -r f; do "
                f"  echo \"===PATH:$f===\"; "
                f"  awk 'NR==1 && /^---/{{p=1; print; next}} p && /^---/{{print; exit}} p{{print}}' \"$f\" 2>/dev/null; "
                f"done"
            )
            
            exit_code, batch_output, _ = self.env.exec(cmd, timeout_sec=120)
            
            if not batch_output or not batch_output.strip():
                continue
            
            # Parse the batch output
            current_path = None
            current_lines: List[str] = []
            
            for line in batch_output.split("\n"):
                if line.startswith("===PATH:") and line.endswith("==="):
                    # Process previous skill
                    if current_path and current_lines:
                        self._parse_and_add(current_path, "\n".join(current_lines))
                    # Start new skill
                    current_path = line[8:-3]  # strip ===PATH: and ===
                    current_lines = []
                else:
                    current_lines.append(line)
            
            # Don't forget the last one
            if current_path and current_lines:
                self._parse_and_add(current_path, "\n".join(current_lines))
    
    def _parse_and_add(self, skill_md_path: str, content: str) -> None:
        """Parse frontmatter from content and add to metadata if valid."""
        fm = self._parse_frontmatter(content)
        if fm is not None:
            self._metadata.append(SkillMetadata(
                name=fm["name"],
                description=fm["description"],
                location=skill_md_path,
            ))
    
    def build_skills_section(self) -> Optional[str]:
        """
        Build a JSON skills list for the prompt.
        
        Returns JSON format like:
        available_skills:
        [
          {
            "name": "mesh-analysis",
            "description": "Analyzes 3D mesh files...",
            "location": "/root/.terminus/skills/mesh-analysis/SKILL.md"
          }
        ]
        
        Returns:
            JSON string if skills are available, None otherwise
        """
        if not self._metadata:
            return None
        
        skills_list = [
            {
                "name": skill.name,
                "description": skill.description,
                "location": skill.location
            }
            for skill in self._metadata
        ]
        
        return "available_skills:\n" + json.dumps(skills_list, indent=2)
    
    def load_skill(self, name: str) -> Optional[str]:
        """Load the full content of a skill by name."""
        for skill in self._metadata:
            if skill.name == name:
                exit_code, content, _ = self.env.exec(
                    f"cat '{skill.location}' 2>/dev/null",
                    timeout_sec=10
                )
                if exit_code == 0:
                    return content
        return None
    
    def load_skill_by_path(self, path: str) -> Optional[str]:
        """Load skill content by full path (for cat-to-load pattern)."""
        exit_code, content, _ = self.env.exec(
            f"cat '{path}' 2>/dev/null",
            timeout_sec=10
        )
        if exit_code == 0:
            return content
        return None
    
    def get_metadata(self) -> List[SkillMetadata]:
        """Get the list of skill metadata."""
        return self._metadata.copy()
    
    def _parse_frontmatter(self, content: str) -> Optional[Dict[str, str]]:
        """
        Parse YAML frontmatter from SKILL.md content.
        
        Returns dict with 'name' and 'description' fields, or None if parsing fails.
        """
        lines = content.strip().split("\n")
        if not lines or lines[0].strip() != "---":
            return None
        
        # Find the closing ---
        frontmatter_end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                frontmatter_end = i
                break
        
        if frontmatter_end == -1:
            return None
        
        # Parse the frontmatter (simple YAML parsing with multiline values)
        fm_lines = lines[1:frontmatter_end]
        result = {}
        current_key = None
        
        for line in fm_lines:
            if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value and value != "|":
                    result[key] = value
                    current_key = key
                else:
                    result[key] = ""
                    current_key = key
            elif current_key and (line.startswith("  ") or line.startswith("\t")):
                result[current_key] += " " + line.strip()
        
        # Validate required fields
        if "name" not in result or "description" not in result:
            return None
        
        return {"name": result["name"], "description": result["description"]}


class DynamicRAGSkillSearcher:
    """Runtime skill search tool backed by precomputed skill embeddings."""

    def __init__(self, config: AgentConfig, top_k: int, logger: logging.Logger):
        self.config = config
        self.top_k = top_k
        self.logger = logger
        self.rag_dir = Path(config.rag_precomputed_dir) if config.rag_precomputed_dir else None
        self.embedding_model = config.preselect_embedding_model or DEFAULT_QWEN_EMBEDDING_MODEL
        self.embedding_api_base = (
            os.environ.get("EMBEDDING_API_BASE")
            or os.environ.get("DASHSCOPE_API_BASE")
            or os.environ.get("QWEN_BASE_URL")
            or DEFAULT_QWEN_EMBEDDING_API_BASE
        ).rstrip("/")
        self.embedding_api_key = (
            os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or DEFAULT_QWEN_EMBEDDING_API_KEY
        )
        self._cache_vectors: Dict[str, Any] = {}
        self._skill_vectors: List[Tuple[SkillMetadata, Any]] = []
        self._load_precomputed_cache()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _load_precomputed_cache(self) -> None:
        if not self.rag_dir:
            self.logger.warning("[SkillSearch/dynamic-rag] No rag_precomputed_dir configured")
            return

        index_file = self.rag_dir / "cache" / "embedding_index.json"
        cache_file = self.rag_dir / "cache" / "embedding_cache.npz"
        metadata_file = self.rag_dir / "index.json"

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f).get("metadata", {})
                model = metadata.get("embedding_model")
                if model:
                    self.embedding_model = model
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"[SkillSearch/dynamic-rag] Failed to read metadata: {e}")

        if np is None:
            self.logger.warning("[SkillSearch/dynamic-rag] numpy is unavailable; cannot load npz embedding cache")
            return
        if not index_file.exists() or not cache_file.exists():
            self.logger.warning(
                f"[SkillSearch/dynamic-rag] Precomputed embedding cache not found under {self.rag_dir}"
            )
            return

        try:
            with open(index_file) as f:
                index = json.load(f)
            data = np.load(cache_file)
            for text_hash, array_key in index.items():
                if array_key in data:
                    self._cache_vectors[text_hash] = data[array_key]
            self.logger.info(
                f"[SkillSearch/dynamic-rag] Loaded {len(self._cache_vectors)} precomputed embeddings "
                f"from {cache_file}"
            )
        except Exception as e:
            self.logger.warning(f"[SkillSearch/dynamic-rag] Failed to load embedding cache: {e}")
            self._cache_vectors = {}

    def set_skills(self, skills: List[SkillMetadata]) -> None:
        self._skill_vectors = []
        missing = 0
        for skill in skills:
            text = f"{skill.name.strip()}: {skill.description.strip()}"
            vector = self._cache_vectors.get(self._text_hash(text))
            if vector is None:
                missing += 1
                continue
            self._skill_vectors.append((skill, vector))
        self.logger.info(
            f"[SkillSearch/dynamic-rag] Indexed {len(self._skill_vectors)}/{len(skills)} candidate skills "
            f"from precomputed embeddings (missing={missing})"
        )

    def search(self, query: str) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if not self._skill_vectors:
            self.logger.warning("[SkillSearch/dynamic-rag] No skill vectors available for search")
            return []

        query_vector = self._cache_vectors.get(self._text_hash(query))
        if query_vector is None:
            query_vector = self._embed_query(query)
        if query_vector is None:
            return []

        scored = []
        for skill, vector in self._skill_vectors:
            scored.append((self._cosine_similarity(query_vector, vector), skill))
        scored.sort(key=lambda item: item[0], reverse=True)

        results = []
        for rank, (score, skill) in enumerate(scored[: self.top_k], 1):
            results.append({
                "name": skill.name,
                "description": skill.description,
                "location": skill.location,
                "similarity": round(float(score), 6),
                "rank": rank,
            })
        self.logger.info(
            f"[SkillSearch/dynamic-rag] Query={query!r} -> {[r['name'] for r in results]}"
        )
        return results

    def _embed_query(self, query: str) -> Optional[Any]:
        url = f"{self.embedding_api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.embedding_api_key}",
        }
        payload = {"model": self.embedding_model, "input": [query]}

        for attempt in range(1, 4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 10 * attempt
                    self.logger.warning(
                        f"[SkillSearch/dynamic-rag] Embedding HTTP {resp.status_code} "
                        f"(attempt {attempt}/3), retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    self.logger.error(
                        f"[SkillSearch/dynamic-rag] Embedding API error {resp.status_code}: {resp.text[:500]}"
                    )
                    return None
                data = resp.json()
                item = sorted(data["data"], key=lambda x: x.get("index", 0))[0]
                vector = item["embedding"]
                if np is not None:
                    vector = np.array(vector, dtype=np.float32)
                return vector
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.logger.warning(
                    f"[SkillSearch/dynamic-rag] Embedding connection error (attempt {attempt}/3): {e}"
                )
                if attempt < 3:
                    time.sleep(10)
        return None

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        if np is not None:
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# =============================================================================
# Prompt Templates
# =============================================================================

JSON_PROMPT_TEMPLATE = '''You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim. Most bash commands should end with a newline (\\n) to execute.

---
SKILL SYSTEM:
If skills are available (shown in "available_skills"), you MUST check if any skill matches the task BEFORE executing commands.

Skill context:
{skills_block}

To load a skill, respond with ONLY this JSON (no other fields):
{{"load_skill": "skill-name"}}

After a skill is loaded, proceed with the normal response format above.
Skills contain best-practice workflows - use them instead of writing custom solutions.
---

Task Description:
{instruction}

Current terminal state:
{terminal_state}
'''


# =============================================================================
# Main Agent
# =============================================================================

class StandaloneAgent:
    """
    Standalone agent for executing SkillsBench tasks.
    
    This agent is independent of harbor and litellm.
    """
    
    def __init__(self, config: AgentConfig, logs_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.logs_dir = logs_dir
        self.logger = logger or logging.getLogger("experiment.agent")
        
        # Initialize components — share the same logger so LLM logs
        # go to the same task-level log file. TokenPlan is the default because
        # it follows OpenClaw's OpenAI-compatible `openai-completions` provider.
        provider = (config.llm_provider or "tokenplan").lower().strip()
        if provider in {"tokenplan", "tencent-token-plan", "tencent_tokenplan"}:
            self.llm = TokenPlanLLMClient(config, logger=self.logger)
        else:
            self.llm = SimpleLLMClient(config, logger=self.logger)
        self.parser = JSONResponseParser() if config.response_format == "json" else XMLResponseParser()
        self.prompt_template = JSON_PROMPT_TEMPLATE  # TODO: Add XML template
        
        # Context window management
        self.context_window = get_context_window(config.model_name)
        self.logger.info(f"[Agent] Model: {config.model_name}, context window: {self.context_window:,} tokens")
        
        # Initialize skill preselector
        self.preselector = create_preselector(
            strategy=config.preselect_strategy,
            max_skills=config.preselect_max_skills,
            model_name=config.preselect_model_name or config.model_name,
            api_base=config.api_base,
            api_key=config.api_key,
            embedding_model=config.preselect_embedding_model,
            temperature=0.0,  # Use deterministic selection
            logger=self.logger,
            rag_precomputed_dir=config.rag_precomputed_dir,
            hybrid_rag_retrieval_k=config.hybrid_rag_retrieval_k,
            preselect_log_dir=(self.logs_dir / "preselect_llm_io") if self.logs_dir else None,
        )
        self.logger.info(f"[Agent] Skill preselector: strategy={config.preselect_strategy}, "
                         f"max_skills={config.preselect_max_skills}"
                         f"{', precomputed=' + config.rag_precomputed_dir if config.rag_precomputed_dir else ''}")
        
        # Runtime RAG skill searcher for dynamic-rag mode.
        self._dynamic_skill_searcher: Optional[DynamicRAGSkillSearcher] = None
        if self._is_dynamic_rag_mode():
            self._dynamic_skill_searcher = DynamicRAGSkillSearcher(
                config=config,
                top_k=config.preselect_max_skills,
                logger=self.logger,
            )
        
        # State
        self._loaded_skills: Dict[str, str] = {}
        self._skills_metadata: List[SkillMetadata] = []
        # Tracks skills loaded during the current task (accessible from outside
        # even if run_task_sync raises an exception mid-way through).
        self._loaded_skills_log: List[Dict[str, Any]] = []
        
        # Pre-selection state (for result tracking)
        self._preselected_skills: List[str] = []
        self._preselect_input_tokens: int = 0
        self._preselect_output_tokens: int = 0
    
    def _is_dynamic_rag_mode(self) -> bool:
        return self.config.preselect_strategy.lower().strip() in {
            "dynamic-rag",
            "dynamic_rag",
            "rag-tool",
            "rag_tool",
        }
    
    async def run_task(
        self,
        instruction: str,
        environment: DockerEnvironmentBase,
        skill_dirs: Optional[List[str]] = None,
    ) -> AgentResult:
        """
        Run a task in the given environment (async wrapper).
        
        Args:
            instruction: The task instruction
            environment: The Docker environment to execute in
            skill_dirs: Optional list of skill directories in the container
        
        Returns:
            AgentResult with execution details
        """
        return self.run_task_sync(instruction, environment, skill_dirs)
    
    def run_task_sync(
        self,
        instruction: str,
        environment: DockerEnvironmentBase,
        skill_dirs: Optional[List[str]] = None,
        task_name: Optional[str] = None,
        extra_skills_count: Optional[int] = None,
    ) -> AgentResult:
        """
        Run a task in the given environment (synchronous).
        
        This is the actual implementation. All operations (LLM calls via
        requests.post, docker exec via subprocess) are synchronous.
        Use this directly when running inside a worker thread.
        
        Args:
            instruction: The task instruction
            environment: The Docker environment to execute in
            skill_dirs: Optional list of skill directories in the container
            task_name: Task name (used by PrecomputedRAGPreselector to locate results)
            extra_skills_count: Number of extra skills (used by PrecomputedRAGPreselector)
        
        Returns:
            AgentResult with execution details
        """
        trajectory = []
        
        # Initialize skill loader and build index
        skill_loader = SkillLoader(environment)
        skill_loader.build_index(skill_dirs)  # Now returns None, populates internal state
        all_skills_metadata = skill_loader.get_metadata()
        self._skill_loader = skill_loader  # Save reference for skill loading
        
        self.logger.info(f"[Skills] Found {len(all_skills_metadata)} skills in container: {[s.name for s in all_skills_metadata]}")
        if self._dynamic_skill_searcher is not None:
            self._dynamic_skill_searcher.set_skills(all_skills_metadata)
        
        # ══════════════════════════════════════════════════════════════════════
        # SKILL PRE-SELECTION: Filter skills before adding to prompt
        # ══════════════════════════════════════════════════════════════════════
        # Convert SkillMetadata to SkillInfo for the preselector
        all_skill_infos = [
            SkillInfo(name=s.name, description=s.description, location=s.location)
            for s in all_skills_metadata
        ]
        
        # Set task context for selectors that read per-(task, skills_count) precomputed files.
        set_context = getattr(self.preselector, "set_context", None)
        if callable(set_context):
            if task_name is not None and extra_skills_count is not None:
                set_context(task_name, extra_skills_count)
            else:
                self.logger.warning(
                    f"[Preselect/{self.preselector.strategy_name}] task_name or extra_skills_count not provided; "
                    "preselector may fall back or select no skills"
                )
        
        # Run pre-selection (this may call LLM or compute embeddings)
        selected_skill_infos = self.preselector.select(instruction, all_skill_infos)
        
        # Track pre-selection results
        self._preselected_skills = [s.name for s in selected_skill_infos]
        if hasattr(self.preselector, 'total_input_tokens'):
            self._preselect_input_tokens = self.preselector.total_input_tokens
            self._preselect_output_tokens = self.preselector.total_output_tokens
        
        # Convert back to SkillMetadata (only selected ones)
        selected_names = set(s.name for s in selected_skill_infos)
        self._skills_metadata = [s for s in all_skills_metadata if s.name in selected_names]
        
        self.logger.info(f"[Preselect/{self.preselector.strategy_name}] Selected {len(self._skills_metadata)} skills "
                         f"for prompt: {[s.name for s in self._skills_metadata]}")
        
        # Build initial prompt with PRE-SELECTED skills section
        skills_block = self._build_skills_block(skill_loader, selected_only=True)
        
        # Get initial terminal state
        _, initial_output, _ = environment.exec("pwd && ls -la")
        terminal_state = f"Current Terminal Screen:\n{initial_output}"
        
        prompt = self.prompt_template.format(
            instruction=instruction,
            terminal_state=terminal_state,
            skills_block=skills_block,
        )
        
        # Message history for the LLM
        messages = [{"role": "user", "content": prompt}]
        
        # Track skills loaded during this task (in order)
        # Also update instance attribute so caller can access partial results
        # even if this method raises mid-execution.
        loaded_skills_log: List[Dict[str, Any]] = []
        self._loaded_skills_log = loaded_skills_log  # Same list object — mutations visible outside
        
        # Agent loop
        for episode in range(self.config.max_episodes):
            self.logger.info(f"[Episode {episode}] Starting...")
            
            # ── Context window management ──
            # Trim message history to fit within the model's context window.
            # Keeps the first message (system prompt) + last N turns.
            messages = trim_messages_to_fit(
                messages,
                max_tokens=self.context_window,
                max_context_turns=self.config.max_context_turns,
                logger=self.logger,
            )
            
            # Save episode data
            if self.logs_dir:
                episode_dir = self.logs_dir / f"episode-{episode}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                (episode_dir / "prompt.txt").write_text(messages[-1]["content"])
            
            # Call LLM (no try-except, let errors propagate)
            response = self.llm.call(messages)
            
            # Save response
            if self.logs_dir:
                (episode_dir / "response.txt").write_text(response)
            
            trajectory.append({
                "episode": episode,
                "prompt": messages[-1]["content"],
                "response": response,
            })
            
            # Check for skill loading request
            skill_output = self._handle_skill_request(response, skill_loader)
            if skill_output:
                # Record which skill was loaded and when
                skill_name = self._extract_skill_name(response)
                if skill_name:
                    loaded_skills_log.append({"episode": episode, "skill": skill_name})
                    self.logger.info(f"[Episode {episode}] Loaded skill: {skill_name}")
                # Add skill content to context and continue
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"{skill_output}\n\nCurrent Terminal Screen:\n"})
                continue
            
            # Parse response
            result = self.parser.parse_response(response)
            
            if result.error:
                self.logger.warning(f"[Episode {episode}] Parse error: {result.error}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Previous response had parsing errors:\n{result.error}\n\nPlease fix these issues and provide a proper JSON response."})
                continue
            
            # Execute commands
            terminal_output = self._execute_commands(result.commands, environment)
            self.logger.info(f"[Episode {episode}] Executed {len(result.commands)} commands")
            
            # Check if task is complete
            if result.is_task_complete:
                self.logger.info(f"[Episode {episode}] Task marked as complete!")
                if loaded_skills_log:
                    self.logger.info(f"[Skills] Loaded {len(loaded_skills_log)} skill(s) during task: "
                                     f"{[s['skill'] for s in loaded_skills_log]}")
                return AgentResult(
                    success=True,
                    episodes=episode + 1,
                    total_input_tokens=self.llm.total_input_tokens,
                    total_output_tokens=self.llm.total_output_tokens,
                    trajectory=trajectory,
                    loaded_skills=loaded_skills_log,
                    preselect_strategy=self.config.preselect_strategy,
                    preselected_skills=self._preselected_skills,
                    preselect_input_tokens=self._preselect_input_tokens,
                    preselect_output_tokens=self._preselect_output_tokens,
                )
            
            # Prepare next prompt
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": terminal_output})
        
        # Max episodes reached
        if loaded_skills_log:
            self.logger.info(f"[Skills] Loaded {len(loaded_skills_log)} skill(s) during task: "
                             f"{[s['skill'] for s in loaded_skills_log]}")
        return AgentResult(
            success=False,
            episodes=self.config.max_episodes,
            total_input_tokens=self.llm.total_input_tokens,
            total_output_tokens=self.llm.total_output_tokens,
            error="Max episodes reached",
            trajectory=trajectory,
            loaded_skills=loaded_skills_log,
            preselect_strategy=self.config.preselect_strategy,
            preselected_skills=self._preselected_skills,
            preselect_input_tokens=self._preselect_input_tokens,
            preselect_output_tokens=self._preselect_output_tokens,
        )
    
    def _build_skills_block(self, skill_loader: SkillLoader, selected_only: bool = False) -> str:
        """
        Build the skills block for the prompt.
        
        Following harbor's approach, this returns:
        1. XML format <available_skills> section if skills are available
        2. Any already loaded skill content
        3. Instructions for how to load and use skills
        
        Args:
            skill_loader: The SkillLoader instance
            selected_only: If True, only include pre-selected skills (self._skills_metadata)
                          If False, include all skills from skill_loader
        """
        if self._is_dynamic_rag_mode():
            return self._build_dynamic_skills_block()

        if selected_only:
            # Use only the pre-selected skills
            skills_metadata = self._skills_metadata
        else:
            # Use all skills from the loader
            skills_metadata = skill_loader.get_metadata()
        
        if not skills_metadata:
            return "No skills available."
        
        # Build JSON skills list manually (similar to skill_loader.build_skills_section)
        skills_list = [
            {
                "name": skill.name,
                "description": skill.description,
                "location": skill.location
            }
            for skill in skills_metadata
        ]
        skills_section = "available_skills:\n" + json.dumps(skills_list, indent=2)
        
        # Build the full skills block
        active_skills = self._format_loaded_skills()
        
        return f"""{skills_section}

To load a skill, use: cat <location> (where location is from the skill's <location> tag above)

LOADED SKILLS:
{active_skills}
"""
    
    def _build_dynamic_skills_block(self) -> str:
        active_skills = self._format_loaded_skills()
        return f"""No real skills are listed initially.

Example skill reference format only (placeholder; do NOT load unless it appears in search results):
available_skills:
[
  {{
    "name": "skill-name",
    "description": "Example placeholder showing how a skill will be listed after search.",
    "location": "/root/.terminus/skills/skill-name/SKILL.md"
  }}
]

Dynamic skill-search tool:
When you believe the current step needs domain-specific workflow guidance, search for relevant skills by responding with ONLY this JSON (no other fields):
{{"search_skills": "describe the specific capability, workflow, library, or domain guidance needed now"}}

The tool will embed your query and return the top {self.config.preselect_max_skills} matching skills from the precomputed skill embedding index. After that, load a returned skill with:
{{"load_skill": "skill-name"}}

LOADED SKILLS:
{active_skills}
"""
    
    def _format_loaded_skills(self) -> str:
        """Format loaded skills for display."""
        if not self._loaded_skills:
            return "No skills loaded."
        
        sections = []
        for name, content in self._loaded_skills.items():
            sections.append(f"Loaded skill: {name}\n---\n{content}")
        
        return "\n\n".join(sections)
    
    def _handle_skill_request(self, response: str, loader: SkillLoader) -> Optional[str]:
        """Handle skill search / loading request in response."""
        if "search_skills" in response:
            search_output = self._handle_skill_search_request(response)
            if search_output:
                return search_output

        if "load_skill" not in response:
            return None
        
        try:
            match = re.search(r'\{\s*"load_skill[^"]*"\s*:\s*"([^"]+)"\s*\}', response)
            if not match:
                return None
            
            skill_name = match.group(1).strip()
            content = loader.load_skill(skill_name)
            
            if content:
                self._loaded_skills[skill_name] = content
                return f"Loaded skill: {skill_name}\n---\n{content}"
            else:
                return f"Skill not found: {skill_name}"
        except Exception:
            return None
    
    def _handle_skill_search_request(self, response: str) -> Optional[str]:
        """Handle dynamic-rag search_skills tool request."""
        if self._dynamic_skill_searcher is None:
            return None
        try:
            match = re.search(r'\{\s*"search_skills[^"]*"\s*:\s*"([^"]+)"\s*\}', response, re.DOTALL)
            if not match:
                return None
            query = match.group(1).strip()
            results = self._dynamic_skill_searcher.search(query)
            if not results:
                return (
                    f"No matching skills found for query: {query}\n\n"
                    "You may try a more specific search_skills query or continue without skills."
                )
            return (
                f"Skill search results for query: {query}\n"
                + "available_skills:\n"
                + json.dumps(results, indent=2, ensure_ascii=False)
                + "\n\nTo load one of these skills, respond with ONLY: "
                + '{"load_skill": "exact-skill-name"}'
            )
        except Exception as e:
            self.logger.warning(f"[SkillSearch/dynamic-rag] Failed to handle search request: {e}")
            return None
    
    def _extract_skill_name(self, response: str) -> Optional[str]:
        """Extract skill name from a load_skill request in the response."""
        try:
            match = re.search(r'\{\s*"load_skill[^"]*"\s*:\s*"([^"]+)"\s*\}', response)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        return None
    
    def _execute_commands(self, commands: List[Command], env: DockerEnvironmentBase) -> str:
        """Execute commands and return terminal output."""
        outputs = []
        
        for i, cmd in enumerate(commands):
            keystrokes = cmd.keystrokes.strip()
            if not keystrokes:
                continue
            
            # Log the command being executed (truncate long commands)
            cmd_preview = keystrokes.replace('\n', '\\n')[:120]
            self.logger.info(f"[Exec {i+1}/{len(commands)}] Running: {cmd_preview}")
            
            # Execute command
            cmd_start = time.time()
            exit_code, stdout, stderr = env.exec(keystrokes)
            cmd_elapsed = time.time() - cmd_start
            
            # Log execution result
            out_len = len(stdout or "") + len(stderr or "")
            if cmd_elapsed > 10:
                self.logger.warning(f"[Exec {i+1}/{len(commands)}] Completed in {cmd_elapsed:.1f}s "
                                    f"(exit={exit_code}, output={out_len} chars)")
            else:
                self.logger.info(f"[Exec {i+1}/{len(commands)}] Completed in {cmd_elapsed:.1f}s "
                                 f"(exit={exit_code}, output={out_len} chars)")
            
            combined = "\n".join(part for part in [stdout, stderr] if part).strip()
            if combined:
                # Filter shell warnings
                lines = [
                    line for line in combined.split("\n")
                    if not line.startswith("bash:") and "no job control" not in line
                ]
                filtered = "\n".join(lines).strip()
                if filtered:
                    outputs.append(filtered)
            
            # Note: No additional sleep needed — docker exec is synchronous,
            # the command has already completed by this point.
            # (duration was originally for harbor's terminal keystroke mode)
        
        if not outputs:
            return "Current Terminal Screen:\n"

        output_prefix = "New Terminal Output:\n"
        combined_output = "\n\n".join(outputs)
        max_output_bytes = max(0, int(self.config.max_output_bytes))
        encoded_output = combined_output.encode("utf-8", errors="replace")
        prefix_bytes = output_prefix.encode("utf-8")
        content_budget = max(0, max_output_bytes - len(prefix_bytes))
        if max_output_bytes and len(encoded_output) > content_budget:
            # The omitted byte count affects the marker length, so converge on
            # the exact retained-payload budget before slicing the output.
            marker = b""
            for _ in range(3):
                payload_budget = max(0, content_budget - len(marker))
                omitted_bytes = len(encoded_output) - payload_budget
                marker = (
                    f"\n\n...[terminal output truncated: {omitted_bytes} bytes omitted; "
                    f"original size {len(encoded_output)} bytes]...\n\n"
                ).encode("utf-8")
            if len(marker) > content_budget:
                marker = marker[:content_budget]
            payload_budget = max(0, content_budget - len(marker))
            head_budget = payload_budget * 2 // 3
            tail_budget = payload_budget - head_budget
            head = encoded_output[:head_budget].decode("utf-8", errors="ignore")
            tail = (
                encoded_output[-tail_budget:].decode("utf-8", errors="ignore")
                if tail_budget
                else ""
            )
            combined_output = head + marker.decode("utf-8") + tail
            self.logger.warning(
                "[Exec] Terminal output exceeded %d bytes and was truncated "
                "before being returned to the model (original=%d bytes)",
                max_output_bytes,
                len(encoded_output),
            )

        return output_prefix + combined_output


# =============================================================================
# Utility Functions
# =============================================================================

async def run_single_task(
    task_dir: Path,
    config: AgentConfig,
    output_dir: Optional[Path] = None,
) -> AgentResult:
    """
    Run a single task.
    
    Args:
        task_dir: Path to task directory (contains instruction.md, environment/, etc.)
        config: Agent configuration
        output_dir: Optional output directory for logs
    
    Returns:
        AgentResult
    """
    # Read instruction
    instruction_file = task_dir / "instruction.md"
    if not instruction_file.exists():
        raise FileNotFoundError(f"No instruction.md found in {task_dir}")
    
    instruction = instruction_file.read_text()
    
    # Setup logs directory
    logs_dir = None
    if output_dir:
        logs_dir = output_dir / task_dir.name
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create agent
    agent = StandaloneAgent(config, logs_dir)
    
    # Setup Docker environment
    # For now, use a simple approach - build from Dockerfile if present, else use base image
    image_name = "ubuntu:22.04"  # Default base image
    dockerfile = task_dir / "environment" / "Dockerfile"
    
    if dockerfile.exists():
        # Build the image
        logging.info(f"Building Docker image from {dockerfile}...")
        image_name = f"skillsbench_{task_dir.name}".lower().replace("-", "_")
        subprocess.run(
            ["docker", "build", "-t", image_name, str(task_dir / "environment")],
            check=True,
        )
    
    container_name = f"agent_{task_dir.name}_{int(time.time())}"
    env = DockerEnvironment(
        image_name=image_name,
        container_name=container_name,
        logs_dir=logs_dir,
    )
    
    try:
        # Start environment
        logging.info(f"Starting Docker container: {container_name}")
        env.start()
        
        # Copy task environment files if present
        env_dir = task_dir / "environment"
        if env_dir.exists():
            # Copy files (excluding Dockerfile)
            for item in env_dir.iterdir():
                if item.name != "Dockerfile" and item.name != "docker-compose.yml":
                    env.copy_to_container(item, "/root/")
        
        # Copy skills if present
        skills_dir = task_dir / "environment" / "skills"
        if skills_dir.exists():
            env.copy_to_container(skills_dir, "/root/.terminus/")
        
        # Run the task
        result = await agent.run_task(
            instruction=instruction,
            environment=env,
            skill_dirs=["/root/.terminus/skills"],
        )
        
        return result
        
    finally:
        # Cleanup
        logging.info(f"Stopping container: {container_name}")
        env.stop()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    # Set up basic logging for standalone mode
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Standalone Agent for SkillsBench tasks")
    parser.add_argument("task_dir", type=Path, help="Path to task directory")
    parser.add_argument("--model", default="glm-5", help="Model name")
    parser.add_argument("--api-base", default=get_default_tokenplan_api_base(), help="API base URL (defaults to Tencent TokenPlan)")
    parser.add_argument("--api-key", default=get_default_tokenplan_api_key(), help="API key (defaults to Tencent TokenPlan env/.env)")
    parser.add_argument("--llm-provider", default="tokenplan", choices=["tokenplan", "openai-compatible"], help="LLM provider client")
    parser.add_argument("--output-dir", type=Path, help="Output directory for logs")
    parser.add_argument("--max-episodes", type=int, default=50, help="Max episodes")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (default: model default)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (default: model default)")
    
    args = parser.parse_args()
    
    config = AgentConfig(
        model_name=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        llm_provider=args.llm_provider,
        temperature=args.temperature,
        top_p=args.top_p,
        max_episodes=args.max_episodes,
    )
    
    result = asyncio.run(run_single_task(
        task_dir=args.task_dir,
        config=config,
        output_dir=args.output_dir,
    ))
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Episodes: {result.episodes}")
    print(f"Input tokens: {result.total_input_tokens}")
    print(f"Output tokens: {result.total_output_tokens}")
    if result.error:
        print(f"Error: {result.error}")
