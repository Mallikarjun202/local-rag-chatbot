import logging
import time
from typing import Dict, List, Optional, Tuple

from . import config

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MAX_OUTPUT_TOKENS = 4096
# Thinking uses part of the output budget, so give it more room.
THINKING_OUTPUT_TOKENS = 16384

# model_name_prefix -> provider key. Checked in order, so the more specific
# Groq-hosted "openai/" prefix must come before OpenAI's own "gpt-".
_PROVIDER_PREFIXES = {
    "openai/": "groq",
    "llama-": "groq",
    "qwen": "groq",
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
}

# Google AI Studio's free tier and Groq's free tier don't require a billing
# account, unlike OpenAI/Anthropic. Model names checked October 2026.
MODEL_CATALOG = {
    "google": ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "openai": ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"],
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
}

# Free providers first, so the fallback default model doesn't need paid credits.
# Groq first: in testing it answered in ~0.5 s vs 2-48 s for Gemini's free tier.
_PROVIDER_ORDER = ["groq", "google", "openai", "anthropic"]


log = logging.getLogger("chatbot")

_PLACEHOLDER_HINTS = ("your", "here", "xxx", "changeme", "replace", "<", ">")


def _real_key(value: Optional[str]) -> Optional[str]:
    """Treat empty or template values like 'put-your-key-here' as not set."""
    if not value:
        return None
    v = value.strip().strip('"').strip("'")
    if len(v) < 20 or any(h in v.lower() for h in _PLACEHOLDER_HINTS):
        return None
    return v


def _provider_keys() -> Dict[str, Optional[str]]:
    return {
        "openai": _real_key(config.OPENAI_API_KEY),
        "anthropic": _real_key(config.ANTHROPIC_API_KEY),
        "google": _real_key(config.GOOGLE_API_KEY),
        "groq": _real_key(config.GROQ_API_KEY),
    }


def get_available_models() -> List[Dict[str, str]]:
    """Only returns models for providers whose API key is actually set."""
    keys = _provider_keys()
    return [{"id": m, "provider": p}
            for p in _PROVIDER_ORDER if keys[p]
            for m in MODEL_CATALOG[p]]


def get_default_model() -> Optional[str]:
    """DEFAULT_MODEL from .env if it's usable, otherwise the first available model."""
    available = [m["id"] for m in get_available_models()]
    if config.DEFAULT_MODEL and config.DEFAULT_MODEL in available:
        return config.DEFAULT_MODEL
    return available[0] if available else None


def _provider_for_model(model: str) -> str:
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    raise ValueError(f"Could not determine provider for model '{model}'")


def chat_completion(model: str, messages: List[Dict[str, str]], system_prompt: Optional[str] = None,
                    thinking: bool = False) -> Tuple[str, Optional[str]]:
    """Returns (answer_text, reasoning_text_or_None).

    thinking=True asks the model to reason more deeply (slower, better on hard
    questions) and returns its reasoning summary when the provider exposes one.
    """
    provider = _provider_for_model(model)
    if not _provider_keys()[provider]:
        raise ValueError(f"{provider} API key is not set")
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    handlers = {"openai": _openai_chat, "anthropic": _anthropic_chat,
                "google": _google_chat, "groq": _groq_chat}
    if provider not in handlers:
        raise ValueError(f"Unsupported provider '{provider}'")
    out = handlers[provider](model=model, messages=messages, system_prompt=system_prompt, thinking=thinking)
    if isinstance(out, tuple):
        return out[0] or "", (out[1] or None)
    return out or "", None


def _openai_compatible_chat(client, model: str, messages: List[Dict[str, str]], system_prompt: str,
                            thinking: bool = False):
    payload = [{"role": "system", "content": system_prompt}] + messages
    # Reasoning models (GPT-5, gpt-oss) reject the older max_tokens parameter.
    kwargs = dict(model=model, messages=payload,
                  max_completion_tokens=THINKING_OUTPUT_TOKENS if thinking else MAX_OUTPUT_TOKENS)
    try:
        resp = client.chat.completions.create(
            **kwargs, reasoning_effort="high" if thinking else "low")
    except Exception as e:
        if "reasoning" not in str(e).lower():
            raise
        # This model doesn't support reasoning_effort; call it plainly.
        resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    # Groq's gpt-oss returns its reasoning in a separate field.
    reasoning = getattr(msg, "reasoning", None) if thinking else None
    return msg.content or "", reasoning


def _openai_chat(model: str, messages: List[Dict[str, str]], system_prompt: str, thinking: bool = False):
    from openai import OpenAI
    client = OpenAI(api_key=_provider_keys()["openai"])
    return _openai_compatible_chat(client, model, messages, system_prompt, thinking)


def _anthropic_chat(model: str, messages: List[Dict[str, str]], system_prompt: str, thinking: bool = False):
    from anthropic import Anthropic
    client = Anthropic(api_key=_provider_keys()["anthropic"])
    resp = client.messages.create(
        model=model,
        messages=messages,
        system=system_prompt,
        # Current Claude models always think, which uses part of this budget.
        max_tokens=16000,
    )
    if resp.stop_reason == "refusal":
        raise ValueError("The model declined to answer this request.")
    text = "".join(block.text for block in resp.content if block.type == "text")
    thoughts = "\n".join(getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking")
    return text, (thoughts if thinking else None)


def _google_chat(model: str, messages: List[Dict[str, str]], system_prompt: str, thinking: bool = False):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_provider_keys()["google"])
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in messages]
    base = dict(system_instruction=system_prompt,
                max_output_tokens=THINKING_OUTPUT_TOKENS if thinking else MAX_OUTPUT_TOKENS)
    try:
        # Normal mode: little thinking = fast. Thinking mode: deep thinking + a reasoning summary.
        level = "HIGH" if thinking else config.GEMINI_THINKING_LEVEL
        cfg = types.GenerateContentConfig(
            **base, thinking_config=types.ThinkingConfig(thinking_level=level, include_thoughts=thinking))
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    except Exception as e:
        if "thinking" not in str(e).lower():
            raise
        # This model doesn't accept a thinking level; retry with defaults.
        resp = client.models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(**base))
    answer, thoughts = [], []
    try:
        for part in resp.candidates[0].content.parts or []:
            if not getattr(part, "text", None):
                continue
            (thoughts if getattr(part, "thought", False) else answer).append(part.text)
    except (AttributeError, IndexError, TypeError):
        return resp.text or "", None
    return "".join(answer) or (resp.text or ""), ("\n".join(thoughts) or None)


def _groq_chat(model: str, messages: List[Dict[str, str]], system_prompt: str, thinking: bool = False):
    # Groq exposes an OpenAI-compatible API, so the existing openai package works.
    from openai import OpenAI
    client = OpenAI(api_key=_provider_keys()["groq"],
                    base_url="https://api.groq.com/openai/v1")
    return _openai_compatible_chat(client, model, messages, system_prompt, thinking)


# ---------------------------------------------------------------------------
# Retry + fallback
# ---------------------------------------------------------------------------
_TRANSIENT_HINTS = ("503", "429", "unavailable", "overloaded", "high demand",
                    "resource_exhausted", "rate limit", "timeout", "timed out", "529")


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(h in msg for h in _TRANSIENT_HINTS)


def chat_with_fallback(model: str, messages: List[Dict[str, str]],
                       system_prompt: Optional[str] = None,
                       thinking: bool = False) -> Tuple[str, str, List[str], Optional[str]]:
    """Try `model`; retry once on a temporary error, then fall back to other models.

    Returns (reply, model_that_answered, errors_from_failed_attempts, reasoning_or_None).
    Raises RuntimeError if every attempt fails.
    """
    available = [m["id"] for m in get_available_models()]
    # Prefer a *different provider* for fallback: when Gemini is overloaded,
    # its other models usually are too.
    try:
        own = _provider_for_model(model)
    except ValueError:
        own = None
    others = [m for m in available if m != model]
    others.sort(key=lambda m: _provider_for_model(m) == own)  # other providers first (stable)
    candidates = [model] + others[: int(config.MAX_FALLBACKS)]
    errors: List[str] = []
    for i, candidate in enumerate(candidates):
        for attempt in range(2):
            try:
                reply, thoughts = chat_completion(candidate, messages, system_prompt, thinking)
                return reply, candidate, errors, thoughts
            except Exception as e:
                short = str(e).splitlines()[0][:200]
                log.warning("model %s failed (attempt %d): %s", candidate, attempt + 1, short)
                errors.append(f"{candidate}: {short}")
                if attempt == 0 and _is_transient(e):
                    time.sleep(1.5)
                    continue
                break
    raise RuntimeError("All models failed. " + " | ".join(errors))
