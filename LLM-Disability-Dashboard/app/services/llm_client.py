"""Utility helpers for invoking LLM-backed service functions and normalizing responses."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastapi import Response
from openai import OpenAI

from .cache import LLMCache, cache_config_from_env

JSONLike = Union[Dict[str, Any], List[Any]]
Payload = Union[Response, JSONLike, str, None]
AsyncCallable = Callable[..., Awaitable[Any]]


class LLMClient:
    """Wrapper that normalizes outputs from async service functions."""

    def __init__(self) -> None:
        self._json_dumps_kwargs = {"ensure_ascii": False}
        config = cache_config_from_env()
        self._cache = LLMCache(
            ttl_seconds=config.get("ttl_seconds", 600),
            max_entries=config.get("max_entries", 128),
        )
        env_flag = os.getenv("LANGGRAPH_CACHE_ENABLED", "true").strip().lower()
        self._cache_enabled = env_flag not in {"0", "false", "no", "off"}
        self._last_cache_hit = False
        self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def invoke(
        self,
        handler: AsyncCallable,
        *args: Any,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> JSONLike:
        """Execute a handler and coerce the payload into JSON-compatible data."""

        cache_key: Optional[str] = None
        if self._cache_enabled and use_cache:
            cache_key = self._make_cache_key(handler, args, kwargs)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._last_cache_hit = True
                return cached

        payload = await handler(*args, **kwargs)
        normalized = self._normalize_payload(payload)

        self._last_cache_hit = False

        if cache_key is not None and self._cache_enabled and use_cache:
            self._cache.set(cache_key, normalized)

        return normalized

    async def invoke_with_prompt(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        use_cache: bool = True,
    ) -> JSONLike:
        """Execute a direct prompt call to OpenAI and return normalized JSON response."""
        messages = [{"role": "user", "content": prompt}]
        return await self.invoke_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            use_cache=use_cache,
        )

    async def invoke_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        use_cache: bool = True,
    ) -> JSONLike:
        """Execute a chat-style prompt call to OpenAI and return normalized JSON response."""

        cache_key: Optional[str] = None
        if self._cache_enabled and use_cache:
            cache_key = self._make_messages_cache_key(messages, model, temperature)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._last_cache_hit = True
                return cached

        try:
            response = self._openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")

            json_data = json.loads(content)
            normalized = self._normalize_payload(json_data)

            self._last_cache_hit = False

            if cache_key is not None and self._cache_enabled and use_cache:
                self._cache.set(cache_key, normalized)

            return normalized

        except Exception as e:
            raise ValueError(f"Error calling OpenAI: {str(e)}")

    def ensure_dict(self, data: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
        """Ensure the provided data is returned as a dictionary."""

        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            text = data.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Expected JSON object string") from exc
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object after parsing")
            return parsed
        raise ValueError("Expected dictionary payload or JSON object string")

    def dumps(self, data: Any) -> str:
        """Serialize Python data into JSON string for downstream LLM calls."""

        return json.dumps(data, **self._json_dumps_kwargs)

    def _normalize_payload(self, payload: Payload) -> JSONLike:
        """Convert FastAPI `Response`, raw JSON strings, or dicts into JSON structures."""

        if payload is None:
            return {}

        if isinstance(payload, Response):
            raw_body: Optional[bytes] = payload.body
            text = ""
            if isinstance(raw_body, bytes):
                charset = getattr(payload, "charset", "utf-8") or "utf-8"
                text = raw_body.decode(charset).strip()
            elif raw_body is None:
                text = ""
            else:  # pragma: no cover - defensive branch
                text = str(raw_body).strip()

            if not text:
                return {}

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Unable to parse Response body as JSON") from exc
            return parsed

        if isinstance(payload, (dict, list)):
            return payload

        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Unable to parse string payload as JSON") from exc
            if isinstance(parsed, (dict, list)):
                return parsed
            raise ValueError("Parsed payload is not JSON object/array")

        raise ValueError(f"Unsupported payload type: {type(payload)!r}")

    def _make_cache_key(self, handler: AsyncCallable, args: Any, kwargs: Dict[str, Any]) -> str:
        payload = {
            "handler": getattr(handler, "__qualname__", getattr(handler, "__name__", repr(handler))),
            "module": getattr(handler, "__module__", ""),
            "args": self._prepare_for_cache(args),
            "kwargs": self._prepare_for_cache(kwargs),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _make_prompt_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create cache key for direct prompt calls."""
        payload = {
            "type": "prompt",
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _make_messages_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        payload = {
            "type": "chat",
            "messages": self._prepare_for_cache(messages),
            "model": model,
            "temperature": temperature,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _prepare_for_cache(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {
                str(key): self._prepare_for_cache(value[key])
                for key in sorted(value)
            }
        if isinstance(value, (list, tuple)):
            return [self._prepare_for_cache(item) for item in value]
        if isinstance(value, set):
            return sorted(self._prepare_for_cache(item) for item in value)
        # Fallback to string representation for unsupported types
        return repr(value)

    @property
    def last_cache_hit(self) -> bool:
        return self._last_cache_hit
