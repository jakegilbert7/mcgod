#!/usr/bin/env python3
"""Provider-neutral model client for Anthropic and OpenRouter.

The agent speaks the small subset of Anthropic's Messages API it was built around.
OpenRouter uses OpenAI-compatible chat completions, so translation happens once here rather
than leaking provider conditionals into every world-state subsystem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from config import (MODEL_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    OPENROUTER_REASONING_EFFORT)


def _provider_module(package: str, provider: str):
    """Load only the selected provider and explain how to repair its environment."""
    try:
        return __import__(package)
    except ModuleNotFoundError as exc:
        if exc.name != package:
            raise
        raise ModuleNotFoundError(
            f"{provider} is selected, but its Python SDK is not installed in this "
            "environment. Run: python -m pip install -r requirements.lock"
        ) from exc


@dataclass
class TextBlock:
    text: str
    type: str = "text"

    def model_dump(self, **_kwargs):
        return {"type": self.type, "text": self.text}


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self, **_kwargs):
        return {"type": self.type, "id": self.id, "name": self.name,
                "input": self.input}


class ModelMessage:
    def __init__(self, content, stop_reason="end_turn", usage=None,
                 parsed_output=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or SimpleNamespace(input_tokens=0, output_tokens=0)
        self.parsed_output = parsed_output


def _data_image(source: dict) -> str:
    if source.get("type") == "base64":
        media = source.get("media_type", "image/png")
        return f"data:{media};base64,{source.get('data', '')}"
    return str(source.get("url") or source.get("image_url") or "")


def _user_parts(content) -> str | list:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if kind == "text":
            value = block.get("text", "") if isinstance(block, dict) else block.text
            parts.append({"type": "text", "text": value})
        elif kind == "image":
            source = block.get("source", {}) if isinstance(block, dict) else block.source
            parts.append({"type": "image_url", "image_url": {"url": _data_image(source)}})
    return parts or ""


def _tool_result_messages(content: list) -> list[dict]:
    messages, images = [], []
    for block in content:
        result, texts = block.get("content", ""), []
        if isinstance(result, str):
            texts.append(result)
        else:
            for item in result or []:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    images.append({"type": "image_url", "image_url": {
                        "url": _data_image(item.get("source", {}))}})
        if block.get("is_error"):
            texts.insert(0, "Tool error:")
        messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                         "content": "\n".join(texts) or "{}"})
    # Chat Completions tool messages cannot portably contain image parts. Return every tool
    # result first (satisfying the tool-call contract), then attach the render as user input.
    if images:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "Images returned by the preceding tool calls:"},
            *images,
        ]})
    return messages


def openrouter_messages(messages: list[dict], system=None) -> list[dict]:
    """Translate Messages-API history without losing tool or visual evidence."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        role, content = message["role"], message.get("content", "")
        if (role == "user" and isinstance(content, list) and content
                and all(item.get("type") == "tool_result" for item in content)):
            out.extend(_tool_result_messages(content))
            continue
        if role == "assistant" and isinstance(content, list):
            texts, calls = [], []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    texts.append(block.get("text", ""))
                elif kind == "tool_use":
                    calls.append({
                        "id": block["id"], "type": "function",
                        "function": {"name": block["name"],
                                     "arguments": json.dumps(block.get("input") or {})},
                    })
                # Anthropic thinking signatures are provider-specific. Tool calls and their
                # evidence are retained; OpenRouter reasoning is configured per request.
            item = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                item["tool_calls"] = calls
            out.append(item)
            continue
        out.append({"role": role, "content": _user_parts(content)})
    return out


def openrouter_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object"}),
    }} for tool in tools]


def _strict_schema(value):
    """Make Pydantic's schema acceptable to strict OpenAI/OpenRouter responders."""
    if isinstance(value, dict):
        out = {key: _strict_schema(item) for key, item in value.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    return value


def _json_schema(output_format) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": output_format.__name__, "strict": True,
        "schema": _strict_schema(output_format.model_json_schema()),
    }}


def _normalise(response, output_format=None) -> ModelMessage:
    choice, blocks = response.choices[0], []
    message = choice.message
    if message.content:
        value = (message.content if isinstance(message.content, str)
                 else "".join(getattr(part, "text", "") for part in message.content))
        if value:
            blocks.append(TextBlock(value))
    for call in message.tool_calls or []:
        arguments = call.function.arguments or "{}"
        try:
            values = arguments if isinstance(arguments, dict) else json.loads(arguments)
        except json.JSONDecodeError:
            values = {"_invalid_json": arguments}
        blocks.append(ToolUseBlock(call.id, call.function.name, values))
    reasons = {"tool_calls": "tool_use", "length": "max_tokens",
               "stop": "end_turn", "content_filter": "refusal"}
    usage = getattr(response, "usage", None)
    normal_usage = SimpleNamespace(
        input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
    )
    parsed = None
    if output_format is not None:
        text = "".join(block.text for block in blocks if block.type == "text")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("model returned no JSON object")
        parsed = output_format.model_validate_json(text[start:end + 1])
    return ModelMessage(blocks, reasons.get(choice.finish_reason, choice.finish_reason),
                        normal_usage, parsed)


def _request(values: dict, output_format=None) -> dict:
    values = dict(values)
    system = values.pop("system", None)
    values.pop("thinking", None)
    values["messages"] = openrouter_messages(values["messages"], system)
    tools = openrouter_tools(values.pop("tools", None))
    if tools:
        values["tools"] = tools
    if output_format is not None:
        values["response_format"] = _json_schema(output_format)
    if OPENROUTER_REASONING_EFFORT:
        values["reasoning_effort"] = OPENROUTER_REASONING_EFFORT
    return values


class _EagerStream:
    def __init__(self, messages, values):
        self.messages, self.values, self.reply = messages, values, None

    def __enter__(self):
        self.reply = self.messages.create(**self.values)
        return self

    def __exit__(self, *_args):
        return False

    def get_final_message(self):
        return self.reply


class OpenRouterMessages:
    def __init__(self, client):
        self.client = client

    def create(self, **values):
        return _normalise(self.client.chat.completions.create(**_request(values)))

    def parse(self, output_format, **values):
        response = self.client.chat.completions.create(
            **_request(values, output_format))
        return _normalise(response, output_format)

    def stream(self, **values):
        return _EagerStream(self, values)

    def count_tokens(self, messages, **_values):
        from context import estimate_tokens
        return SimpleNamespace(input_tokens=estimate_tokens(json.dumps(messages)))


class AsyncOpenRouterMessages:
    def __init__(self, client):
        self.client = client

    async def create(self, **values):
        return _normalise(await self.client.chat.completions.create(**_request(values)))

    async def parse(self, output_format, **values):
        response = await self.client.chat.completions.create(
            **_request(values, output_format))
        return _normalise(response, output_format)


class OpenRouterClient:
    def __init__(self, client=None):
        if client is None:
            openai = _provider_module("openai", "OpenRouter")
            client = openai.OpenAI(api_key=OPENROUTER_API_KEY,
                                   base_url=OPENROUTER_BASE_URL)
        self.messages = OpenRouterMessages(client)


class AsyncOpenRouterClient:
    def __init__(self, client=None):
        if client is None:
            openai = _provider_module("openai", "OpenRouter")
            client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY,
                                        base_url=OPENROUTER_BASE_URL)
        self.messages = AsyncOpenRouterMessages(client)


def model_client(*, asynchronous=False):
    if MODEL_PROVIDER == "openrouter":
        return AsyncOpenRouterClient() if asynchronous else OpenRouterClient()
    anthropic = _provider_module("anthropic", "Anthropic")
    return anthropic.AsyncAnthropic() if asynchronous else anthropic.Anthropic()


class _UnavailableProviderError(Exception):
    """Placeholder used only until model_client reports a missing selected SDK."""


try:
    _error_sdk = __import__("openai" if MODEL_PROVIDER == "openrouter" else "anthropic")
except ModuleNotFoundError:
    _error_sdk = None


def _error_type(name: str):
    return getattr(_error_sdk, name, _UnavailableProviderError)


NOT_FOUND_ERRORS = (_error_type("NotFoundError"),)
RATE_LIMIT_ERRORS = (_error_type("RateLimitError"),)
STATUS_ERRORS = (_error_type("APIStatusError"),)
CONNECTION_ERRORS = (_error_type("APIConnectionError"),)
