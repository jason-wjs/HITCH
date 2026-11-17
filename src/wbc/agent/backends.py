"""Model backends: the conversation rendered to one provider API.

A backend turns the neutral conversation into a request, posts it, and turns
the reply into an assistant ``Message``. Which adapter a profile uses is its
``api``; ``APIS`` lists the adapters (the OpenAI Responses API). Adding
another API means one more class with ``complete`` and an entry in ``APIS``;
the loop, the tools and the prompts do not change.

Responses, as used here:

- Stateless (``store: false``): every request carries the whole conversation,
  so the episode record is exactly what the model saw, without relying on
  server-side response storage. A reply's output items (reasoning with its
  ``encrypted_content``, message, function calls) are kept as ``Message.raw``
  and sent back verbatim and in order, as the API requires for reasoning to
  carry over tool calls.
- Images ride inside ``function_call_output``, labelled with their paths.
- ``prompt_cache_key`` names the episode; cache hits remain best effort.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from wbc.agent.conversation import Image, Message, ToolCall

log = logging.getLogger(__name__)
PROFILES = Path(__file__).with_name("backends.yaml")


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    api: str  # a key of APIS
    base_url: str
    model: str | None = None
    api_key_env: str | None = None
    proxy_env: str | None = None  # optional environment variable holding the model API proxy URL
    extra_body: dict = field(default_factory=dict)  # merged into every request (reasoning effort, ...)
    max_output_tokens: int | None = None
    timeout_s: float = 600.0
    retries: int = 0  # a failed request (HTTP 400/408/409/429/5xx, timeout, connection) sent again, with backoff

    @classmethod
    def load(cls, name: str, **overrides) -> Profile:
        profiles = yaml.safe_load(PROFILES.read_text())
        if name not in profiles:
            raise BackendError(f"no backend profile {name!r}; known: {sorted(profiles)}")
        values = {**profiles[name], **{k: v for k, v in overrides.items() if v is not None}}
        profile = cls(name=name, **values)
        if profile.api not in APIS:
            raise BackendError(f"{name}: api must be one of {sorted(APIS)}")
        if not profile.model:
            raise BackendError(f"backend {name} needs a model (--model)")
        return profile

    def public(self) -> dict:
        """The profile as recorded with an episode (keys are named, never stored)."""
        return dict(self.__dict__)


@dataclass
class Reply:
    message: Message
    request: dict  # as sent, with images replaced by their file names
    response: dict  # as received
    usage: dict  # prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens
    seconds: float
    note: str = ""  # e.g. why the reply was cut short


class Backend(Protocol):
    profile: Profile

    def complete(self, messages: list[Message], tools: list[dict], cache_key: str | None = None) -> Reply: ...


def open_backend(profile: Profile, post=None) -> Backend:
    """The adapter for ``profile.api``; ``post(payload) -> dict`` may be overridden to replace HTTP."""
    return APIS[profile.api](profile, post)


class ResponsesBackend:
    """The OpenAI Responses API (``POST {base_url}/responses``)."""

    def __init__(self, profile: Profile, post=None):
        self.profile = profile
        self.post = post or (lambda payload: _post(profile, payload))

    def complete(self, messages: list[Message], tools: list[dict], cache_key: str | None = None) -> Reply:
        started = time.monotonic()
        request = self.render(messages, tools, lambda image: image, cache_key)  # once; each image written twice
        response = self.post(_with_images(request, _data_url))
        message, note = self.parse(response)
        return Reply(
            message,
            _with_images(request, lambda image: f"file:{image.path.name}"),
            response,
            _usage(response.get("usage") or {}),
            time.monotonic() - started,
            note,
        )

    # ---------------------------------------------------------------- request
    def render(self, messages: list[Message], tools: list[dict], image_url, cache_key=None) -> dict:
        system = [m.text for m in messages if m.role == "system"]
        items = []
        for message in messages:
            if message.role == "user":
                items.append({"role": "user", "content": _content(message.text, message.images, image_url)})
            elif message.role == "assistant":
                if not message.raw:
                    raise BackendError("an assistant message without its Responses output cannot be sent back")
                items += message.raw  # verbatim: reasoning (encrypted), message, function calls, in order
            elif message.role == "tool":
                output = _content(message.text, message.images, image_url) if message.images else message.text
                items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": output})
        p = self.profile
        payload = {
            "model": p.model,
            "instructions": "\n\n".join(system),
            "input": items,
            "tools": [{"type": "function", **spec, "strict": False} for spec in tools],
            "store": False,
            "include": ["reasoning.encrypted_content"],  # the default when not stored; asked for explicitly
            **p.extra_body,
        }
        if cache_key:
            payload["prompt_cache_key"] = cache_key
        if p.max_output_tokens:
            payload["max_output_tokens"] = p.max_output_tokens
        return payload

    # --------------------------------------------------------------- response
    def parse(self, response: dict) -> tuple[Message, str]:
        try:
            return self._parse(response)
        except (KeyError, TypeError, AttributeError) as error:  # a response not in the expected shape
            raise BackendError(f"malformed response ({error!r}): {json.dumps(response, default=str)[:500]}") from None

    def _parse(self, response: dict) -> tuple[Message, str]:
        if response.get("error"):
            raise BackendError(f"{self.profile.name}: {response['error']}")
        output = response.get("output")
        if not isinstance(output, list):
            raise BackendError(f"no output in the response: {json.dumps(response)[:500]}")
        text, calls, thoughts = [], [], []
        note = ""
        status = response.get("status")
        if status == "incomplete":
            note = f"cut short: {(response.get('incomplete_details') or {}).get('reason', 'unknown')}"
        elif status not in (None, "completed"):
            note = f"response not completed: {status}"
        for item in output:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    text.append(part.get("text") or part.get("refusal") or "")
            elif kind == "function_call":
                if item.get("namespace") is not None:
                    raise BackendError(f"unexpected tool namespace: {item['namespace']}")
                arguments = item.get("arguments")
                if item.get("status") not in (None, "completed"):
                    note = note or f"function call not completed: {item.get('call_id')}"
                elif not isinstance(arguments, str) or not arguments.strip():
                    note = note or f"function call missing arguments: {item.get('call_id')}"
                else:
                    calls.append(ToolCall(item["call_id"], item["name"], arguments))
            elif kind == "reasoning":
                # Local servers can return full plaintext reasoning in content.
                # Prefer it for display; raw output is replayed unchanged above.
                content = [part["text"] for part in item.get("content") or [] if part.get("text")]
                thoughts += content or [s.get("text", "") for s in item.get("summary") or []]
        if note:
            calls = []  # do not dispatch any part of an unfinished batch
        message = Message(
            "assistant", "".join(text), tool_calls=tuple(calls), raw=output, reasoning_text="\n\n".join(thoughts)
        )
        return message, note


APIS = {"responses": ResponsesBackend}


# -------------------------------------------------------------------- helpers
def _content(text: str, images: tuple[Image, ...], image_url) -> list[dict]:
    parts = [{"type": "input_text", "text": text}] if text else []
    for image in images:
        parts += [
            {"type": "input_text", "text": f"[{image.label}]"},
            {"type": "input_image", "image_url": image_url(image), "detail": "auto"},
        ]
    return parts


def _usage(usage: dict) -> dict:
    """Responses usage in the loop's terms."""
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
    }


def _data_url(image: Image) -> str:
    return "data:image/png;base64," + base64.b64encode(image.path.read_bytes()).decode()


def _with_images(value, image_url):
    """A copy of a request rendered with the images themselves, each written by ``image_url``."""
    if isinstance(value, Image):
        return image_url(value)
    if isinstance(value, dict):
        return {key: _with_images(item, image_url) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_images(item, image_url) for item in value]
    return value


def _post(profile: Profile, payload: dict) -> dict:
    token = "unused"  # the SDK needs a key; a server without credentials (a local one, the fake server) ignores it
    if profile.api_key_env:
        token = os.environ.get(profile.api_key_env)
        if not token:
            raise BackendError(f"set {profile.api_key_env} for backend {profile.name}")
    return _sdk_response(profile, payload, token)


def _sdk_response(profile: Profile, payload: dict, token: str, *, http_client=None) -> dict:
    """One Responses request through the official SDK, sent again up to ``profile.retries`` times."""
    from openai import APIStatusError, DefaultHttpxClient, OpenAI

    # Provider options (e.g. chat_template_kwargs) go through the SDK's documented escape hatch.
    request = {k: v for k, v in payload.items() if k not in profile.extra_body}
    extra = {k: payload[k] for k in profile.extra_body if k in payload}
    if extra:
        request["extra_body"] = extra
    with OpenAI(
        api_key=token,
        base_url=profile.base_url,
        timeout=profile.timeout_s,
        max_retries=0,
        http_client=http_client
        or DefaultHttpxClient(trust_env=False, proxy=os.environ.get(profile.proxy_env) if profile.proxy_env else None),
    ) as client:
        for attempt in range(profile.retries + 1):
            last = attempt == profile.retries
            try:
                return client.responses.create(**request, stream=False).to_dict()
            except APIStatusError as error:
                # Status, request ID and the error's code diagnose provider failures without logging keys,
                # request images, or an arbitrary server response body (its message).
                failure = (
                    f"{profile.name}: Responses HTTP {error.status_code} (request_id={error.request_id!r}, "
                    f"code={_error_code(error.body)!r})"
                )
                if last or error.status_code not in (400, 408, 409, 429) and error.status_code < 500:
                    raise BackendError(f"{failure}; {_retried(attempt)}") from None
            except Exception as error:
                failure = f"{profile.name}: Responses request failed ({type(error).__name__})"
                if last:
                    raise BackendError(f"{failure}; {_retried(attempt)}") from None
            log.warning("%s; sending it again (%s/%s)", failure, attempt + 1, profile.retries)
            time.sleep(min(60, 5 * 2**attempt))


def _retried(attempt: int) -> str:
    return f"retried {attempt} times" if attempt else "not retried"


def _error_code(body) -> str | None:
    """The provider's error code or type (e.g. rate_limit), never its message."""
    error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("code") or error.get("type")
    return str(code)[:80] if code else None
