"""Devin inference transport: an OpenAI-SDK-shaped client (``client.chat.completions.create``) over
Codeium Cascade's ``GetChatMessage`` Connect stream.

Per turn: ``GetUserJwt`` exchanges the stored credential for a user JWT (cached until it expires), a
router model (``adaptive``) is resolved to a concrete uid through ``AssignModel``, and the OpenAI-shaped
history becomes ``ChatMessagePrompt`` rows (USER / SYSTEM = assistant / TOOL). Streamed
``GetChatMessageResponse`` frames come back as ``chat.completion.chunk`` objects, so the agent loop stays
OpenAI-shaped.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import hashlib
import json
import logging
import re
import struct
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import httpx

from . import catalog
from . import protocol as wire

logger = logging.getLogger(__name__)

NATIVE_DETAILS_TYPE = "devin.native_assistant"
MISSING_KEY_ERROR = ("Devin needs a credential. Run `hermes auth add devin` to sign in, or set DEVIN_API_KEY "
                     "(a Devin/Windsurf API key).")

# Cascade forwards replayed tool-call ids to the backend serving the model; OpenAI-routed models reject ids
# past 40 chars or outside [A-Za-z0-9_-] with an opaque invalid_argument.
_TOOL_CALL_ID_MAX = 40
_TOOL_CALL_ID_BAD = re.compile(r"[^A-Za-z0-9_-]")
_MIN_COMPRESSED_BYTES = 1024
# An opaque ``invalid_argument: internal error`` on a history this large is treated as context overflow so
# compression runs, instead of failing the turn as a malformed request (the backend states no limit).
_LARGE_HISTORY_BYTES = 512 * 1024
_MESSAGE_ID_NAMESPACE = uuid.UUID("6f0b7f2e-5d0c-4f43-9c8e-3d6a1f1b2c71")
_JWT_FALLBACK_TTL = 300.0
_END = object()

# Connect error codes → the HTTP status Hermes' error classifier reasons about.
_CONNECT_STATUS = {
    "canceled": 499, "unknown": 500, "invalid_argument": 400, "deadline_exceeded": 504, "not_found": 404,
    "already_exists": 409, "permission_denied": 403, "resource_exhausted": 429, "failed_precondition": 400,
    "aborted": 409, "out_of_range": 400, "unimplemented": 501, "internal": 500, "unavailable": 503,
    "data_loss": 500, "unauthenticated": 401,
}


class DevinAPIError(Exception):
    """Error shape compatible with Hermes retry/error classification."""

    def __init__(self, message: str, *, code: str = "devin_api_error", status_code: Optional[int] = None,
                 response: Optional[httpx.Response] = None, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code, self.status_code, self.response, self.details = code, status_code, response, details or {}
        self.retry_after = None
        if response is not None:
            with contextlib.suppress(Exception):
                from agent.retry_utils import parse_retry_after_seconds
                self.retry_after = parse_retry_after_seconds(response.headers)


def _error_detail(body: bytes) -> str:
    """Bounded human text from an error body (Connect JSON ``{code, message}``); never HTML or binary."""
    try:
        text = body.decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""
    with contextlib.suppress(ValueError):
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            err = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
            text = str(err.get("message") or err.get("code") or "").strip()
    text = " ".join(text.split())
    if text.lower().startswith(("<!doctype html", "<html")) or any(ord(c) < 32 for c in text):
        return ""
    return text[:2000]


def _connect_code(body: bytes) -> str:
    with contextlib.suppress(ValueError, UnicodeDecodeError):
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, dict):
            err = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
            return str(err.get("code") or "")
    return ""


def http_error(operation: str, response: httpx.Response, body: bytes) -> DevinAPIError:
    status = response.status_code
    detail = _error_detail(body)
    connect_code = _connect_code(body)
    return DevinAPIError(f"Devin {operation} error HTTP {status}{f': {detail}' if detail else ''}",
                         code=f"devin_{connect_code or f'http_{status}'}", status_code=status, response=response,
                         details={"connect_code": connect_code, "message": detail})


def trailer_error(text: str) -> Optional[DevinAPIError]:
    """The structured error of a Connect end-of-stream trailer ``{"error": {code, message, details}}``."""
    try:
        parsed = json.loads(text) if text.strip() else None
    except ValueError:
        return None
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(err, dict):
        return None
    code, message = str(err.get("code") or ""), str(err.get("message") or "")
    if not code and not message:
        return None
    return DevinAPIError(f"Devin stream error{f' {code}' if code else ''}: {message}",
                         code=f"devin_{code or 'stream_error'}", status_code=_CONNECT_STATUS.get(code.lower(), 500),
                         details={"connect_code": code, "message": message, "trailer": text[:2000]})


# ── OpenAI messages → Cascade prompts ────────────────────────────────────────────────────────────


def sanitize_tool_call_id(call_id: Any) -> str:
    """A Cascade-safe tool-call id: the ``call_id`` half of a Responses composite, ``[A-Za-z0-9_-]`` only,
    at most 40 chars (an over-long id keeps a hash suffix so distinct ids stay distinct)."""
    raw = str(call_id or "").split("|", 1)[0]
    clean = _TOOL_CALL_ID_BAD.sub("_", raw)
    if len(clean) <= _TOOL_CALL_ID_MAX:
        return clean
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{clean[:_TOOL_CALL_ID_MAX - 9]}_{digest}"


def _message_id(cascade_id: str, index: int, role: str) -> str:
    # Stable across turns and history rebuilds: prompt text is not part of the seed.
    return str(uuid.uuid5(_MESSAGE_ID_NAMESPACE, f"{cascade_id}\0{index}\0{role}"))


def _image(url: str) -> Optional[dict[str, str]]:
    match = re.match(r"data:([^;,]+);base64,(.*)$", url or "", re.DOTALL)
    return {"base64_data": match.group(2), "mime_type": match.group(1)} if match else None


def _flatten(content: Any) -> tuple[str, list[dict[str, str]]]:
    """``(text, images)`` of an OpenAI content value. Remote image URLs (Cascade takes inline bytes only)
    are kept as a text reference so the model still knows one was attached."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    text, images = [], []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
            text.append(str(part.get("text") or ""))
        elif isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if (image := _image(str(url or ""))) is not None:
                images.append(image)
            elif url:
                text.append(f"[image: {url}]")
    return "".join(text), images


def _native_record(message: dict[str, Any], model: str) -> dict[str, Any]:
    for detail in message.get("reasoning_details") or ():
        if isinstance(detail, dict) and detail.get("type") == NATIVE_DETAILS_TYPE and detail.get("model") == model:
            return detail
    return {}


def build_prompts(messages: list[dict[str, Any]], cascade_id: str, model: str) -> tuple[str, list[dict[str, Any]]]:
    """``(system prompt, chat_message_prompts)``. Leading system/developer messages form the top-level
    prompt; a later one rides as a USER row so its position in the conversation is kept."""
    system_parts: list[str] = []
    prompts: list[dict[str, Any]] = []
    leading = True
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("system", "developer") and leading:
            text, _ = _flatten(message.get("content"))
            if text:
                system_parts.append(text)
            continue
        leading = False
        if role in ("user", "system", "developer"):
            text, images = _flatten(message.get("content"))
            prompts.append({"message_id": _message_id(cascade_id, index, str(role)), "source": wire.SOURCE_USER,
                            "prompt": text, "images": images})
        elif role == "assistant":
            text, _ = _flatten(message.get("content"))
            thinking = message.get("reasoning_content") or message.get("reasoning") or ""
            calls = [{"id": sanitize_tool_call_id(call.get("id")), "name": str((call.get("function") or {}).get("name") or ""),
                      "arguments_json": str((call.get("function") or {}).get("arguments") or "{}")}
                     for call in message.get("tool_calls") or () if isinstance(call, dict)]
            # A thinking signature is valid only for the model that signed it.
            native = _native_record(message, model)
            if not (text or thinking or calls):
                continue
            prompts.append({
                "message_id": str(native.get("message_id") or "") or _message_id(cascade_id, index, "assistant"),
                "source": wire.SOURCE_SYSTEM, "prompt": text, "thinking": str(thinking) if isinstance(thinking, str) else "",
                "signature": str(native.get("signature") or ""), "signature_type": str(native.get("signature_type") or ""),
                "tool_calls": calls,
            })
        elif role == "tool":
            text, images = _flatten(message.get("content"))
            call_id = sanitize_tool_call_id(message.get("tool_call_id"))
            prompts.append({"message_id": _message_id(cascade_id, index, f"tool\0{call_id}"), "source": wire.SOURCE_TOOL,
                            "tool_call_id": call_id, "prompt": text, "images": images,
                            "tool_result_is_error": bool(message.get("is_error"))})
    return "\n\n".join(system_parts), prompts


def _uses_gemini_schema(*uids: str) -> bool:
    # Devin's Gemini backend applies Google's tool-schema constraints (no JSON-Schema type arrays) and
    # rejects them as an opaque invalid_argument; router-assigned uids use the server's enum namespace.
    return any("gemini" in uid.lower() for uid in uids if uid)


def build_tools(tools: Any, *, gemini: bool) -> list[dict[str, Any]]:
    out = []
    for tool in tools or ():
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        if gemini:
            from agent.gemini_schema import sanitize_gemini_tool_parameters
            schema = sanitize_gemini_tool_parameters(schema)
        out.append({"name": str(fn["name"]), "description": str(fn.get("description") or ""),
                    "json_schema_string": json.dumps(schema, ensure_ascii=False), "strict": bool(fn.get("strict"))})
    return out


def _tool_choice(choice: Any) -> dict[str, str]:
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") if isinstance(choice.get("function"), dict) else None
        if name:
            return {"tool_name": str(name)}
    if isinstance(choice, str) and choice:
        return {"option_name": choice}
    return {"option_name": "auto"}


# ── Cascade stream → OpenAI chunks ───────────────────────────────────────────────────────────────


class DevinStreamChunk(SimpleNamespace):
    """One ``chat.completion.chunk``."""


def _usage(stats: dict[str, Any]) -> SimpleNamespace:
    # Cascade reports uncached input separately; OpenAI's prompt_tokens includes the cached share.
    fresh, out = int(stats.get("input_tokens") or 0), int(stats.get("output_tokens") or 0)
    read, write = int(stats.get("cache_read_tokens") or 0), int(stats.get("cache_write_tokens") or 0)
    prompt = fresh + read + write
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=read, cache_write_tokens=write))


class _Turn:
    """Accumulates one streamed turn and translates each frame into OpenAI chunks."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.created = int(time.time())
        self.text: list[str] = []
        self.thinking: list[str] = []
        self.signature = self.signature_type = self.message_id = self.actual_model = ""
        self.calls: dict[str, dict[str, Any]] = {}  # tool-call id → {index, name, arguments}
        self.active_call: Optional[str] = None
        self.stop_reason = 0
        self.usage: Optional[SimpleNamespace] = None
        self.credit_cost: Optional[int] = None

    def _chunk(self, *, content: Optional[str] = None, reasoning: Optional[str] = None, tool_calls: Any = None,
               finish_reason: Optional[str] = None, reasoning_details: Any = None) -> DevinStreamChunk:
        delta = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls, reasoning=reasoning,
                                reasoning_content=reasoning, reasoning_details=reasoning_details)
        return DevinStreamChunk(id=self.id, object="chat.completion.chunk", created=self.created, model=self.model,
                                choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)], usage=None)

    def feed(self, frame: dict[str, Any]) -> list[DevinStreamChunk]:
        chunks: list[DevinStreamChunk] = []
        self.message_id = self.message_id or str(frame.get("message_id") or "")
        self.actual_model = str(frame.get("actual_model_uid") or "") or self.actual_model
        if thinking := frame.get("delta_thinking"):
            self.thinking.append(thinking)
            chunks.append(self._chunk(reasoning=thinking))
        if frame.get("delta_signature"):
            self.signature = frame["delta_signature"]
        if frame.get("delta_signature_type"):
            self.signature_type = frame["delta_signature_type"]
        if text := frame.get("delta_text"):
            self.text.append(text)
            chunks.append(self._chunk(content=text))
        for call in frame.get("delta_tool_calls") or ():
            call_id = str(call.get("id") or "") or self.active_call
            if not call_id:
                continue
            slot = self.calls.get(call_id)
            if slot is None:
                slot = self.calls[call_id] = {"index": len(self.calls), "name": "", "arguments": ""}
            slot["name"] = str(call.get("name") or "") or slot["name"]
            self.active_call = call_id
            fragment = str(call.get("arguments_json") or "")
            # Frames carry either the accumulated arguments or just the new suffix.
            previous = slot["arguments"]
            accumulated = fragment if fragment.startswith(previous) else previous + fragment
            slot["arguments"] = accumulated
            delta = accumulated[len(previous):]
            chunks.append(self._chunk(tool_calls=[SimpleNamespace(
                index=slot["index"], id=call_id, type="function",
                function=SimpleNamespace(name=slot["name"], arguments=delta))]))
        if frame.get("stop_reason"):
            self.stop_reason = frame["stop_reason"]
        if frame.get("usage") is not None:
            self.usage = _usage(frame["usage"])
        if frame.get("credit_cost"):
            self.credit_cost = frame["credit_cost"]
        return chunks

    @property
    def finish_reason(self) -> str:
        if self.calls:
            return "tool_calls"
        if self.stop_reason == wire.STOP_REASON_MAX_TOKENS:
            return "length"
        if self.stop_reason == wire.STOP_REASON_CONTENT_FILTER:
            return "content_filter"
        return "stop"

    def reasoning_details(self) -> Optional[list[dict[str, Any]]]:
        """Private replay carrier: the response id and thinking signature, valid for this model only."""
        if not (self.signature or self.message_id):
            return None
        record = {"type": NATIVE_DETAILS_TYPE, "model": self.model, "message_id": self.message_id}
        if self.signature:
            record.update(signature=self.signature, signature_type=self.signature_type)
        return [record]

    def finish_chunk(self) -> DevinStreamChunk:
        chunk = self._chunk(finish_reason=self.finish_reason, reasoning_details=self.reasoning_details())
        chunk.usage = self.usage
        return chunk

    def completion(self) -> SimpleNamespace:
        tool_calls = [SimpleNamespace(id=call_id, type="function", index=slot["index"],
                                      function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"] or "{}"))
                      for call_id, slot in self.calls.items()]
        reasoning = "".join(self.thinking) or None
        message = SimpleNamespace(role="assistant", content="".join(self.text) or ("" if not tool_calls else None),
                                  tool_calls=tool_calls or None, reasoning=reasoning, reasoning_content=reasoning,
                                  reasoning_details=self.reasoning_details())
        usage = self.usage or SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                                              prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(id=self.id, object="chat.completion", created=self.created, model=self.model,
                               choices=[SimpleNamespace(index=0, message=message, finish_reason=self.finish_reason)],
                               usage=usage)


def _jwt_expiry(token: str) -> Optional[float]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        exp = payload.get("exp") if isinstance(payload, dict) else None
        return float(exp) if isinstance(exp, (int, float)) else None
    except (ValueError, TypeError, struct.error):
        return None


# ── client ───────────────────────────────────────────────────────────────────────────────────────


class DevinClient:
    """Minimal OpenAI-SDK-compatible facade over Devin's Cascade chat API."""

    # A complete client: never re-dispatched through a wire adapter (auxiliary_client).
    HERMES_SKIP_TRANSPORT_WRAP = True

    def __init__(self, *, api_key: str = "", base_url: Optional[str] = None, timeout: Any = None,
                 http_client: Optional[httpx.Client] = None, default_headers: Optional[dict[str, str]] = None,
                 **_: Any) -> None:
        if not str(api_key or "").strip():
            raise RuntimeError(MISSING_KEY_ERROR)
        self.api_key, self.is_closed = str(api_key).strip(), False
        self.base_url = (str(base_url or "").strip() or wire.DEFAULT_BASE_URL).rstrip("/")
        self._default_headers = dict(default_headers or {})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self._http = http_client or httpx.Client(
            timeout=timeout or httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=30.0))
        self._auth_lock = threading.Lock()
        self._auth: Optional[tuple[str, str, float]] = None  # (user_jwt, chat base URL, expires monotonic)

    def close(self) -> None:
        self.is_closed = True
        with contextlib.suppress(Exception):
            self._http.close()

    def __enter__(self) -> DevinClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def as_async_client(self) -> AsyncDevinClient:
        return AsyncDevinClient(self)

    # ── unary RPCs ──

    def _unary(self, operation: str, url: str, schema_in: Any, message: dict[str, Any], schema_out: Any,
               timeout: Any) -> dict[str, Any]:
        kwargs = {"timeout": timeout} if timeout is not None else {}
        response = self._http.post(url, content=wire.encode(schema_in, message), headers=wire.unary_headers(), **kwargs)
        body = response.content
        if response.status_code != 200:
            raise http_error(operation, response, body)
        decoded = wire.decode_unary(schema_out, body)
        if decoded is None:
            raise DevinAPIError(f"Devin {operation} returned an undecodable response", code="devin_bad_response",
                                status_code=response.status_code)
        return decoded

    def _user_jwt(self, timeout: Any) -> tuple[str, str]:
        """``(user_jwt, chat base URL)``; GetUserJwt is called once per JWT lifetime, not per turn."""
        with self._auth_lock:
            if self._auth is not None and time.monotonic() < self._auth[2]:
                return self._auth[0], self._auth[1]
            decoded = self._unary("auth", self.base_url + wire.GET_USER_JWT_PATH, wire.GET_USER_JWT_REQUEST,
                                  {"metadata": wire.cli_metadata(self.api_key)}, wire.GET_USER_JWT_RESPONSE, timeout)
            user_jwt = str(decoded.get("user_jwt") or "")
            if not user_jwt:
                raise DevinAPIError("Devin auth error: GetUserJwt returned an empty user JWT", code="devin_unauthenticated",
                                    status_code=401)
            chat_base = str(decoded.get("custom_api_server_url") or "").strip().rstrip("/") or self.base_url
            exp = _jwt_expiry(user_jwt)
            ttl = max(0.0, exp - time.time() - 60.0) if exp is not None else _JWT_FALLBACK_TTL
            self._auth = (user_jwt, chat_base, time.monotonic() + ttl)
            return user_jwt, chat_base

    def _forget_auth(self) -> None:
        with self._auth_lock:
            self._auth = None

    def _assign(self, router_uid: str, cascade_id: str, prompts: list[dict[str, Any]], user_jwt: str, chat_base: str,
                timeout: Any) -> dict[str, Any]:
        """Resolve a router into ``{assignment_jwt, model_uid}``. A router uid is never a legal chat uid,
        so a failed assignment fails the turn instead of sending the router id."""
        router_prompt = next(({**p, "message_id": ""} for p in reversed(prompts) if p["source"] == wire.SOURCE_USER), None)
        decoded = self._unary("AssignModel", chat_base + wire.ASSIGN_MODEL_PATH, wire.ASSIGN_MODEL_REQUEST, {
            "metadata": wire.cli_metadata(self.api_key, user_jwt), "model_router_uid": router_uid,
            "cascade_id": cascade_id, "chat_message_prompt": router_prompt}, wire.ASSIGN_MODEL_RESPONSE, timeout)
        assignment = decoded.get("assignment") or {}
        if not assignment.get("assignment_jwt") or not assignment.get("model_uid"):
            raise DevinAPIError("Devin AssignModel returned no assignment JWT and model uid", code="devin_bad_response",
                                status_code=502)
        return assignment

    # ── chat ──

    def _create_chat_completion(
        self, *, model: str = "", messages: Optional[list[dict[str, Any]]] = None, stream: bool = False, tools: Any = None,
        tool_choice: Any = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None, top_p: Optional[float] = None, stop: Any = None,
        extra_body: Optional[dict[str, Any]] = None, timeout: Any = None, **_: Any,
    ) -> Any:
        extra = extra_body if isinstance(extra_body, dict) else {}
        model = str(model or "").strip()
        if not model:
            raise DevinAPIError("Devin request has no model", code="devin_no_model", status_code=400)
        cascade_id = str(extra.get("devin_cascade_id") or "") or str(uuid.uuid4())
        system_prompt, prompts = build_prompts(list(messages or ()), cascade_id, model)
        known = catalog.cached_model(model)
        request: dict[str, Any] = {
            "prompt": system_prompt, "chat_message_prompts": prompts, "request_type": wire.REQUEST_TYPE_CASCADE,
            "planner_mode": wire.PLANNER_MODE_DEFAULT, "tool_choice": _tool_choice(tool_choice),
            "system_prompt_cache_options": {"type": wire.CACHE_CONTROL_EPHEMERAL},
            "disable_parallel_tool_calls": not (known.supports_parallel_tool_calls if known else False),
            "cascade_id": cascade_id,
            "configuration": {
                "num_completions": 1,
                "max_tokens": int(max_tokens or max_completion_tokens or (known.max_output_tokens if known else 0)
                                  or catalog.DEFAULT_MAX_OUTPUT_TOKENS),
                "max_newlines": 400, "temperature": 1.0 if temperature is None else float(temperature), "top_k": 40,
                # The CLI sends top_p as a float32 widened to double.
                "top_p": struct.unpack("<f", struct.pack("<f", 0.95 if top_p is None else float(top_p)))[0],
                "stop_patterns": [stop] if isinstance(stop, str) else list(stop or ()),
            },
        }
        turn = self._stream(model, request, tools, timeout)
        if stream:
            return _chunks_only(turn)
        state = None
        for item in turn:
            if isinstance(item, _Turn):
                state = item
        return state.completion() if state is not None else _Turn(model).completion()

    def _stream(self, model: str, request: dict[str, Any], tools: Any, timeout: Any) -> Iterator[Any]:
        """Chunks of one turn, then the finished ``_Turn`` (read by the non-streaming path only)."""
        user_jwt, chat_base = self._user_jwt(timeout)
        chat_uid = model
        if catalog.is_router(model):
            assignment = self._assign(model, request["cascade_id"], request["chat_message_prompts"], user_jwt, chat_base,
                                      timeout)
            chat_uid = assignment["model_uid"]
            request["model_assignment_jwt"] = assignment["assignment_jwt"]
        request.update(metadata=wire.cli_metadata(self.api_key, user_jwt), chat_model_uid=chat_uid,
                       execution_id=str(uuid.uuid4()),
                       tools=build_tools(tools, gemini=_uses_gemini_schema(model, chat_uid)))
        body = wire.encode(wire.GET_CHAT_MESSAGE_REQUEST, request)
        compress = len(body) >= _MIN_COMPRESSED_BYTES
        headers = {"Content-Type": "application/connect+proto", "Connect-Protocol-Version": "1",
                   "Accept-Encoding": "identity", "Connect-Accept-Encoding": "gzip", "User-Agent": wire.USER_AGENT,
                   **({"Connect-Content-Encoding": "gzip"} if compress else {}), **self._default_headers}
        payload = wire.frame(gzip.compress(body) if compress else body, wire.FLAG_COMPRESSED if compress else 0)
        logger.debug("devin: chat request model=%s uid=%s bytes=%d tools=%d", model, chat_uid, len(body),
                     len(request["tools"]))
        turn = _Turn(model)
        kwargs = {"timeout": timeout} if timeout is not None else {}
        try:
            with self._http.stream("POST", chat_base + wire.GET_CHAT_MESSAGE_PATH, content=payload, headers=headers,
                                   **kwargs) as response:
                if response.status_code != 200:
                    error = http_error("API", response, response.read())
                    if response.status_code in (401, 403):
                        self._forget_auth()
                    raise error
                decoder = wire.FrameDecoder()
                for data in response.iter_bytes():
                    for flags, frame_payload in decoder.feed(data):
                        if flags & wire.FLAG_END_STREAM:
                            error = trailer_error(frame_payload.decode("utf-8", "replace"))
                            if error is not None:
                                self._raise_trailer(error, request, produced=bool(turn.text or turn.thinking or turn.calls))
                            continue
                        yield from turn.feed(wire.decode(wire.GET_CHAT_MESSAGE_RESPONSE, frame_payload))
        except (httpx.HTTPError, wire.ProtoDecodeError, wire.FrameTooLarge, OSError) as exc:
            raise DevinAPIError(f"Devin streaming request failed: {type(exc).__name__}: {exc}",
                                code="devin_stream_error") from exc
        yield turn.finish_chunk()
        yield turn

    def _raise_trailer(self, error: DevinAPIError, request: dict[str, Any], *, produced: bool) -> None:
        code = str(error.details.get("connect_code") or "").lower()
        if code == "unauthenticated":
            self._forget_auth()
        if not produced and code == "invalid_argument" and re.search(r"\binternal error\b", str(error.details.get("message")),
                                                                      re.IGNORECASE):
            # Only the history can be compacted: the active user turn and the system prompt/tools stay.
            history = request["chat_message_prompts"]
            if history and history[-1]["source"] == wire.SOURCE_USER:
                history = history[:-1]
            history_bytes = len(wire.encode(wire.GET_CHAT_MESSAGE_REQUEST, {"chat_message_prompts": history}))
            error.details["large_history"] = history_bytes >= _LARGE_HISTORY_BYTES
            error.details["history_bytes"] = history_bytes
        logger.warning("devin: stream rejected via Connect trailer (code=%s)", code or "?")
        raise error


def _chunks_only(events: Iterator[Any]) -> Iterator[DevinStreamChunk]:
    """The streaming view of a turn; closing it closes the underlying HTTP stream."""
    try:
        for item in events:
            if isinstance(item, DevinStreamChunk):
                yield item
    finally:
        events.close()


class AsyncDevinClient:
    """Async twin for auxiliary callers: the sync client run on a worker thread."""

    def __init__(self, sync_client: DevinClient) -> None:
        self._sync = self._real_client = sync_client
        self.api_key, self.base_url = sync_client.api_key, sync_client.base_url
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        result = await asyncio.to_thread(self._sync.chat.completions.create, **kwargs)
        return self._async_stream(result) if kwargs.get("stream") else result

    async def _async_stream(self, iterator: Iterator[Any]) -> Any:
        while True:
            item = await asyncio.to_thread(next, iterator, _END)
            if item is _END:
                return
            yield item

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)
