"""Devin (Cognition) wire protocol: Codeium Cascade's Connect RPCs carrying protobuf.

The backend is the Windsurf/Codeium API server (``server.codeium.com``). Unary RPCs (``GetUserJwt``,
``AssignModel``, ``GetCliModelConfigs``) POST a bare protobuf body (``application/proto``); the chat RPC
``GetChatMessage`` is Connect server-streaming (``application/connect+proto``): every message rides in a
5-byte envelope (flag byte + big-endian length) and the stream ends with a JSON trailer frame.

Only the fields Hermes reads or writes are declared; unknown fields are skipped on decode. Field numbers
come from the vendored Codeium descriptors (``exa.*_pb``) the oh-my-pi Devin provider generates from.
Stdlib only, so provider discovery can import it without the app's HTTP stack.
"""

from __future__ import annotations

import gzip
import struct
import sys
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

DEFAULT_BASE_URL = "https://server.codeium.com"

GET_USER_JWT_PATH = "/exa.auth_pb.AuthService/GetUserJwt"
ASSIGN_MODEL_PATH = "/exa.api_server_pb.ApiServerService/AssignModel"
GET_CHAT_MESSAGE_PATH = "/exa.api_server_pb.ApiServerService/GetChatMessage"
GET_CLI_MODEL_CONFIGS_PATH = "/exa.api_server_pb.ApiServerService/GetCliModelConfigs"

# The native Devin CLI is a connect-go client; the backend sees the same request identity from Hermes.
USER_AGENT = "connect-go/1.18.1 (go1.26.3)"

# ── protobuf (proto3) codec ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class F:
    """One field: number, kind, nested schema (``message`` kind) and repetition."""

    no: int
    kind: str  # string | bytes | bool | uint | int | double | float | message | map
    schema: Optional[Mapping[str, F]] = None
    repeated: bool = False


_VARINT_KINDS = frozenset({"bool", "uint", "int"})


def _varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64  # int32/int64 negatives are 10-byte two's complement on the wire
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _key(no: int, wire_type: int) -> bytes:
    return _varint((no << 3) | wire_type)


def _is_default(value: Any) -> bool:
    return value is None or value == "" or value == b"" or value is False or (
        not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0) or value == [] or value == {}


def _encode_scalar(field: F, value: Any) -> bytes:
    if field.kind == "string":
        data = str(value).encode("utf-8")
        return _key(field.no, 2) + _varint(len(data)) + data
    if field.kind == "bytes":
        return _key(field.no, 2) + _varint(len(value)) + bytes(value)
    if field.kind in _VARINT_KINDS:
        return _key(field.no, 0) + _varint(int(value))
    if field.kind == "double":
        return _key(field.no, 1) + struct.pack("<d", float(value))
    if field.kind == "float":
        return _key(field.no, 5) + struct.pack("<f", float(value))
    if field.kind == "message":
        data = encode(field.schema or {}, value)
        return _key(field.no, 2) + _varint(len(data)) + data
    raise ValueError(f"cannot encode protobuf kind {field.kind!r}")


def encode(schema: Mapping[str, F], message: Mapping[str, Any]) -> bytes:
    """Serialize ``message`` (a dict keyed by schema names). Proto3 defaults are omitted; repeated
    varint scalars are packed. Keys outside ``schema`` are a programming error."""
    out = bytearray()
    unknown = set(message) - set(schema)
    if unknown:
        raise KeyError(f"unknown protobuf field(s): {sorted(unknown)}")
    for name, field in sorted(schema.items(), key=lambda item: item[1].no):
        value = message.get(name)
        if field.repeated:
            items = list(value or ())
            if not items:
                continue
            if field.kind in _VARINT_KINDS:
                packed = b"".join(_varint(int(item)) for item in items)
                out += _key(field.no, 2) + _varint(len(packed)) + packed
            else:
                for item in items:
                    out += _encode_scalar(field, item)
        elif field.kind == "message":
            if value is not None:
                out += _encode_scalar(field, value)
        elif not _is_default(value):
            out += _encode_scalar(field, value)
    return bytes(out)


class ProtoDecodeError(ValueError):
    """The bytes are not a valid encoding of the expected message."""


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ProtoDecodeError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise ProtoDecodeError("varint too long")


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def _scalar_from_varint(kind: str, raw: int) -> Any:
    if kind == "bool":
        return raw != 0
    return _signed(raw) if kind == "int" else raw


def decode(schema: Mapping[str, F], buf: bytes) -> dict[str, Any]:
    """Parse ``buf`` into a dict keyed by schema names. Absent fields are absent keys (read with
    ``.get``); repeated fields are lists; unknown fields are skipped; ``map`` fields are skipped."""
    by_no = {field.no: (name, field) for name, field in schema.items()}
    out: dict[str, Any] = {}
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        no, wire_type = key >> 3, key & 7
        if wire_type == 0:
            raw, pos = _read_varint(buf, pos)
            payload: Any = raw
        elif wire_type == 1:
            if pos + 8 > end:
                raise ProtoDecodeError("truncated fixed64")
            payload, pos = buf[pos:pos + 8], pos + 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            if pos + length > end:
                raise ProtoDecodeError("truncated length-delimited field")
            payload, pos = buf[pos:pos + length], pos + length
        elif wire_type == 5:
            if pos + 4 > end:
                raise ProtoDecodeError("truncated fixed32")
            payload, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ProtoDecodeError(f"unsupported wire type {wire_type}")
        entry = by_no.get(no)
        if entry is None:
            continue
        name, field = entry
        values = _decode_field(field, wire_type, payload)
        if values is None:
            continue
        if field.repeated:
            out.setdefault(name, []).extend(values)
        else:
            out[name] = values[-1]
    return out


def _decode_field(field: F, wire_type: int, payload: Any) -> Optional[list[Any]]:
    kind = field.kind
    if kind == "map":
        return None
    if kind in _VARINT_KINDS:
        if wire_type == 0:
            return [_scalar_from_varint(kind, payload)]
        if wire_type == 2:  # packed
            values, pos = [], 0
            while pos < len(payload):
                raw, pos = _read_varint(payload, pos)
                values.append(_scalar_from_varint(kind, raw))
            return values
        raise ProtoDecodeError(f"wire type {wire_type} for varint field {field.no}")
    if kind == "double" and wire_type == 1:
        return [struct.unpack("<d", payload)[0]]
    if kind == "float" and wire_type == 5:
        return [struct.unpack("<f", payload)[0]]
    if wire_type != 2:
        raise ProtoDecodeError(f"wire type {wire_type} for {kind} field {field.no}")
    if kind == "string":
        try:
            return [bytes(payload).decode("utf-8")]
        except UnicodeDecodeError as exc:
            raise ProtoDecodeError(f"invalid UTF-8 in string field {field.no}") from exc
    if kind == "bytes":
        return [bytes(payload)]
    if kind == "message":
        return [decode(field.schema or {}, bytes(payload))]
    raise ProtoDecodeError(f"cannot decode protobuf kind {kind!r}")


def decode_unary(schema: Mapping[str, F], payload: bytes) -> Optional[dict[str, Any]]:
    """Decode a unary response body. Edges return bare or gzipped protobuf, so the bare decode is
    tried first, then a gunzip; None when neither parses."""
    try:
        return decode(schema, payload)
    except ProtoDecodeError:
        pass
    try:
        return decode(schema, gzip.decompress(payload))
    except (OSError, EOFError, ProtoDecodeError):
        return None


# ── schemas (exa.*_pb) ───────────────────────────────────────────────────────────────────────────

METADATA = {
    "ide_name": F(1, "string"), "extension_version": F(2, "string"), "api_key": F(3, "string"),
    "locale": F(4, "string"), "os": F(5, "string"), "ide_version": F(7, "string"),
    "extension_name": F(12, "string"), "user_jwt": F(21, "string"), "ide_type": F(28, "string"),
    "supported_model_displays": F(30, "uint", repeated=True),
}
GET_USER_JWT_REQUEST = {"metadata": F(1, "message", METADATA)}
GET_USER_JWT_RESPONSE = {"user_jwt": F(1, "string"), "custom_api_server_url": F(2, "string")}

IMAGE_DATA = {"base64_data": F(1, "string"), "mime_type": F(2, "string")}
CHAT_TOOL_CALL = {"id": F(1, "string"), "name": F(2, "string"), "arguments_json": F(3, "string")}
PROMPT_CACHE_OPTIONS = {"type": F(1, "uint")}
CHAT_MESSAGE_PROMPT = {
    "message_id": F(1, "string"), "source": F(2, "uint"), "prompt": F(3, "string"),
    "tool_calls": F(6, "message", CHAT_TOOL_CALL, repeated=True), "tool_call_id": F(7, "string"),
    "tool_result_is_error": F(9, "bool"), "images": F(10, "message", IMAGE_DATA, repeated=True),
    "thinking": F(11, "string"), "signature": F(12, "string"), "signature_type": F(18, "string"),
}
CHAT_TOOL_DEFINITION = {
    "name": F(1, "string"), "description": F(2, "string"), "json_schema_string": F(3, "string"),
    "strict": F(12, "bool"),
}
CHAT_TOOL_CHOICE = {"option_name": F(1, "string"), "tool_name": F(2, "string")}  # oneof choice
COMPLETION_CONFIGURATION = {
    "num_completions": F(1, "uint"), "max_tokens": F(2, "uint"), "max_newlines": F(3, "uint"),
    "temperature": F(5, "double"), "top_k": F(7, "uint"), "top_p": F(8, "double"),
    "stop_patterns": F(9, "string", repeated=True),
}
GET_CHAT_MESSAGE_REQUEST = {
    "metadata": F(1, "message", METADATA), "prompt": F(2, "string"),
    "chat_message_prompts": F(3, "message", CHAT_MESSAGE_PROMPT, repeated=True),
    "request_type": F(7, "uint"), "configuration": F(8, "message", COMPLETION_CONFIGURATION),
    "tools": F(10, "message", CHAT_TOOL_DEFINITION, repeated=True), "disable_parallel_tool_calls": F(11, "bool"),
    "tool_choice": F(12, "message", CHAT_TOOL_CHOICE),
    "system_prompt_cache_options": F(13, "message", PROMPT_CACHE_OPTIONS), "cascade_id": F(16, "string"),
    "planner_mode": F(20, "uint"), "chat_model_uid": F(21, "string"), "execution_id": F(22, "string"),
    "model_assignment_jwt": F(26, "string"),
}
MODEL_USAGE_STATS = {
    "input_tokens": F(2, "uint"), "output_tokens": F(3, "uint"), "cache_write_tokens": F(4, "uint"),
    "cache_read_tokens": F(5, "uint"),
}
GET_CHAT_MESSAGE_RESPONSE = {
    "message_id": F(1, "string"), "delta_text": F(3, "string"), "stop_reason": F(5, "uint"),
    "delta_tool_calls": F(6, "message", CHAT_TOOL_CALL, repeated=True), "usage": F(7, "message", MODEL_USAGE_STATS),
    "delta_thinking": F(9, "string"), "delta_signature": F(10, "string"), "credit_cost": F(14, "int"),
    "delta_signature_type": F(21, "string"), "actual_model_uid": F(23, "string"),
}
ASSIGN_MODEL_REQUEST = {
    "metadata": F(1, "message", METADATA), "model_router_uid": F(2, "string"), "cascade_id": F(3, "string"),
    "chat_message_prompt": F(5, "message", CHAT_MESSAGE_PROMPT),
}
MODEL_ASSIGNMENT = {"assignment_jwt": F(1, "string"), "model_uid": F(2, "string")}
ASSIGN_MODEL_RESPONSE = {"assignment": F(1, "message", MODEL_ASSIGNMENT)}

MODEL_FEATURES = {
    "supports_images": F(11, "bool"), "supports_tool_calls": F(12, "bool"), "supports_thinking": F(15, "bool"),
    "supports_parallel_tool_calls": F(21, "bool"),
}
MODEL_INFO = {
    "max_tokens": F(4, "int"), "model_features": F(6, "message", MODEL_FEATURES), "max_output_tokens": F(13, "int"),
    "harness_uids": F(20, "string", repeated=True), "display_option": F(22, "uint"), "is_model_router": F(25, "bool"),
}
MODEL_DIMENSION = {"label": F(1, "string"), "value": F(2, "float"), "denominator": F(3, "string"), "kind": F(6, "uint")}
CLIENT_MODEL_CONFIG = {
    "label": F(1, "string"), "disabled": F(4, "bool"), "supports_images": F(5, "bool"), "is_beta": F(9, "bool"),
    "is_recommended": F(11, "bool"), "is_new": F(15, "bool"), "max_tokens": F(18, "int"), "model_uid": F(22, "string"),
    "model_info": F(23, "message", MODEL_INFO), "description": F(27, "string"),
    "model_dimensions": F(32, "message", MODEL_DIMENSION, repeated=True),
}
DEFAULT_OVERRIDE_MODEL_CONFIG = {"model_uid": F(3, "string")}
GET_CLI_MODEL_CONFIGS_REQUEST = {"metadata": F(1, "message", METADATA)}
GET_CLI_MODEL_CONFIGS_RESPONSE = {
    "client_model_configs": F(1, "message", CLIENT_MODEL_CONFIG, repeated=True),
    "default_override_model_config": F(3, "message", DEFAULT_OVERRIDE_MODEL_CONFIG),
}

# Enums.
SOURCE_USER, SOURCE_SYSTEM, SOURCE_TOOL = 1, 2, 4  # ChatMessageSource (SYSTEM = assistant turns)
REQUEST_TYPE_CASCADE = 5  # ChatMessageRequestType
PLANNER_MODE_DEFAULT = 1  # ConversationalPlannerMode
CACHE_CONTROL_EPHEMERAL = 1  # CacheControlType
STOP_REASON_MAX_TOKENS = 3  # StopReason
STOP_REASON_CONTENT_FILTER = 11
DISPLAY_MODEL_ROUTER, DISPLAY_QUICK_REVIEW = 3, 4  # DisplayOption
DISPLAY_INTERNAL_DEFAULT, DISPLAY_UNCLASSIFIED, DISPLAY_NORMAL = 6, 7, 8  # past the vendored descriptor
COST_KINDS = frozenset({1, 2})  # ModelDimensionKind COST / COST_FUZZY

# ── request identity ─────────────────────────────────────────────────────────────────────────────

_SESSION_TOKEN_PREFIX = "devin-session-token$"
_OS = "darwin" if sys.platform == "darwin" else "windows" if sys.platform.startswith("win") else "linux"

# Released Devin CLI identity: ``ide_type="chisel"`` is what unlocks router assignment and the CLI model
# surface for GetUserJwt / AssignModel / GetChatMessage.
_CLI_IDENTITY = {
    "ide_name": "devin-cli", "ide_type": "chisel", "ide_version": "3000.11.205", "extension_name": "chisel",
    "extension_version": "3000.11.205", "locale": "en", "os": _OS,
}
# GetCliModelConfigs answers the full catalog only to the CLI's dev-channel identity.
_DISCOVERY_IDENTITY = {
    "ide_name": "chisel", "ide_version": "0.0.0-dev", "extension_name": "chisel", "extension_version": "0.0.0-dev",
    "locale": "en", "os": _OS,
}
SUPPORTED_MODEL_DISPLAYS = (DISPLAY_MODEL_ROUTER, DISPLAY_QUICK_REVIEW, DISPLAY_INTERNAL_DEFAULT,
                            DISPLAY_UNCLASSIFIED, DISPLAY_NORMAL)


def encode_credential(api_key: Optional[str]) -> str:
    """``Metadata.api_key`` for a stored credential: a Devin login token (a bare JWT) goes out with the
    ``devin-session-token$`` prefix; an already-prefixed token and any other key (a Windsurf
    ``sk-ws-…`` key, a Devin API key) go out as stored."""
    key = (api_key or "").strip()
    if not key or key.startswith(_SESSION_TOKEN_PREFIX):
        return key
    return f"{_SESSION_TOKEN_PREFIX}{key}" if key.count(".") == 2 else key


def cli_metadata(api_key: Optional[str], user_jwt: str = "") -> dict[str, Any]:
    return {**_CLI_IDENTITY, "api_key": encode_credential(api_key), "user_jwt": user_jwt}


def discovery_metadata(api_key: Optional[str]) -> dict[str, Any]:
    return {**_DISCOVERY_IDENTITY, "api_key": encode_credential(api_key),
            "supported_model_displays": list(SUPPORTED_MODEL_DISPLAYS)}


def unary_headers() -> dict[str, str]:
    return {"Content-Type": "application/proto", "Connect-Protocol-Version": "1", "Accept": "*/*",
            "User-Agent": USER_AGENT}


# ── Connect streaming envelope ───────────────────────────────────────────────────────────────────

FLAG_COMPRESSED = 0x01
FLAG_END_STREAM = 0x02
_HEADER = struct.Struct(">BI")
# A corrupt or hostile length prefix (up to 4 GiB) must fail fast instead of being buffered.
MAX_FRAME_PAYLOAD = 16 * 1024 * 1024


def frame(payload: bytes, flags: int = 0) -> bytes:
    return _HEADER.pack(flags, len(payload)) + payload


class FrameTooLarge(ValueError):
    pass


class FrameDecoder:
    """Incremental Connect envelope decoder: ``feed`` bytes, get every frame now complete."""

    def __init__(self, max_payload: int = MAX_FRAME_PAYLOAD) -> None:
        self._buf = bytearray()
        self._max = max_payload

    def feed(self, chunk: bytes) -> Iterator[tuple[int, bytes]]:
        self._buf += chunk
        while len(self._buf) >= _HEADER.size:
            flags, length = _HEADER.unpack_from(self._buf)
            if length > self._max:
                raise FrameTooLarge(f"Connect frame length {length} exceeds the {self._max}-byte cap")
            total = _HEADER.size + length
            if len(self._buf) < total:
                return
            payload = bytes(self._buf[_HEADER.size:total])
            del self._buf[:total]
            yield flags, (gzip.decompress(payload) if flags & FLAG_COMPRESSED else payload)

    @property
    def pending(self) -> int:
        return len(self._buf)
