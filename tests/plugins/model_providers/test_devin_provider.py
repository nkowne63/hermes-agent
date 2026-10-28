"""Devin (Cognition) provider: Cascade wire codec, catalog normalization, the OpenAI-shaped client over a
recorded-shape fake Connect server, and the profile's auth/registration seams. No network."""

from __future__ import annotations

import asyncio
import base64
import gzip
import importlib
import json
import struct
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

import providers

providers.list_providers()  # bundled plugins import under plugins.model_providers.<name>
wire = importlib.import_module("plugins.model_providers.devin.protocol")
catalog = importlib.import_module("plugins.model_providers.devin.catalog")
transport = importlib.import_module("plugins.model_providers.devin.transport")
devin_auth = importlib.import_module("plugins.model_providers.devin.auth")
devin_plugin = importlib.import_module("plugins.model_providers.devin")


def _jwt(claims: dict) -> str:
    enc = lambda obj: base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'none'})}.{enc(claims)}.sig"


SESSION_JWT = _jwt({"sub": "user-1", "exp": int(time.time()) + 86400})
USER_JWT = _jwt({"sub": "user-1", "exp": int(time.time()) + 3600})


@pytest.fixture(autouse=True)
def _fresh_catalog():
    catalog.clear_cache()
    yield
    catalog.clear_cache()


# ── protobuf / Connect codec ─────────────────────────────────────────────────────────────────────


def test_encoding_matches_protobuf_wire_bytes():
    assert wire.encode(wire.METADATA, {"ide_name": "a"}) == b"\x0a\x01a"
    # Proto3 defaults are omitted; repeated enums are packed; field order follows field numbers.
    assert wire.encode(wire.METADATA, {"api_key": "", "os": "linux", "supported_model_displays": [3, 300]}) == (
        b"\x2a\x05linux" + b"\xf2\x01\x03\x03\xac\x02")
    assert wire.encode(wire.CHAT_TOOL_CHOICE, {"option_name": "auto"}) == b"\x0a\x04auto"
    # An empty sub-message is still present (presence matters for oneof/message fields).
    assert wire.encode(wire.GET_USER_JWT_REQUEST, {"metadata": {}}) == b"\x0a\x00"
    with pytest.raises(KeyError):
        wire.encode(wire.METADATA, {"nope": 1})


def test_decode_round_trips_and_skips_unknown_fields():
    request = {
        "metadata": wire.cli_metadata("sk-ws-key", "jwt"), "prompt": "sys", "chat_model_uid": "swe-1-6",
        "chat_message_prompts": [{"message_id": "m1", "source": wire.SOURCE_USER, "prompt": "hi",
                                  "images": [{"base64_data": "AAA", "mime_type": "image/png"}]}],
        "configuration": {"max_tokens": 64000, "temperature": 1.0, "top_p": 0.5, "stop_patterns": ["a", "b"]},
        "disable_parallel_tool_calls": True,
    }
    blob = wire.encode(wire.GET_CHAT_MESSAGE_REQUEST, request)
    # Unknown fields (a varint, a fixed32 and a length-delimited one) are skipped.
    blob += b"\xf8\x07\x05" + b"\xfd\x07\x00\x00\x80\x3f" + b"\xfa\x07\x02hi"
    decoded = wire.decode(wire.GET_CHAT_MESSAGE_REQUEST, blob)
    assert decoded["metadata"]["api_key"] == "sk-ws-key" and decoded["metadata"]["ide_type"] == "chisel"
    assert decoded["chat_message_prompts"][0]["images"] == [{"base64_data": "AAA", "mime_type": "image/png"}]
    assert decoded["configuration"]["stop_patterns"] == ["a", "b"] and decoded["configuration"]["top_p"] == 0.5
    assert decoded["disable_parallel_tool_calls"] is True
    # Signed int32 negatives are 10-byte two's complement.
    neg = wire.encode(wire.GET_CHAT_MESSAGE_RESPONSE, {"credit_cost": -2})
    assert len(neg) == 11 and wire.decode(wire.GET_CHAT_MESSAGE_RESPONSE, neg) == {"credit_cost": -2}
    with pytest.raises(wire.ProtoDecodeError):
        wire.decode(wire.GET_CHAT_MESSAGE_REQUEST, b"\x12\x05ab")


def test_unary_decode_accepts_gzipped_bodies():
    body = wire.encode(wire.GET_USER_JWT_RESPONSE, {"user_jwt": "t"})
    assert wire.decode_unary(wire.GET_USER_JWT_RESPONSE, gzip.compress(body)) == {"user_jwt": "t"}
    assert wire.decode_unary(wire.GET_USER_JWT_RESPONSE, b"\xff\xff") is None


def test_frame_decoder_joins_split_frames_and_caps_length():
    stream = wire.frame(b"abc") + wire.frame(gzip.compress(b"zipped"), wire.FLAG_COMPRESSED) + wire.frame(b"{}", 2)
    decoder, out = wire.FrameDecoder(), []
    for i in range(0, len(stream), 3):
        out.extend(decoder.feed(stream[i:i + 3]))
    assert out == [(0, b"abc"), (1, b"zipped"), (2, b"{}")] and decoder.pending == 0
    with pytest.raises(wire.FrameTooLarge):
        list(wire.FrameDecoder(max_payload=4).feed(struct.pack(">BI", 0, 5)))


def test_credentials_are_encoded_by_shape():
    assert wire.encode_credential(SESSION_JWT) == f"devin-session-token${SESSION_JWT}"
    assert wire.encode_credential(f"devin-session-token${SESSION_JWT}") == f"devin-session-token${SESSION_JWT}"
    assert wire.encode_credential(" sk-ws-01abc ") == "sk-ws-01abc"
    assert wire.encode_credential(None) == ""
    meta = wire.discovery_metadata("k")
    assert meta["ide_name"] == "chisel" and meta["supported_model_displays"] == [3, 4, 6, 7, 8]


# ── catalog ──────────────────────────────────────────────────────────────────────────────────────


def _catalog_response() -> bytes:
    cost = lambda label, value: {"label": label, "value": value, "denominator": "1M tokens", "kind": 1}
    configs = [
        {"label": "SWE-1.6", "model_uid": "swe-1-6", "model_info": {
            "max_tokens": 200000, "max_output_tokens": 128000,
            "model_features": {"supports_tool_calls": True, "supports_thinking": True,
                               "supports_parallel_tool_calls": True}}},
        {"label": "Claude", "model_uid": "claude-x", "model_dimensions": [
            cost("Input", 3.0), cost("Cached Input", 0.3), cost("Output", 15.0), {"label": "Sidekick", "kind": 0},
            cost("Input", 99.0)],
         "model_info": {"max_tokens": 1_000_000,
                        "model_features": {"supports_tool_calls": True, "supports_images": True}}},
        {"label": "Adaptive", "model_uid": "adaptive", "model_info": {"display_option": wire.DISPLAY_MODEL_ROUTER}},
        {"label": "Fusion", "model_uid": "fusion", "model_info": {"is_model_router": True, "harness_uids": ["h"]}},
        {"label": "Pairing", "model_uid": "fusion-swe-1-6-sidekick-claude-x"},
        {"label": "Off", "model_uid": "off-model", "disabled": True},
        {"label": "Review", "model_uid": "review", "model_info": {"display_option": wire.DISPLAY_QUICK_REVIEW}},
        {"label": "Dup", "model_uid": "swe-1-6"},
    ]
    return wire.encode(wire.GET_CLI_MODEL_CONFIGS_RESPONSE, {
        "client_model_configs": configs, "default_override_model_config": {"model_uid": "swe-1-6"}})


def test_catalog_normalizes_the_account_roster():
    models = catalog.normalize(wire.decode(wire.GET_CLI_MODEL_CONFIGS_RESPONSE, _catalog_response()))
    assert [m.uid for m in models] == ["swe-1-6", "adaptive", "claude-x", "fusion"]
    swe, adaptive, claude, fusion = models
    assert swe.is_default and swe.context_window == 200000 and swe.max_output_tokens == 128000
    assert swe.supports_parallel_tool_calls and swe.supports_thinking and not swe.is_router
    assert adaptive.is_router and adaptive.supports_tools and adaptive.context_window == catalog.DEFAULT_CONTEXT_WINDOW
    assert not fusion.is_router  # harness-backed composite: an ordinary chat uid
    assert claude.supports_images and claude.cost_per_million == {"input": 3.0, "output": 15.0, "cache_read": 0.3}


def test_catalog_fetch_uses_discovery_identity_and_caches_routers():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_catalog_response(), headers={"content-type": "application/proto"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        models = catalog.fetch(SESSION_JWT, "https://server.example/", http_client=client)
    assert [m.uid for m in models][:1] == ["swe-1-6"]
    assert str(seen[0].url) == "https://server.example/exa.api_server_pb.ApiServerService/GetCliModelConfigs"
    assert seen[0].headers["content-type"] == "application/proto"
    meta = wire.decode(wire.GET_CLI_MODEL_CONFIGS_REQUEST, seen[0].content)["metadata"]
    assert meta["ide_name"] == "chisel" and meta["api_key"] == f"devin-session-token${SESSION_JWT}"
    assert catalog.is_router("adaptive") and not catalog.is_router("swe-1-6")
    assert catalog.cached_model("claude-x").supports_images

    empty = wire.encode(wire.GET_CLI_MODEL_CONFIGS_RESPONSE, {})
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=empty))) as client:
        assert catalog.fetch("k", "https://server.example", http_client=client) is None
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401))) as client:
        assert catalog.fetch("k", "https://server.example", http_client=client) is None


# ── fake Cascade server ──────────────────────────────────────────────────────────────────────────


def _frames(*messages: dict, trailer: dict | None = None) -> bytes:
    out = b"".join(wire.frame(wire.encode(wire.GET_CHAT_MESSAGE_RESPONSE, m)) for m in messages)
    return out + wire.frame(json.dumps(trailer or {}).encode(), wire.FLAG_END_STREAM)


class FakeCascade:
    def __init__(self, chat_body: bytes, *, chat_status: int = 200) -> None:
        self.chat_body, self.chat_status = chat_body, chat_status
        self.calls: list[tuple[str, httpx.Request]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((path, request))
        if path == wire.GET_USER_JWT_PATH:
            body = wire.encode(wire.GET_USER_JWT_RESPONSE, {"user_jwt": USER_JWT,
                                                            "custom_api_server_url": "https://chat.example/"})
            return httpx.Response(200, content=body)
        if path == wire.ASSIGN_MODEL_PATH:
            body = wire.encode(wire.ASSIGN_MODEL_RESPONSE, {"assignment": {"assignment_jwt": "assign-jwt",
                                                                           "model_uid": "MODEL_GOOGLE_GEMINI_3"}})
            return httpx.Response(200, content=body)
        if path == wire.GET_CHAT_MESSAGE_PATH:
            if self.chat_status != 200:
                return httpx.Response(self.chat_status, json={"code": "unauthenticated", "message": "bad token"})
            return httpx.Response(200, content=self.chat_body, headers={"content-type": "application/connect+proto"})
        return httpx.Response(404)

    def chat_request(self, n: int = -1) -> dict:
        chats = [r for p, r in self.calls if p == wire.GET_CHAT_MESSAGE_PATH]
        raw = chats[n].content
        flags, length = struct.unpack(">BI", raw[:5])
        payload = raw[5:5 + length]
        return wire.decode(wire.GET_CHAT_MESSAGE_REQUEST, gzip.decompress(payload) if flags & 1 else payload)

    def paths(self) -> list[str]:
        return [p for p, _ in self.calls]


def _client(fake: FakeCascade, api_key: str = SESSION_JWT):
    return transport.DevinClient(api_key=api_key, http_client=httpx.Client(transport=httpx.MockTransport(fake)))


HISTORY = [
    {"role": "system", "content": "You are a test agent."},
    {"role": "user", "content": [{"type": "text", "text": "look "},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}]},
    {"role": "assistant", "content": "", "reasoning_content": "thinking...",
     "tool_calls": [{"id": "call_1|fc_item", "type": "function", "function": {"name": "terminal",
                                                                             "arguments": "{\"command\": \"ls\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1|fc_item", "content": "a.txt"},
    {"role": "user", "content": "thanks"},
]
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command.", "parameters": {
    "type": "object", "properties": {"command": {"type": ["string", "null"]}}, "required": ["command"]}}}]

TOOL_TURN = _frames(
    {"message_id": "bot-1", "delta_thinking": "plan", "delta_signature": "sig-1"},
    {"delta_text": "Running "},
    {"delta_tool_calls": [{"id": "tc1", "name": "terminal", "arguments_json": "{\"comm"}]},
    {"delta_tool_calls": [{"arguments_json": "and\": \"pwd\"}"}]},
    {"stop_reason": 10, "actual_model_uid": "swe-1-6",
     "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 50, "cache_write_tokens": 5}},
)


def test_streaming_turn_maps_history_and_streams_openai_chunks():
    fake = FakeCascade(TOOL_TURN)
    chunks = list(_client(fake).chat.completions.create(
        model="swe-1-6", messages=HISTORY, tools=TOOLS, stream=True, extra_body={"devin_cascade_id": "cascade-1"}))

    assert fake.paths() == [wire.GET_USER_JWT_PATH, wire.GET_CHAT_MESSAGE_PATH]
    chat_url = next(r for p, r in fake.calls if p == wire.GET_CHAT_MESSAGE_PATH).url
    assert str(chat_url).startswith("https://chat.example/")  # GetUserJwt's custom_api_server_url wins
    auth_meta = wire.decode(wire.GET_USER_JWT_REQUEST, fake.calls[0][1].content)["metadata"]
    assert auth_meta["api_key"] == f"devin-session-token${SESSION_JWT}" and auth_meta["ide_name"] == "devin-cli"

    req = fake.chat_request()
    assert req["prompt"] == "You are a test agent." and req["chat_model_uid"] == "swe-1-6"
    assert req["metadata"]["user_jwt"] == USER_JWT and req["cascade_id"] == "cascade-1"
    assert req["request_type"] == wire.REQUEST_TYPE_CASCADE and req["tool_choice"] == {"option_name": "auto"}
    assert req["disable_parallel_tool_calls"] is True  # unknown model capability: conservative
    user, assistant, tool, last = req["chat_message_prompts"]
    assert (user["source"], user["prompt"], user["images"]) == (
        wire.SOURCE_USER, "look ", [{"base64_data": "iVBOR", "mime_type": "image/png"}])
    assert assistant["source"] == wire.SOURCE_SYSTEM and assistant["thinking"] == "thinking..."
    assert assistant["tool_calls"] == [{"id": "call_1", "name": "terminal", "arguments_json": "{\"command\": \"ls\"}"}]
    assert (tool["source"], tool["tool_call_id"], tool["prompt"]) == (wire.SOURCE_TOOL, "call_1", "a.txt")
    assert last["prompt"] == "thanks"
    # Message ids are stable for the same cascade + position.
    again = FakeCascade(TOOL_TURN)
    list(_client(again).chat.completions.create(model="swe-1-6", messages=HISTORY, stream=True,
                                                extra_body={"devin_cascade_id": "cascade-1"}))
    assert [p["message_id"] for p in again.chat_request()["chat_message_prompts"]] == [
        p["message_id"] for p in req["chat_message_prompts"]]
    (tool_def,) = req["tools"]
    assert tool_def["name"] == "terminal" and json.loads(tool_def["json_schema_string"])["required"] == ["command"]

    deltas = [c.choices[0].delta for c in chunks]
    assert "".join(d.reasoning or "" for d in deltas) == "plan"
    assert "".join(d.content or "" for d in deltas) == "Running "
    calls = [tc for d in deltas for tc in (d.tool_calls or ())]
    assert {(tc.id, tc.index, tc.function.name) for tc in calls} == {("tc1", 0, "terminal")}
    assert "".join(tc.function.arguments for tc in calls) == "{\"command\": \"pwd\"}"
    final = chunks[-1]
    assert final.choices[0].finish_reason == "tool_calls"
    assert (final.usage.prompt_tokens, final.usage.completion_tokens, final.usage.prompt_tokens_details.cached_tokens,
            final.usage.prompt_tokens_details.cache_write_tokens) == (155, 20, 50, 5)
    assert final.choices[0].delta.reasoning_details == [{
        "type": "devin.native_assistant", "model": "swe-1-6", "message_id": "bot-1", "signature": "sig-1",
        "signature_type": ""}]


def test_non_streaming_aggregates_and_replays_native_signature_for_same_model_only():
    fake = FakeCascade(TOOL_TURN)
    client = _client(fake)
    response = client.chat.completions.create(model="swe-1-6", messages=[{"role": "user", "content": "hi"}])
    message = response.choices[0].message
    assert response.choices[0].finish_reason == "tool_calls" and message.content == "Running "
    assert message.reasoning_content == "plan"
    assert [(tc.id, tc.function.name, json.loads(tc.function.arguments)) for tc in message.tool_calls] == [
        ("tc1", "terminal", {"command": "pwd"})]

    replay = {"role": "assistant", "content": "done", "reasoning_content": "plan",
              "reasoning_details": message.reasoning_details}
    convo = [{"role": "user", "content": "hi"}, replay, {"role": "user", "content": "more"}]
    client.chat.completions.create(model="swe-1-6", messages=convo)
    native = fake.chat_request()["chat_message_prompts"][1]
    assert (native["message_id"], native["signature"], native["thinking"]) == ("bot-1", "sig-1", "plan")
    client.chat.completions.create(model="claude-x", messages=convo)
    foreign = fake.chat_request()["chat_message_prompts"][1]
    assert "signature" not in foreign and foreign["message_id"] != "bot-1" and foreign["thinking"] == "plan"
    # One GetUserJwt per JWT lifetime, not per turn.
    assert fake.paths().count(wire.GET_USER_JWT_PATH) == 1


def test_router_model_is_assigned_before_chat_and_gemini_schemas_are_normalized():
    fake = FakeCascade(_frames({"delta_text": "ok", "stop_reason": 2}))
    response = _client(fake).chat.completions.create(model="adaptive", messages=HISTORY, tools=TOOLS)
    assert response.choices[0].message.content == "ok" and response.choices[0].finish_reason == "stop"
    assert fake.paths() == [wire.GET_USER_JWT_PATH, wire.ASSIGN_MODEL_PATH, wire.GET_CHAT_MESSAGE_PATH]
    assign = wire.decode(wire.ASSIGN_MODEL_REQUEST, next(r for p, r in fake.calls if p == wire.ASSIGN_MODEL_PATH).content)
    assert assign["model_router_uid"] == "adaptive" and assign["chat_message_prompt"]["prompt"] == "thanks"
    assert "message_id" not in assign["chat_message_prompt"]
    req = fake.chat_request()
    assert (req["chat_model_uid"], req["model_assignment_jwt"]) == ("MODEL_GOOGLE_GEMINI_3", "assign-jwt")
    # Gemini backends reject JSON-Schema type arrays.
    schema = json.loads(req["tools"][0]["json_schema_string"])
    assert not isinstance(schema["properties"]["command"].get("type"), list)


def test_router_turn_signature_is_bound_to_the_assigned_model():
    fake = FakeCascade(_frames({"message_id": "bot-9", "delta_thinking": "t", "delta_signature": "sig-9"},
                               {"delta_text": "ok"}))
    client = _client(fake)
    first = client.chat.completions.create(model="adaptive", messages=[{"role": "user", "content": "hi"}])
    (record,) = first.choices[0].message.reasoning_details
    assert record["model"] == "MODEL_GOOGLE_GEMINI_3" and record["signature"] == "sig-9"
    convo = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "ok", "reasoning_content": "t", "reasoning_details": [record]},
             {"role": "user", "content": "again"}]
    client.chat.completions.create(model="adaptive", messages=convo)  # assigned to the same signer again
    assert fake.chat_request()["chat_message_prompts"][1]["signature"] == "sig-9"
    client.chat.completions.create(model="swe-1-6", messages=convo)  # a different model never gets it
    assert "signature" not in fake.chat_request()["chat_message_prompts"][1]


def test_large_request_is_gzip_framed():
    fake = FakeCascade(_frames({"delta_text": "ok"}))
    _client(fake).chat.completions.create(model="swe-1-6", messages=[{"role": "user", "content": "x" * 5000}])
    chat = next(r for p, r in fake.calls if p == wire.GET_CHAT_MESSAGE_PATH)
    assert chat.headers["connect-content-encoding"] == "gzip" and chat.content[0] == wire.FLAG_COMPRESSED
    assert fake.chat_request()["chat_message_prompts"][0]["prompt"] == "x" * 5000


def test_trailer_and_http_errors_carry_classifiable_status():
    trailer = _frames(trailer={"error": {"code": "resource_exhausted", "message": "quota"}})
    with pytest.raises(transport.DevinAPIError) as excinfo:
        list(_client(FakeCascade(trailer)).chat.completions.create(model="swe-1-6", messages=HISTORY, stream=True))
    assert excinfo.value.status_code == 429 and "quota" in str(excinfo.value)

    fake = FakeCascade(b"", chat_status=401)
    client = _client(fake)
    with pytest.raises(transport.DevinAPIError) as excinfo:
        client.chat.completions.create(model="swe-1-6", messages=HISTORY)
    assert excinfo.value.status_code == 401 and "bad token" in str(excinfo.value)
    fake.chat_status = 200
    fake.chat_body = _frames({"delta_text": "ok"})
    client.chat.completions.create(model="swe-1-6", messages=HISTORY)
    assert fake.paths().count(wire.GET_USER_JWT_PATH) == 2  # a rejected JWT is not reused


def test_opaque_invalid_argument_on_large_history_classifies_as_context_overflow():
    internal = _frames(trailer={"error": {"code": "invalid_argument", "message": "an internal error occurred"}})
    big = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "y" * (600 * 1024)},
           {"role": "user", "content": "next"}]
    with pytest.raises(transport.DevinAPIError) as excinfo:
        _client(FakeCascade(internal)).chat.completions.create(model="swe-1-6", messages=big)
    assert excinfo.value.details["large_history"] is True

    from agent.error_classifier import FailoverReason, classify_api_error
    verdict = classify_api_error(excinfo.value, provider="devin", model="swe-1-6")
    assert verdict.reason == FailoverReason.context_overflow

    with pytest.raises(transport.DevinAPIError) as small:
        _client(FakeCascade(internal)).chat.completions.create(model="swe-1-6", messages=HISTORY)
    assert small.value.details["large_history"] is False and small.value.status_code == 400


def test_async_twin_streams_through_the_auxiliary_seam():
    from agent.auxiliary_client import _to_async_client

    fake = FakeCascade(_frames({"delta_text": "a"}, {"delta_text": "b"}))
    async_client, model = _to_async_client(_client(fake), "swe-1-6")
    assert isinstance(async_client, transport.AsyncDevinClient) and model == "swe-1-6"

    async def run():
        stream = await async_client.chat.completions.create(model="swe-1-6", messages=HISTORY, stream=True)
        return [c async for c in stream]

    chunks = asyncio.run(run())
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "ab"


def test_tool_call_ids_are_cascade_safe():
    assert transport.sanitize_tool_call_id("call_1|fc_2") == "call_1"
    assert transport.sanitize_tool_call_id("toolu:01/x") == "toolu_01_x"
    long_a, long_b = "a" * 60 + "1", "a" * 60 + "2"
    assert len(transport.sanitize_tool_call_id(long_a)) == 40
    assert transport.sanitize_tool_call_id(long_a) != transport.sanitize_tool_call_id(long_b)


# ── profile, registration, auth ──────────────────────────────────────────────────────────────────


def test_profile_registers_and_supplies_the_cascade_client(monkeypatch):
    import hermes_cli.auth

    profile = providers.get_provider_profile("devin")
    assert profile is not None and providers.get_provider_profile("cognition") is profile
    assert hermes_cli.auth.resolve_provider("cognition") == "devin"
    assert profile.base_url == "https://server.codeium.com" and "DEVIN_API_KEY" in profile.env_vars
    client = profile.create_client(api_key="sk-ws-1", base_url="https://server.codeium.com", default_headers={"x": "1"})
    assert isinstance(client, transport.DevinClient) and client.HERMES_SKIP_TRANSPORT_WRAP
    with pytest.raises(RuntimeError):
        profile.create_client(api_key="")
    assert profile.build_extra_body(session_id="s1") == {"devin_cascade_id": "s1"}
    assert profile.build_extra_body(session_id=None) == {}

    monkeypatch.setattr(catalog, "fetch", lambda key, base, **kw: catalog.normalize(
        wire.decode(wire.GET_CLI_MODEL_CONFIGS_RESPONSE, _catalog_response())))
    assert profile.fetch_models(api_key="k") == ["swe-1-6", "adaptive", "claude-x", "fusion"]
    assert profile.fetch_models(api_key=None) is None


def test_profile_context_length_comes_from_the_live_catalog():
    profile = providers.get_provider_profile("devin")
    assert profile.get_model_context_length("swe-1-6") == 200000  # seed, before any catalog fetch
    assert profile.get_model_context_length("claude-x") is None
    catalog.remember("k", "b", catalog.normalize(wire.decode(wire.GET_CLI_MODEL_CONFIGS_RESPONSE, _catalog_response())))
    assert profile.get_model_context_length("claude-x") == 1_000_000


def test_token_request_exchanges_code_as_json_and_reads_jwt_expiry(monkeypatch):
    from hermes_cli import auth_constants

    posts = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {"token": SESSION_JWT})

    monkeypatch.setattr(auth_constants.httpx, "post", fake_post)
    fields = devin_auth.token_request({"grant_type": "authorization_code", "code": "c0de", "code_verifier": "v",
                                       "redirect_uri": "http://127.0.0.1:59653/callback"})
    url, kwargs = posts[0]
    assert url == "https://api.devin.ai/auth/cli/token" and kwargs["json"] == {"code": "c0de", "code_verifier": "v"}
    assert kwargs["headers"]["Accept"] == "application/json"
    assert fields["access_token"] == SESSION_JWT and 86000 < fields["expires_in"] <= 86400

    from hermes_cli.auth_constants import AuthError
    with pytest.raises(AuthError) as excinfo:
        devin_auth.token_request({"grant_type": "refresh_token", "refresh_token": "x"})
    assert excinfo.value.code == "invalid_grant"


def test_browser_login_sends_no_client_id_and_pools_the_session_token(monkeypatch, tmp_path):
    from urllib.parse import parse_qs, urlparse
    from urllib.request import urlopen

    from hermes_cli import auth_oauth_pkce_plugin as pkce
    from tests.providers.fake_pkce_idp import FakeIdP

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    idp = FakeIdP().start()
    opened = []

    def browser(url):
        opened.append(url)
        with urlopen(url, timeout=5):
            return True

    grants = []
    cfg = replace(devin_plugin._LOGIN, authorize_url=f"{idp.base}/authorize", redirect_port=0, timeout_seconds=5,
                  token_request=lambda grant: grants.append(grant) or {"access_token": SESSION_JWT, "expires_in": 3600})
    monkeypatch.setattr(devin_plugin, "_pkce_handler", pkce.pkce_auth_handler(cfg))
    monkeypatch.setattr(pkce.webbrowser, "open", browser)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    try:
        assert devin_plugin._auth_handler("add", SimpleNamespace(provider="devin", no_browser=False)) is True
    finally:
        idp.stop()
    query = parse_qs(urlparse(opened[0]).query)
    assert "client_id" not in query and query["prompt"] == ["select_account"]
    assert query["code_challenge_method"] == ["S256"] and grants[0]["code_verifier"]
    rows = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"]["devin"]
    assert [(r["auth_type"], r["access_token"]) for r in rows] == [("oauth", SESSION_JWT)]
    # API-key adds and the other actions stay on the built-in path.
    assert devin_plugin._auth_handler("add", SimpleNamespace(provider="devin", auth_type="api-key")) is False
    assert devin_plugin._auth_handler("status", SimpleNamespace(provider="devin")) is False


def test_production_request_assembly_keeps_the_devin_carrier_for_devin_only():
    """The agent's request path must hand the client its own replay record (and nobody else's)."""
    from unittest.mock import patch

    from agent.turn_api_request import build_api_request
    from agent.turn_context import _PER_TURN_RESET_STATE
    from run_agent import AIAgent

    record = {"type": "devin.native_assistant", "model": "swe-1-6", "message_id": "bot-1", "signature": "s"}
    foreign = {"type": "other.native_assistant", "blob": "x"}
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok", "reasoning_content": "plan", "reasoning_details": [record, foreign]},
                {"role": "user", "content": "next"}]
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}):
        agent = AIAgent(provider="devin", model="swe-1-6", base_url="https://server.codeium.com", api_key="sk-ws-test",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, session_id="s1")
    for name, value in _PER_TURN_RESET_STATE:
        setattr(agent, name, value)
    assert isinstance(agent.client, transport.DevinClient)
    built = build_api_request(
        agent, api_messages=messages, _moa_prepared_request=None, tools_for_api=None, system_message="sys",
        messages=messages, original_user_message="next", approx_tokens=0, total_chars=0, retry_count=0,
        api_call_count=1, api_request_id="req-1", api_start_time=0.0, effective_task_id="task-1", turn_id="turn-1")
    sent = next(m for m in built.api_kwargs["messages"] if m["role"] == "assistant")
    assert sent["reasoning_details"] == [record]
    assert built.api_kwargs["extra_body"]["devin_cascade_id"] == "s1"
