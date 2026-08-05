# origin/main merge conflict resolution log - 2026-07-02

Merged `origin/main` into `nkowne63/main` and resolved conflicts by preserving
both upstream behavior and the existing fork changes.

## Resolutions

- `agent/chat_completion_helpers.py`
  - Kept `nkowne63/main`'s `current_provider` tracking so primary-provider
    cooldown is only extended when the primary actually failed.
  - Kept `origin/main`'s `FailoverReason.upstream_rate_limit` handling so
    upstream 429-style failures enter the same cooldown path.

- `agent/conversation_loop.py`
  - Kept ACP subprocess providers (`copilot-acp`, `devin-acp`, `claude-acp`,
    and `acp://`/`acp+tcp://` URLs) on non-streaming responses because they
    return plain completion objects.
  - Preserved `origin/main`'s MoA streaming support by removing `moa` from the
    ACP non-streaming exclusion and leaving the MoA consumer-gated branch below.

- `agent/copilot_acp_client.py`
  - Kept `nkowne63/main`'s generic ACP provider adapter, provider-specific
    tool-name normalization, logging, and model normalization support.
  - Kept `origin/main`'s `hermes_subprocess_env` use and typed
    `ChatCompletionMessageToolCall` construction.
  - Combined both by normalizing tool names before constructing the OpenAI
    tool-call object.

- `gateway/config.py`
  - Preserved `nkowne63/main`'s `gateway_fallback_notification` setting.
  - Preserved `origin/main`'s `typing_indicator` and `channel_overrides`
    settings.
  - Bridged all three settings from legacy platform blocks into
    `platforms.<name>` config.

- `gateway/run.py`
  - Preserved `origin/main`'s restart task tracking and per-channel
    model/provider/system prompt overrides.
  - Preserved `nkowne63/main`'s session runtime restoration from the persisted
    session database.
  - Ordered runtime resolution as global runtime, then channel override, then
    persisted session runtime, then active in-memory `/model` override, so the
    documented session-over-channel precedence is maintained.

- `hermes_cli/auth.py`
  - Preserved `origin/main`'s Codex OAuth User-Agent tag.
  - Preserved `nkowne63/main`'s one-day Codex token refresh skew.

- `hermes_cli/main.py`
  - Kept `nkowne63/main`'s `devin-acp` and `claude-acp` fallback provider
    choices.
  - Kept `origin/main`'s `vertex` provider fallback choice.

- `hermes_constants.py`
  - Preserved `origin/main`'s Windows hidden-window subprocess flags when
    probing the managed Node executable.

- `plugins/platforms/discord/adapter.py`
  - Preserved `origin/main`'s channel key matching by ID, bare name, `#name`,
    and parent channel.
  - Preserved `nkowne63/main`'s guild/category/channel effective rules for
    mention and thread behavior.
  - Combined both by checking name-aware channel keys and category-aware ID
    sets for free-response and no-thread gating.
