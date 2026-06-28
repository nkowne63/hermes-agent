from mcp_hermes_tools import _tool_aliases


def test_tool_aliases_accept_prefixed_hermes_mcp_names():
    aliases = _tool_aliases(["session_search", "skill_view"])

    assert aliases["session_search"] == "session_search"
    assert aliases["skill_view"] == "skill_view"
    assert aliases["mcp__hermes__session_search"] == "session_search"
    assert aliases["mcp__hermes__skill_view"] == "skill_view"
