"""skills.always_loaded: listed skills reach EVERY agent — top-level, delegate_task children, and
agents without the skills toolset (unlike skills.auto_load, which skips the last two)."""

from __future__ import annotations

import json
import logging

import pytest

_BODY = "PINNED ALWAYS-LOADED CONTENT"
_AGENT_KW = dict(
    api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter", api_mode="chat_completions",
    model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True, skip_memory=True,
    save_trajectories=False, enabled_toolsets=["file"],  # no skills toolset on purpose
)


def _write_skill(root, name, body):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test.\n---\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Profile home with one installed skill; ``home(always_loaded)`` writes the config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_skill(tmp_path / "skills", "pinned-skill", _BODY)
    import agent.agent_init as agent_init
    monkeypatch.setattr(agent_init, "_warned_missing_always_loaded", set())

    def configure(always_loaded=None):
        cfg = {} if always_loaded is None else {"skills": {"always_loaded": always_loaded}}
        (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")  # JSON is valid YAML

    return configure


def _agent(**kw):
    from run_agent import AIAgent
    return AIAgent(**{**_AGENT_KW, **kw})


class TestResolveAlwaysLoaded:
    @pytest.mark.parametrize("raw,expected", [
        (["pinned-skill", " pinned-skill ", "other", ""], ["pinned-skill", "other"]),
        ("pinned-skill, other\nthird", ["pinned-skill", "other", "third"]),
        ([], []),
        (None, []),
        ({"pinned-skill": True}, []),
    ])
    def test_list_or_string_deduplicated(self, raw, expected):
        from agent.skill_commands import resolve_always_loaded_skills
        cfg = {"skills": {"always_loaded": raw}} if raw is not None else {"skills": {}}
        assert resolve_always_loaded_skills(cfg) == expected

    def test_missing_skills_block_is_noop(self):
        from agent.skill_commands import build_always_loaded_prompt, resolve_always_loaded_skills
        assert resolve_always_loaded_skills({}) == []
        assert build_always_loaded_prompt(task_id="s", user_config={}) == ("", [], [])

    def test_unknown_name_is_reported_and_known_still_loads(self, home):
        from agent.skill_commands import build_always_loaded_prompt
        prompt, loaded, missing = build_always_loaded_prompt(
            task_id="s", user_config={"skills": {"always_loaded": ["pinned-skill", "no-such-skill"]}})
        assert loaded == ["pinned-skill"]
        assert missing == ["no-such-skill"]
        assert prompt.count(_BODY) == 1


class TestEveryAgentGetsIt:
    def test_top_level_agent_without_skills_toolset(self, home):
        home(["pinned-skill"])
        agent = _agent(ephemeral_system_prompt="CALLER PROMPT")
        try:
            prompt = agent.ephemeral_system_prompt
            assert prompt.count(_BODY) == 1
            # Skill blocks lead; the caller's own ephemeral prompt is kept after them.
            assert prompt.index(_BODY) < prompt.index("CALLER PROMPT") and prompt.endswith("CALLER PROMPT")
        finally:
            agent.close()

    def test_delegate_child_gets_it(self, home, monkeypatch):
        home(["pinned-skill"])
        from tools import delegate_tool as dt
        import tools.delegate_tool_config as dtc
        monkeypatch.setattr(dt, "_load_config", dict)
        monkeypatch.setattr(dtc, "_load_config", dict)
        monkeypatch.setattr(dt, "_build_child_system_prompt", lambda *_a, **_kw: "DELEGATE CHILD PROMPT")
        parent = _agent(session_id="parent")
        try:
            child = dt._build_child_agent(task_index=0, goal="CHILD GOAL", context=None, toolsets=["file"],
                                          model=None, max_iterations=4, task_count=1, parent_agent=parent)
            try:
                # The child skips context files (auto_load's gate) yet still carries the skill, once,
                # ahead of its delegate prompt.
                assert child.skip_context_files is True
                assert child.ephemeral_system_prompt.count(_BODY) == 1
                assert child.ephemeral_system_prompt.endswith("DELEGATE CHILD PROMPT")
            finally:
                child.close()
        finally:
            parent.close()

    def test_inherited_prompt_is_not_duplicated(self, home):
        """A fork built from the parent's ephemeral prompt (background review) already carries the block."""
        home(["pinned-skill"])
        parent = _agent()
        fork = _agent(ephemeral_system_prompt=parent.ephemeral_system_prompt)
        try:
            assert fork.ephemeral_system_prompt.count(_BODY) == 1
        finally:
            fork.close()
            parent.close()

    @pytest.mark.parametrize("always_loaded", [None, []])
    def test_empty_or_missing_list_is_noop(self, home, always_loaded):
        home(always_loaded)
        agent = _agent()
        bare = _agent(ephemeral_system_prompt="CALLER PROMPT")
        try:
            assert not agent.ephemeral_system_prompt
            assert bare.ephemeral_system_prompt == "CALLER PROMPT"
        finally:
            agent.close()
            bare.close()

    def test_unknown_skill_never_blocks_start_and_warns_once(self, home, caplog):
        home(["no-such-skill", "pinned-skill"])
        with caplog.at_level(logging.WARNING, logger="run_agent"):
            first, second = _agent(), _agent()
        try:
            assert first.ephemeral_system_prompt.count(_BODY) == 1
            assert second.ephemeral_system_prompt.count(_BODY) == 1
            warned = [r.getMessage() for r in caplog.records if "skills.always_loaded" in r.getMessage()]
            assert len(warned) == 1 and "no-such-skill" in warned[0]
        finally:
            first.close()
            second.close()
