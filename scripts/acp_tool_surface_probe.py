#!/usr/bin/env python3
"""Probe Hermes tool-surface behavior through a baseline model and Devin ACP.

This is an intentionally live diagnostic script, not a unit test. It sends a
fixed prompt through Hermes with:

1. openai-codex/gpt-5.4-mini as the reference backend
2. devin-acp/swe-1.6 as the ACP backend under test

It records Hermes tool-progress callbacks, response text, and Devin CLI logs
created during the run, then emits a JSON report with heuristic pass/fail
signals for whether Hermes tools were surfaced and executed cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_repo_venv_python() -> None:
    """Allow direct `python scripts/...` runs from shells without venv active."""
    if os.environ.get("HERMES_ACP_PROBE_VENV_REEXEC") == "1":
        return
    for venv_name in (".venv", "venv"):
        python_bin = REPO_ROOT / venv_name / "bin" / "python"
        if python_bin.exists() and Path(sys.executable).resolve() != python_bin.resolve():
            env = dict(os.environ)
            env["HERMES_ACP_PROBE_VENV_REEXEC"] = "1"
            os.execve(str(python_bin), [str(python_bin), *sys.argv], env)


_ensure_repo_venv_python()

from hermes_constants import get_hermes_home  # noqa: E402
from run_agent import AIAgent  # noqa: E402


DEFAULT_PROMPT = (
    "nuxt3移行関連ファイルを読んで、特にオーケストレータの状況と"
    "使用トークン数の内訳についてフォーカスして短く要約して"
)

RELEVANCE_TERMS = (
    "nuxt3",
    "移行",
    "オーケストレータ",
    "orchestrator",
    "token",
    "トークン",
    "内訳",
)

HERMES_TOOL_HINTS = (
    "read_file",
    "search",
    "search_files",
    "session_search",
    "skill",
    "skill_view",
    "skills_list",
    "terminal",
)

DEVIN_NATIVE_TOOLS = (
    "exec",
    "get_output",
    "read",
    "find_file_by_name",
    "mcp_list_servers",
    "mcp_list_tools",
)

GENERIC_HERMES_PREVIEW_RE = re.compile(r"^Calling\\s+.+\\s+from\\s+hermes$", re.IGNORECASE)
PREVIEW_DETAIL_MARKERS = (
    "/",
    "nuxt",
    "orchestrator",
    "オーケストレータ",
    "token",
    "トークン",
    "agent-token-usage",
    "skill",
)


@dataclass
class ToolEvent:
    event_type: str
    name: str
    preview: str | None = None
    args: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)


def _tool_progress_collector(events: list[ToolEvent]):
    def callback(event_type: str, name: str | None = None, preview: str | None = None, args: Any = None, **kwargs: Any) -> None:
        events.append(
            ToolEvent(
                event_type=str(event_type),
                name=str(name or ""),
                preview=preview,
                args=args,
                kwargs=dict(kwargs),
            )
        )

    return callback


def _normalise_tool_name(name: str) -> str:
    value = str(name or "").strip()
    while value.startswith("mcp__hermes__"):
        value = value[len("mcp__hermes__") :]
    return value


def _find_new_devin_logs(start_time: float) -> list[Path]:
    log_dir = Path.home() / ".local" / "share" / "devin" / "cli" / "logs"
    if not log_dir.exists():
        return []
    logs = []
    for path in log_dir.glob("devin_*.log"):
        try:
            if path.stat().st_mtime >= start_time - 2:
                logs.append(path)
        except OSError:
            continue
    return sorted(logs, key=lambda p: p.stat().st_mtime)


def _read_text(path: Path, limit_bytes: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > limit_bytes:
        data = data[-limit_bytes:]
    return data.decode("utf-8", errors="replace")


def _parse_devin_log(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    permission_tools: list[str] = []
    native_tools: list[str] = []
    hermes_tools: list[str] = []
    cancellations: list[str] = []
    failures: list[str] = []
    double_prefixed: list[str] = []

    for line in text.splitlines():
        perm = re.search(r"Permission decision for tool ([^:]+):", line)
        if perm:
            raw_name = perm.group(1).strip()
            name = _normalise_tool_name(raw_name)
            permission_tools.append(raw_name)
            if raw_name.startswith("mcp__hermes__"):
                hermes_tools.append(name)
            if raw_name.startswith("mcp__hermes__mcp__hermes__"):
                double_prefixed.append(raw_name)
            if name in DEVIN_NATIVE_TOOLS:
                native_tools.append(name)
        lower = line.lower()
        if "noninteractive" in lower or "user rejected tool permission" in lower or "cancel" in lower:
            cancellations.append(line)
        if " failed" in lower or " error" in lower or "panic" in lower or "exception" in lower:
            failures.append(line)

    return {
        "path": str(path),
        "permission_tools": permission_tools,
        "hermes_tools": sorted(set(hermes_tools)),
        "native_tools": sorted(set(native_tools)),
        "double_prefixed_tools": sorted(set(double_prefixed)),
        "cancellations": cancellations,
        "failures": failures,
    }


def _summarise_response(response: str) -> dict[str, Any]:
    lowered = response.lower()
    matched = [term for term in RELEVANCE_TERMS if term.lower() in lowered]
    noisy_markers = [
        marker
        for marker in (
            "<function_calls",
            "<invoke",
            "tool call result",
            "mcp__hermes__",
            "agent_thought",
            "reasoning_scratchpad",
        )
        if marker.lower() in lowered
    ]
    return {
        "chars": len(response),
        "matched_relevance_terms": matched,
        "noisy_markers": noisy_markers,
        "excerpt": response[:1200],
    }


def _summarise_tool_events(events: list[ToolEvent]) -> dict[str, Any]:
    names = [_normalise_tool_name(event.name) for event in events if event.name]
    hermes_like = sorted({name for name in names if any(hint in name for hint in HERMES_TOOL_HINTS)})
    native = sorted({name for name in names if name in DEVIN_NATIVE_TOOLS})
    started_hermes = [
        event
        for event in events
        if event.event_type == "tool.started"
        and any(hint in _normalise_tool_name(event.name) for hint in HERMES_TOOL_HINTS)
    ]
    generic_previews = [
        str(event.preview or "")
        for event in started_hermes
        if event.preview and GENERIC_HERMES_PREVIEW_RE.match(str(event.preview).strip())
    ]
    detailed_previews = [
        str(event.preview or "")
        for event in started_hermes
        if event.preview
        and not GENERIC_HERMES_PREVIEW_RE.match(str(event.preview).strip())
        and any(marker.lower() in str(event.preview).lower() for marker in PREVIEW_DETAIL_MARKERS)
    ]
    return {
        "count": len(events),
        "names": names,
        "hermes_like_names": hermes_like,
        "native_names": native,
        "generic_hermes_previews": generic_previews,
        "detailed_hermes_previews": detailed_previews,
        "events": [
            {
                "event_type": event.event_type,
                "name": event.name,
                "normalised_name": _normalise_tool_name(event.name),
                "preview": event.preview,
                "args": event.args,
                "kwargs": event.kwargs,
            }
            for event in events
        ],
    }


def _run_agent(
    *,
    label: str,
    provider: str,
    model: str,
    prompt: str,
    run_id: str,
    timeout_seconds: float,
    max_iterations: int,
) -> dict[str, Any]:
    events: list[ToolEvent] = []
    started = time.time()
    session_id = f"acp-tool-surface-probe-{label}-{run_id}"
    tagged_prompt = f"{prompt}\n\n[probe_run_id: {run_id}; backend: {label}]"

    agent = AIAgent(
        provider=provider,
        model=model,
        platform="discord",
        session_id=session_id,
        max_iterations=max_iterations,
        quiet_mode=True,
        tool_progress_mode="all",
        tool_progress_callback=_tool_progress_collector(events),
        skip_memory=True,
        skip_context_files=True,
        reasoning_config={"effort": "medium"} if provider == "openai-codex" else None,
    )

    result: dict[str, Any] = {}
    error: str | None = None
    try:
        result = agent.run_conversation(tagged_prompt)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    final_response = str((result or {}).get("final_response") or "")
    return {
        "label": label,
        "provider": provider,
        "model": model,
        "session_id": session_id,
        "elapsed_seconds": round(elapsed, 3),
        "error": error,
        "result_flags": {
            "completed": (result or {}).get("completed"),
            "failed": (result or {}).get("failed"),
            "api_calls": (result or {}).get("api_calls"),
            "error": (result or {}).get("error"),
        },
        "response": _summarise_response(final_response),
        "tool_progress": _summarise_tool_events(events),
        "raw_result_keys": sorted((result or {}).keys()),
        "started_at": started,
        "timeout_seconds": timeout_seconds,
    }


def _score_run(run: dict[str, Any], *, devin_logs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    response = run.get("response") or {}
    tools = run.get("tool_progress") or {}
    relevance_count = len(response.get("matched_relevance_terms") or [])
    noisy = response.get("noisy_markers") or []
    hermes_like = tools.get("hermes_like_names") or []
    native_progress = tools.get("native_names") or []
    generic_previews = tools.get("generic_hermes_previews") or []
    detailed_previews = tools.get("detailed_hermes_previews") or []
    errors = [run.get("error"), (run.get("result_flags") or {}).get("error")]
    errors = [err for err in errors if err]

    log_hermes: set[str] = set()
    log_native: set[str] = set()
    log_cancelled: list[str] = []
    log_failures: list[str] = []
    log_double_prefix: list[str] = []
    for item in devin_logs or []:
        log_hermes.update(item.get("hermes_tools") or [])
        log_native.update(item.get("native_tools") or [])
        log_cancelled.extend(item.get("cancellations") or [])
        log_failures.extend(item.get("failures") or [])
        log_double_prefix.extend(item.get("double_prefixed_tools") or [])

    return {
        "response_relevant": relevance_count >= 3,
        "response_clean": not noisy,
        "hermes_tools_observed": bool(hermes_like or log_hermes),
        "native_devin_tools_observed": bool(native_progress or log_native),
        "devin_cancelled_tool_call": bool(log_cancelled),
        "devin_failures_observed": bool(log_failures),
        "double_prefixed_hermes_tools_observed": bool(log_double_prefix),
        "errors_observed": errors,
        "passed_reference_shape": (
            relevance_count >= 3
            and not noisy
            and not errors
            and bool(hermes_like or log_hermes)
            and bool(detailed_previews)
            and not generic_previews
        ),
        "details": {
            "relevance_terms": response.get("matched_relevance_terms") or [],
            "noisy_markers": noisy,
            "tool_progress_hermes_like": hermes_like,
            "tool_progress_native": native_progress,
            "tool_progress_generic_hermes_previews": generic_previews[:20],
            "tool_progress_detailed_hermes_previews": detailed_previews[:20],
            "devin_log_hermes_tools": sorted(log_hermes),
            "devin_log_native_tools": sorted(log_native),
            "devin_log_cancellations": log_cancelled[:20],
            "devin_log_failures": log_failures[:20],
            "devin_log_double_prefixed": sorted(set(log_double_prefix)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--baseline-provider", default="openai-codex")
    parser.add_argument("--baseline-model", default="gpt-5.4-mini")
    parser.add_argument("--devin-provider", default="devin-acp")
    parser.add_argument("--devin-model", default="swe-1.6")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-devin", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-response", action="store_true")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:10]
    overall_started = time.time()
    report: dict[str, Any] = {
        "run_id": run_id,
        "prompt": args.prompt,
        "hermes_home": str(get_hermes_home()),
        "started_at": overall_started,
        "runs": {},
        "verdicts": {},
    }

    if not args.skip_baseline:
        baseline = _run_agent(
            label="baseline",
            provider=args.baseline_provider,
            model=args.baseline_model,
            prompt=args.prompt,
            run_id=run_id,
            timeout_seconds=args.timeout,
            max_iterations=args.max_iterations,
        )
        report["runs"]["baseline"] = baseline
        report["verdicts"]["baseline"] = _score_run(baseline)
        if args.print_response:
            print("\n[baseline response]\n" + baseline["response"]["excerpt"])

    devin_log_start = time.time()
    if not args.skip_devin:
        devin = _run_agent(
            label="devin",
            provider=args.devin_provider,
            model=args.devin_model,
            prompt=args.prompt,
            run_id=run_id,
            timeout_seconds=args.timeout,
            max_iterations=args.max_iterations,
        )
        devin_logs = [_parse_devin_log(path) for path in _find_new_devin_logs(devin_log_start)]
        devin["devin_logs"] = devin_logs
        report["runs"]["devin"] = devin
        report["verdicts"]["devin"] = _score_run(devin, devin_logs=devin_logs)
        if args.print_response:
            print("\n[devin response]\n" + devin["response"]["excerpt"])

    report["elapsed_seconds"] = round(time.time() - overall_started, 3)

    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
        print(str(args.output))
    else:
        print(output)

    failed = []
    for label, verdict in report["verdicts"].items():
        if not verdict.get("passed_reference_shape"):
            failed.append(label)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
