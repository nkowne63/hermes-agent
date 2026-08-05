"""Local sender-triggered bridge for agmsg -> live Hermes agents.

The agmsg skill persists every message in SQLite first.  This module adds the
optional local push path: a Hermes process exposes a per-process Unix datagram
socket, registers its live parent/child agents, and ``send.sh`` sends a small
notification to the matching socket after the durable DB insert succeeds.

The bridge deliberately calls ``AIAgent.steer`` rather than mutating a
transcript or inserting a new user message.  That preserves the existing
message-role alternation and prompt-cache-safe turn boundary.  Delivery is
best-effort: the SQLite message remains the source of truth when no live target
is registered.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

_PROTOCOL = "agmsg-hermes-push-v1"
_MAX_DATAGRAM = 128 * 1024
_MAX_BODY = 100_000


def _config_value(name: str, default: Any = None) -> Any:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("agmsg") or {}
        if isinstance(section, dict) and name in section:
            return section[name]
    except Exception:
        pass
    return default


def bridge_enabled() -> bool:
    """Return whether the local push bridge is enabled for this installation."""
    raw = os.environ.get("AGMSG_HERMES_PUSH")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return bool(_config_value("enabled", False))


def runtime_dir() -> Path:
    """Return the shared runtime directory used by Hermes and agmsg scripts."""
    explicit = os.environ.get("AGMSG_RUNTIME_PATH") or _config_value("runtime_dir")
    if explicit:
        return Path(str(explicit)).expanduser()
    storage = os.environ.get("AGMSG_STORAGE_PATH")
    if storage:
        return Path(storage).expanduser() / "runtime"
    # This matches the agmsg skill's normal db location without reading its DB.
    return Path.home() / ".agents" / "skills" / "agmsg" / "db" / "runtime"


def registry_key(team: str, agent: str) -> str:
    raw = f"{team}\0{agent}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(raw).hexdigest()


def _hermes_runtime_dir(base: Optional[Path] = None) -> Path:
    path = (base or runtime_dir()) / "hermes"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    registry = path / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(registry, 0o700)
    except OSError:
        pass
    return path


def _registry_path(team: str, agent: str, base: Optional[Path] = None) -> Path:
    return _hermes_runtime_dir(base) / "registry" / f"{registry_key(team, agent)}.json"


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _socket_alive(record: Dict[str, Any]) -> bool:
    return _pid_alive(record.get("pid")) and bool(record.get("socket")) and Path(
        str(record.get("socket"))
    ).exists()


def _read_json(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            return [raw]
    except (OSError, ValueError, TypeError):
        pass
    return []


def _write_json_atomic(path: Path, value: list[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _modify_registry(
    team: str,
    agent: str,
    modifier,
    *,
    base: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Modify one registry file under a best-effort per-key file lock."""
    path = _registry_path(team, agent, base)
    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    lock_file = lock_path.open("r+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        entries = [record for record in _read_json(path) if _socket_alive(record)]
        updated = modifier(entries)
        if updated:
            _write_json_atomic(path, updated)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return updated
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        lock_file.close()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _live_registry_entries(team: str, agent: str, *, base: Optional[Path] = None) -> list[dict[str, Any]]:
    path = _registry_path(team, agent, base)
    entries = [record for record in _read_json(path) if _socket_alive(record)]
    if entries != _read_json(path):
        # Opportunistic stale-record cleanup; failure is harmless.
        try:
            _modify_registry(team, agent, lambda _old: entries, base=base)
        except Exception:
            pass
    return entries


def resolve_identity(project: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Resolve one Hermes identity from explicit config/env or agmsg whoami."""
    team = os.environ.get("AGMSG_TEAM") or _config_value("team")
    agent = os.environ.get("AGMSG_AGENT") or _config_value("agent")
    if team and agent:
        return str(team), str(agent)

    project = (
        project
        or os.environ.get("AGMSG_PROJECT")
        or _config_value("project")
        or os.getcwd()
    )
    script = Path.home() / ".agents" / "skills" / "agmsg" / "scripts" / "whoami.sh"
    if not script.is_file():
        return None
    try:
        completed = subprocess.run(
            ["bash", str(script), str(project), "hermes"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or "").strip()
    if completed.returncode != 0 or not output.startswith("agent="):
        return None
    fields: dict[str, str] = {}
    for item in output.split():
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    teams = [item for item in fields.get("teams", "").split(",") if item]
    if fields.get("agent") and len(teams) == 1:
        return fields["teams"][0], fields["agent"]
    return None


class AgmsgBridge:
    """One local push endpoint shared by all live agents in a Hermes process."""

    def __init__(self, base: Optional[Path] = None) -> None:
        self.base = Path(base).expanduser() if base else runtime_dir()
        self._runtime = _hermes_runtime_dir(self.base)
        self._socket_path = self._runtime / f"bridge-{os.getpid()}.sock"
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        # A key may legitimately exist in multiple live sessions.  Keep all
        # routes and fail closed in the receiver when the target is ambiguous;
        # silently steering the most recently registered session is unsafe.
        self._routes: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self._registered: dict[str, tuple[str, str]] = {}

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def start(self) -> bool:
        with self._lock:
            if self._socket is not None:
                return True
            try:
                self._socket_path.unlink()
            except FileNotFoundError:
                pass
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                sock.bind(str(self._socket_path))
                try:
                    os.chmod(self._socket_path, 0o600)
                except OSError:
                    pass
                sock.settimeout(0.5)
                self._socket = sock
            except OSError as exc:
                logger.debug("agmsg Hermes bridge could not start: %s", exc)
                try:
                    self._socket_path.unlink()
                except OSError:
                    pass
                return False
            self._thread = threading.Thread(
                target=self._serve,
                name="agmsg-hermes-bridge",
                daemon=True,
            )
            self._thread.start()
            return True

    def register(
        self,
        team: str,
        agent: str,
        target: Any,
        *,
        target_id: str = "",
        project: str = "",
        agent_type: str = "hermes",
    ) -> Optional[str]:
        if not team or not agent or not callable(getattr(target, "steer", None)):
            return None
        if not self.start():
            return None
        token = uuid.uuid4().hex
        key = (str(team), str(agent))
        with self._lock:
            self._routes.setdefault(key, {})[token] = {
                "token": token,
                "target": target,
                "team": key[0],
                "agent": key[1],
            }
            self._registered[token] = key
        record = {
            "protocol": _PROTOCOL,
            "route_id": token,
            "team": key[0],
            "agent": key[1],
            "type": agent_type,
            "project": project or os.getcwd(),
            "pid": os.getpid(),
            "socket": str(self._socket_path),
            "target_id": str(target_id or ""),
            "registered_at": time.time(),
        }
        _modify_registry(
            key[0],
            key[1],
            lambda old: [record] + [item for item in old if item.get("route_id") != token],
            base=self.base,
        )
        return token

    def unregister(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            key = self._registered.pop(token, None)
            if key is None:
                return
            routes = self._routes.get(key)
            if routes is not None:
                routes.pop(token, None)
                if not routes:
                    self._routes.pop(key, None)
        team, agent = key
        _modify_registry(
            team,
            agent,
            lambda old: [item for item in old if item.get("route_id") != token],
            base=self.base,
        )

    def _serve(self) -> None:
        while not self._stop.is_set():
            sock = self._socket
            if sock is None:
                return
            try:
                data = sock.recv(_MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if not isinstance(payload, dict) or payload.get("protocol") != _PROTOCOL:
                continue
            team = str(payload.get("team") or "")
            agent = str(payload.get("to") or "")
            with self._lock:
                routes = list((self._routes.get((team, agent)) or {}).values())
            # Fail closed when multiple live sessions claim the same logical
            # identity; the sender-side helper applies the same rule.
            if len(routes) != 1:
                continue
            route = routes[0]
            body = str(payload.get("body") or "")[:_MAX_BODY]
            if not body.strip():
                continue
            sender = str(payload.get("from") or "unknown")
            message = f"[agmsg team={team} from={sender}]\n{body}"
            try:
                route["target"].steer(message)
            except Exception:
                logger.debug("agmsg steer failed for %s/%s", team, agent, exc_info=True)

    def close(self) -> None:
        with self._lock:
            self._stop.set()
            tokens = list(self._registered)
        for token in tokens:
            try:
                self.unregister(token)
            except Exception:
                pass
        with self._lock:
            sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass


_BRIDGE: Optional[AgmsgBridge] = None
_BRIDGE_LOCK = threading.Lock()


def get_bridge() -> Optional[AgmsgBridge]:
    global _BRIDGE
    if not bridge_enabled():
        return None
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = AgmsgBridge()
            atexit.register(_BRIDGE.close)
        return _BRIDGE


def register_top_level_agent(agent: Any) -> Optional[dict[str, str]]:
    """Best-effort registration for a top-level CLI/gateway agent."""
    bridge = get_bridge()
    if bridge is None:
        return None
    identity = resolve_identity(getattr(agent, "_agmsg_project", None) or os.getcwd())
    if identity is None:
        return None
    team, name = identity
    token = bridge.register(
        team,
        name,
        agent,
        target_id=str(getattr(agent, "session_id", "") or ""),
        project=os.getcwd(),
    )
    if token is None:
        return None
    agent._agmsg_bridge = bridge
    agent._agmsg_bridge_token = token
    agent._agmsg_team = team
    agent._agmsg_agent = name
    return {"team": team, "agent": name, "token": token}


def register_child_agent(
    agent: Any,
    parent: Any,
    *,
    subagent_id: str,
    project: str = "",
) -> Optional[dict[str, str]]:
    """Register a child with a unique runtime identity in the parent's team."""
    bridge = getattr(parent, "_agmsg_bridge", None)
    team = getattr(parent, "_agmsg_team", None)
    parent_name = getattr(parent, "_agmsg_agent", None)
    if bridge is None or not team or not parent_name:
        return None
    child_name = f"{parent_name}-{subagent_id}"
    token = bridge.register(
        str(team),
        child_name,
        agent,
        target_id=subagent_id,
        project=project or os.getcwd(),
    )
    if token is None:
        return None
    agent._agmsg_bridge = bridge
    agent._agmsg_bridge_token = token
    agent._agmsg_team = str(team)
    agent._agmsg_agent = child_name
    agent._agmsg_parent_agent = str(parent_name)
    return {"team": str(team), "agent": child_name, "parent": str(parent_name), "token": token}


def unregister_agent(agent: Any) -> None:
    bridge = getattr(agent, "_agmsg_bridge", None)
    token = getattr(agent, "_agmsg_bridge_token", None)
    if bridge is not None and token:
        try:
            bridge.unregister(token)
        except Exception:
            logger.debug("agmsg agent unregister failed", exc_info=True)
    try:
        agent._agmsg_bridge_token = None
    except Exception:
        pass


__all__ = [
    "AgmsgBridge",
    "bridge_enabled",
    "get_bridge",
    "register_top_level_agent",
    "register_child_agent",
    "unregister_agent",
    "resolve_identity",
    "runtime_dir",
    "registry_key",
]
