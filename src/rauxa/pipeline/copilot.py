"""Non-interactive, tools-disabled Copilot processes. No model calls during preflight."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .cli_runner import AttemptResult
from .config import BackendConfig, CopilotSettings
from .llm import BackendUnavailable, LLMError, validate_agent_answer


FLAGS = (
    "--silent", "--no-ask-user", "--available-tools=", "--disable-builtin-mcps",
    "--no-custom-instructions", "--no-auto-update", "--no-bash-env",
    "--no-remote", "--no-remote-export", "--output-format=text",
)
TOKEN_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
BENIGN_SETTINGS = {"bashEnv": False}
MAX_OUTPUT = 8 * 1024 * 1024


def _read_profile_json(path: Path) -> Any:
    lines = path.read_text(encoding="utf-8").splitlines()
    # Copilot CLI prefixes its managed JSON files with `//` comment lines.
    while lines and lines[0].lstrip().startswith("//"):
        lines.pop(0)
    return json.loads("\n".join(lines))


def _benign_settings(path: Path) -> bool:
    try:
        settings = _read_profile_json(path)
    except (ValueError, OSError):
        return False
    return isinstance(settings, dict) and all(
        key in BENIGN_SETTINGS and value is BENIGN_SETTINGS[key] for key, value in settings.items()
    )


class CopilotProcess:
    def __init__(self, settings: CopilotSettings, lock_fd: int):
        if os.name != "posix":
            raise BackendUnavailable("Copilot dispatch must run inside WSL/Linux, not a Windows Python process")
        self.executable = settings.executable()
        self.lock_fd = lock_fd
        self.version = ""

    @staticmethod
    def _environment(directory: Path) -> dict[str, str]:
        env = dict(os.environ)
        for key in list(env):
            if key in ("COPILOT_ALLOW_ALL", "COPILOT_PROVIDERS_CONFIG", "NODE_OPTIONS") or key.startswith("COPILOT_PROVIDER_"):
                del env[key]
        env.update(COPILOT_AUTO_UPDATE="false", GITHUB_COPILOT_PROMPT_MODE_EXTENSIONS="false")
        if any(env.get(key) for key in TOKEN_VARS):
            home = directory / "profile"
            home.mkdir()
        else:
            # Keep keychain/login access without importing global customizations. Users
            # with customized profiles can use a token or a separate, login-only home.
            home = Path(env.get("COPILOT_HOME", str(Path.home() / ".copilot"))).resolve()
            custom = (
                "settings.json", "hooks", "extensions", "installed-plugins", "plugins",
                "mcp-config.json", "lsp-config.json", "providers.json", "agents", "skills",
                "instructions", "copilot-instructions.md",
            )
            present = [
                name for name in custom
                if (home / name).exists() and (not (home / name).is_dir() or any((home / name).iterdir()))
                and not (name == "settings.json" and _benign_settings(home / name))
            ]
            if present:
                raise BackendUnavailable(
                    "tools-disabled dispatch needs a token environment variable or a login-only COPILOT_HOME; "
                    f"customizations found: {', '.join(present)}. No profile was modified."
                )
            state = home / "config.json"
            if state.exists():
                try:
                    config = _read_profile_json(state)
                except (ValueError, OSError) as e:
                    raise BackendUnavailable("cannot verify Copilot profile; use a token or a login-only COPILOT_HOME") from e
                if not isinstance(config, dict) or any("plugin" in k.lower() and v for k, v in config.items()):
                    raise BackendUnavailable("Copilot profile has plugin state; use a token or a login-only COPILOT_HOME")
        env["COPILOT_HOME"] = str(home)
        return env

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        # Signal the owned group even if its leader exited, to reap remaining helpers.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    @staticmethod
    def _redact(text: str) -> str:
        for key in TOKEN_VARS:
            if os.environ.get(key):
                text = text.replace(os.environ[key], "[redacted]")
        return text[-2000:]

    def _execute(
        self, args: list[str], prompt: str, timeout: float, cancel: threading.Event,
        started: Callable[[int], None] = lambda _: None,
    ) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory(prefix="rauxa-copilot-") as tmp:
            directory = Path(tmp)
            env = self._environment(directory)
            with (directory / "stdout").open("w+b") as stdout, (directory / "stderr").open("w+b") as stderr:
                try:
                    process = subprocess.Popen(
                        [sys.executable, str(Path(__file__).with_name("copilot_worker.py")),
                         str(timeout), str(self.lock_fd), self.executable, *args],
                        stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                        cwd=directory, env=env, start_new_session=True, pass_fds=(self.lock_fd,),
                    )
                except OSError as e:
                    raise BackendUnavailable(f"cannot launch Copilot executable: {e}") from e
                try:
                    started(process.pid)
                    deadline = time.monotonic() + timeout
                    data = prompt.encode("utf-8")
                    while True:
                        if cancel.is_set():
                            raise InterruptedError("Copilot dispatch interrupted")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"Copilot exceeded the {timeout:g}s attempt timeout")
                        if os.fstat(stdout.fileno()).st_size > MAX_OUTPUT or os.fstat(stderr.fileno()).st_size > MAX_OUTPUT:
                            raise LLMError("Copilot output exceeded the 8 MiB limit")
                        try:
                            process.communicate(input=data, timeout=min(0.1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            data = None
                    if os.fstat(stdout.fileno()).st_size > MAX_OUTPUT:
                        raise LLMError("Copilot output exceeded the 8 MiB limit")
                    stdout.seek(0)
                    stderr.seek(max(0, os.fstat(stderr.fileno()).st_size - 8192))
                    return process.returncode, stdout.read().decode("utf-8"), self._redact(stderr.read().decode("utf-8", errors="replace"))
                finally:
                    self._terminate(process)

    def preflight(self) -> None:
        try:
            code, help_text, diagnostic = self._execute(["--help"], "", 15, threading.Event())
            if code:
                raise BackendUnavailable(f"Copilot --help failed: {diagnostic}")
            required = [flag.split("=")[0] for flag in FLAGS] + ["--model"]
            missing = [flag for flag in required if flag not in help_text]
            if missing:
                raise BackendUnavailable(f"Copilot CLI lacks required automation controls: {', '.join(missing)}")
            code, version, diagnostic = self._execute(["--version", "--no-auto-update"], "", 15, threading.Event())
            if code or not version.strip():
                raise BackendUnavailable(f"Copilot version preflight failed: {diagnostic}")
            self.version = version.strip()[:200]
        except (TimeoutError, LLMError, UnicodeError) as e:
            raise BackendUnavailable(f"Copilot preflight failed: {e}") from e

    def run(
        self, request: dict[str, Any], backend: BackendConfig, cancel: threading.Event,
        started: Callable[[int], None] = lambda _: None,
    ) -> AttemptResult:
        prompt = (
            "Answer the conversation below as the assistant. Use only this conversation. "
            "Return exactly one JSON object on stdout, without code fences, tools, or file writes.\n\n"
            + "\n\n".join(f"## {m['role'].capitalize()}\n{m['content']}" for m in request["messages"])
        )
        try:
            code, output, diagnostic = self._execute(
                [*FLAGS, "--model", request["model"]], prompt, backend.timeout, cancel, started,
            )
            if code:
                if code == 124 and "rauxa: Copilot attempt timeout" in diagnostic:
                    return AttemptResult(category="timeout", message=f"Copilot exceeded the {backend.timeout:g}s attempt timeout")
                if code == 127 and "rauxa: cannot launch Copilot:" in diagnostic:
                    raise BackendUnavailable(diagnostic)
                if re.search(
                    r"\b(?:401|403|404)\b|not authenticated|authentication (?:failed|required)|"
                    r"no authentication|not logged in|login required|unknown model|model .{0,80}(?:not found|not available)|"
                    r"invalid model|permission denied|(?:unknown|invalid) (?:option|argument|tool)",
                    diagnostic, re.IGNORECASE,
                ):
                    raise BackendUnavailable(f"Copilot setup failed: {diagnostic}")
                return AttemptResult(category="exit", message=f"Copilot exited {code}: {diagnostic}")
            value = json.loads(output)
            return AttemptResult(answer=validate_agent_answer(value, request["kind"]))
        except TimeoutError as e:
            return AttemptResult(category="timeout", message=str(e))
        except (ValueError, LLMError, UnicodeError) as e:
            return AttemptResult(category="invalid_response", message=str(e))
