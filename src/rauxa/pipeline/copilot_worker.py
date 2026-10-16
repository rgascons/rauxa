"""Subprocess guard: hold the inherited run lock until the CLI and its helpers stop."""

import os
import signal
import subprocess
import sys


def _interrupt(*_: object) -> None:
    raise InterruptedError


def main() -> int:
    timeout = float(sys.argv[1])
    lock_fd = int(sys.argv[2])
    os.fstat(lock_fd)
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    process = None
    try:
        process = subprocess.Popen(sys.argv[3:], start_new_session=True)
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("rauxa: Copilot attempt timeout", file=sys.stderr)
        return 124
    except InterruptedError:
        return 130
    except OSError as e:
        print(f"rauxa: cannot launch Copilot: {e}", file=sys.stderr)
        return 127
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
