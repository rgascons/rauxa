#!/usr/bin/env python3
"""A local-only CLI fixture: never contacts a model or a network service."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time


FLAGS = (
    "--silent --no-ask-user --available-tools --disable-builtin-mcps --no-custom-instructions "
    "--no-auto-update --no-bash-env --sync-session --output-format --model"
)
if "--help" in sys.argv:
    print("" if os.environ.get("RAUXA_FAKE_MODE") == "old_cli" else FLAGS)
    sys.exit(0)
if "--version" in sys.argv:
    print("fake-copilot 1.0")
    sys.exit(0)

prompt = sys.stdin.read()
model = sys.argv[sys.argv.index("--model") + 1]
mode = os.environ.get("RAUXA_FAKE_MODE", "normal")
role = "judge" if "strict reviewer" in prompt else ("fix" if "did not pass our checks" in prompt else "write")
child = None
if mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
log = os.environ.get("RAUXA_FAKE_LOG")
if log:
    import fcntl
    with open(log, "a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({
            "pid": os.getpid(), "child": child.pid if child else None, "args": sys.argv[1:],
            "model": model, "role": role, "cwd": os.getcwd(), "home": os.environ["COPILOT_HOME"],
            "time": time.monotonic(),
        }) + "\n")
        stream.flush()
if mode in ("sleep", "child"):
    time.sleep(60)
if mode == "invalid":
    print("not JSON")
    sys.exit(0)
if mode in ("exit", "auth"):
    print("authentication required" if mode == "auth" else "worker error", file=sys.stderr)
    sys.exit(7)
if role == "judge":
    answer = {
        "good_is_correct": True, "variants_are_correct": True, "bad_has_stated_error": True,
        "bad_has_other_errors": False, "natural": mode != "negative", "same_meaning": True, "comment": "",
    }
elif role == "fix":
    good = re.search(r'"good": "([^"]+)"', prompt).group(1).replace("de acabar", "d'acabar")
    bad = re.search(r'"bad": "([^"]+)"', prompt).group(1)
    answer = {"good": good, "bad": bad, "alternatives": []}
else:
    n = int(re.search(r"Write (\d+) new pairs", prompt).group(1))
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    items = []
    for i in range(n):
        verb = "acabar" if mode == "repair" else "treballar"
        items.append({
            "good": f"El grup {seed + i} ha de {verb}.",
            "bad": f"El grup {seed + i} té que {verb}.",
            "alternatives": [],
        })
    answer = {"items": items}
text = json.dumps(answer, ensure_ascii=False)
if mode == "fenced":
    # Models often wrap JSON in a Markdown fence despite being told not to.
    text = f"Here is the JSON:\n```json\n{text}\n```"
print(text)
