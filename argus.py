#!/usr/bin/env python3
"""
Argus one-command launcher - works on Windows, macOS and Linux.

    python argus.py start     set everything up and run it
    python argus.py stop      stop everything
    python argus.py status    show what is running
    python argus.py doctor    check this machine can run Argus
    python argus.py reset     stop, then delete the venv / build output

`start` is idempotent: it detects what is already done and skips it, so the
second run is fast. It picks a runtime automatically:

    Docker present and healthy  ->  docker compose up
    otherwise                   ->  a local virtualenv

Everything is stdlib-only, because this script has to run BEFORE any
dependency is installed. Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".argus"
PID_FILE = STATE_DIR / "backend.pid"
MODE_FILE = STATE_DIR / "mode"
LOG_FILE = STATE_DIR / "backend.log"
ENV_FILE = ROOT / ".env"

DEFAULT_PORT = 8000
MIN_PYTHON = (3, 9)
IS_WINDOWS = os.name == "nt"

COMPOSE_FILE = ROOT / "docker-compose.yml"
REQUIREMENTS = ROOT / "requirements.txt"
FRONTEND = ROOT / "frontend"
FRONTEND_DIST = FRONTEND / "dist"
WEIGHTS = ROOT / "backend" / "models" / "yolov8n.pt"


# ── output helpers ───────────────────────────────────────────────────────────

def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if IS_WINDOWS:
        # Modern Windows Terminal / PowerShell handle ANSI; legacy cmd does not.
        return os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")
    return True


_COLOUR = bool(_supports_colour())


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def info(msg: str) -> None:
    print(f"  {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n{_c('1;36', '==>')} {_c('1', msg)}", flush=True)


def ok(msg: str) -> None:
    print(f"  {_c('32', 'OK')}   {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  {_c('33', 'WARN')} {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  {_c('31', 'FAIL')} {msg}", flush=True)


def die(msg: str, hint: str = "") -> "NoReturn":  # type: ignore[valid-type]
    fail(msg)
    if hint:
        print(f"\n       {hint}\n", flush=True)
    sys.exit(1)


# ── small utilities ──────────────────────────────────────────────────────────

def run(cmd, **kw):
    """Run a command, inheriting stdio unless capture=True."""
    capture = kw.pop("capture", False)
    kw.setdefault("cwd", str(ROOT))
    if capture:
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.STDOUT)
        kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


def have(exe: str) -> bool:
    return shutil.which(exe) is not None


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def free_port_from(start: int, tries: int = 20) -> int:
    for p in range(start, start + tries):
        if not port_busy(p):
            return p
    die(f"No free port found in {start}..{start + tries}")


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


# ── virtualenv paths ─────────────────────────────────────────────────────────

def venv_dir() -> Path:
    return ROOT / ".venv"


def venv_python() -> Path:
    d = venv_dir()
    return d / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


# ── docker detection ─────────────────────────────────────────────────────────

def docker_compose_cmd():
    """Return a working compose command, or None.

    Requires the daemon to actually respond - `docker` on PATH is not enough,
    Docker Desktop is frequently installed but not running.
    """
    if not have("docker"):
        return None
    try:
        p = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        if p.returncode != 0:
            return None
    except Exception:
        return None
    for cmd in (["docker", "compose"], ["docker-compose"]):
        try:
            p = subprocess.run(cmd + ["version"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20)
            if p.returncode == 0:
                return cmd
        except Exception:
            continue
    return None


# ── secret generation ────────────────────────────────────────────────────────

def ensure_env_file() -> None:
    """Create .env with a strong JWT secret on first run.

    Without this the server boots with an ephemeral key and every restart
    silently invalidates all issued tokens - a confusing first-run experience.
    """
    if ENV_FILE.exists():
        text = ENV_FILE.read_text(encoding="utf-8", errors="replace")
        has_secret = any(
            line.strip().startswith("ARGUS_JWT_SECRET=")
            and len(line.split("=", 1)[1].strip()) >= 32
            for line in text.splitlines()
        )
        if has_secret:
            ok(".env present with a JWT secret")
            return
        import secrets
        token = secrets.token_urlsafe(48)
        lines = [l for l in text.splitlines()
                 if not l.strip().startswith("ARGUS_JWT_SECRET=")]
        lines.append(f"ARGUS_JWT_SECRET={token}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok("Added a generated ARGUS_JWT_SECRET to .env")
        return

    import secrets
    token = secrets.token_urlsafe(48)
    sample = ROOT / ".env.example"
    body = sample.read_text(encoding="utf-8", errors="replace") if sample.exists() else ""
    if body:
        out = []
        replaced = False
        for line in body.splitlines():
            if line.strip().startswith("ARGUS_JWT_SECRET="):
                out.append(f"ARGUS_JWT_SECRET={token}")
                replaced = True
            else:
                out.append(line)
        if not replaced:
            out.append(f"ARGUS_JWT_SECRET={token}")
        body = "\n".join(out) + "\n"
    else:
        body = f"ARGUS_JWT_SECRET={token}\n"
    ENV_FILE.write_text(body, encoding="utf-8")
    ok("Created .env with a generated ARGUS_JWT_SECRET")


def load_env_file() -> dict:
    env = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            env[k.strip()] = v
    return env


# ── doctor ───────────────────────────────────────────────────────────────────

def cmd_doctor(args) -> int:
    step("Environment")
    info(f"OS        {platform.system()} {platform.release()} ({platform.machine()})")
    info(f"Python    {platform.python_version()} at {sys.executable}")
    info(f"Project   {ROOT}")

    problems = []

    step("Requirements")
    if sys.version_info >= MIN_PYTHON:
        ok(f"Python {platform.python_version()} (need >= {'.'.join(map(str, MIN_PYTHON))})")
    else:
        fail(f"Python {platform.python_version()} is too old")
        problems.append(f"Install Python {'.'.join(map(str, MIN_PYTHON))} or newer")

    compose = docker_compose_cmd()
    if compose:
        ok(f"Docker available ({' '.join(compose)}) - will use the container path")
    elif have("docker"):
        warn("Docker installed but the daemon is not responding - will use the local venv")
    else:
        info("Docker not installed - will use the local venv")

    if have("node"):
        try:
            v = run(["node", "--version"], capture=True).stdout.strip()
            ok(f"Node {v} (dashboard can be built)")
        except Exception:
            warn("Node present but not runnable")
    else:
        if FRONTEND_DIST.joinpath("index.html").is_file():
            ok("Node missing, but frontend/dist is already built")
        elif compose:
            ok("Node missing - the Docker image builds the dashboard for you")
        else:
            warn("Node not installed - the API will run, the dashboard will not be built")
            problems.append("Install Node 18+ from https://nodejs.org to get the dashboard")

    if have("git"):
        ok("git available")
    else:
        info("git not installed (only needed to update the source)")

    step("Disk and network")
    try:
        free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
        if free_gb >= 3:
            ok(f"{free_gb:.1f} GB free")
        else:
            warn(f"only {free_gb:.1f} GB free - the ML dependencies need roughly 3 GB")
            problems.append("Free up disk space")
    except Exception:
        pass

    if WEIGHTS.is_file():
        ok(f"Detection weights present ({WEIGHTS.stat().st_size / 1e6:.1f} MB)")
    else:
        info("Detection weights absent - they download automatically on first start")

    if port_busy(DEFAULT_PORT):
        warn(f"Port {DEFAULT_PORT} is in use - start will pick the next free port")
    else:
        ok(f"Port {DEFAULT_PORT} is free")

    print()
    if problems:
        fail("Not ready yet:")
        for p in problems:
            print(f"       - {p}")
        return 1
    ok("This machine can run Argus. Next:  python argus.py start")
    return 0


# ── native setup ─────────────────────────────────────────────────────────────

def ensure_venv() -> Path:
    py = venv_python()
    if py.is_file():
        ok("Virtualenv present")
        return py
    info("Creating virtualenv at .venv (one time)")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(str(venv_dir()))
    except Exception as exc:
        die(f"Could not create a virtualenv: {exc}",
            "On Debian/Ubuntu install it with:  sudo apt install python3-venv")
    py = venv_python()
    if not py.is_file():
        die("Virtualenv created but its Python is missing")
    ok("Virtualenv created")
    return py


def deps_installed(py: Path) -> bool:
    probe = "import fastapi, uvicorn, cv2, torch, ultralytics, jwt, yaml"
    p = subprocess.run([str(py), "-c", probe],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0


def install_deps(py: Path, force: bool = False) -> None:
    if not force and deps_installed(py):
        ok("Dependencies already installed")
        return
    # The ML stack needs roughly 3 GB unpacked. Failing here with pip's raw
    # ENOSPC deep in a wall of download output is hostile, so check up front.
    try:
        free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
        if free_gb < 3.0:
            die(f"Only {free_gb:.1f} GB free on the disk holding {ROOT}",
                "The dependencies (PyTorch, OpenCV, ultralytics) need about 3 GB.\n"
                "       Free up space, or move the project to a larger drive.")
    except Exception:
        free_gb = None

    # pip unpacks wheels into the system temp dir before installing them. On
    # many systems (and most containers) /tmp is a small tmpfs - torch alone
    # unpacks to well over 1 GB, so the install dies with ENOSPC even though
    # the project disk has plenty of room. Redirect temp to the project when
    # the default is too small.
    pip_env = os.environ.copy()
    try:
        import tempfile
        tmp_free = shutil.disk_usage(tempfile.gettempdir()).free / 1e9
        if tmp_free < 4.0:
            local_tmp = STATE_DIR / "tmp"
            local_tmp.mkdir(parents=True, exist_ok=True)
            pip_env["TMPDIR"] = str(local_tmp)   # POSIX
            pip_env["TEMP"] = str(local_tmp)     # Windows
            pip_env["TMP"] = str(local_tmp)
            info(f"System temp only has {tmp_free:.1f} GB free - using {local_tmp.name} instead")
    except Exception:
        pass

    info("Installing Python dependencies - this takes a few minutes the first time")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], env=pip_env)
    # CPU wheels for torch keep the download near 200 MB instead of ~2.5 GB of
    # CUDA payload that most machines cannot use anyway.
    cmd = [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS),
           "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    p = run(cmd, capture=True, env=pip_env)
    if p.returncode != 0:
        out = p.stdout or ""
        print(out[-2500:])
        low = out.lower()
        if "no space left" in low or "errno 28" in low:
            hint = ("The disk filled up while installing.\n"
                    "       Free up a few GB and run:  python argus.py start --reinstall")
        elif ("could not find a version" in low or "temporary failure in name resolution"
              in low or "network is unreachable" in low or "retries exceeded" in low):
            hint = ("Could not reach the package index - check your internet\n"
                    "       connection or proxy settings, then try again.")
        else:
            hint = "See pip's output above for the cause."
        die("Dependency installation failed", hint)
    if not deps_installed(py):
        die("Dependencies installed but a core import still fails",
            "Try:  python argus.py reset   then start again.")
    ok("Dependencies installed")


def seed_database(py: Path, env: dict) -> None:
    db = ROOT / "data" / "argus.db"
    fresh = not db.is_file()
    run([str(py), str(ROOT / "backend" / "scripts" / "init_db.py")],
        env=env, capture=True)
    p = run([str(py), str(ROOT / "backend" / "scripts" / "run_admin.py"), "--setup-only"],
            env=env, capture=True)
    if p.returncode != 0:
        warn("Admin bootstrap reported a problem:")
        print((p.stdout or "")[-500:])
    else:
        ok("Database ready" + (" (created, admin user seeded)" if fresh else ""))


def build_frontend() -> None:
    if FRONTEND_DIST.joinpath("index.html").is_file():
        ok("Dashboard already built")
        return
    if not have("npm"):
        warn("npm not found - skipping the dashboard build (the API will still run)")
        info("Install Node 18+ from https://nodejs.org, then re-run start")
        return
    info("Building the dashboard - this takes a minute the first time")
    npm = "npm.cmd" if IS_WINDOWS else "npm"
    if not FRONTEND.joinpath("node_modules").is_dir():
        p = run([npm, "install", "--no-audit", "--no-fund"], cwd=str(FRONTEND))
        if p.returncode != 0:
            warn("npm install failed - the API will run without the dashboard")
            return
    p = run([npm, "run", "build"], cwd=str(FRONTEND))
    if p.returncode != 0:
        warn("Dashboard build failed - the API will run without it")
        return
    ok("Dashboard built")


# ── process state ────────────────────────────────────────────────────────────

def save_state(mode, port, pid=None):
    STATE_DIR.mkdir(exist_ok=True)
    MODE_FILE.write_text(json.dumps({"mode": mode, "port": port}), encoding="utf-8")
    if pid is not None:
        PID_FILE.write_text(str(pid), encoding="utf-8")


def read_state() -> dict:
    if not MODE_FILE.is_file():
        return {}
    try:
        return json.loads(MODE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_pid():
    if not PID_FILE.is_file():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ── start ────────────────────────────────────────────────────────────────────

def start_docker(compose, port: int, args) -> int:
    step("Starting with Docker")
    cmd = compose + ["-f", str(COMPOSE_FILE), "up", "-d"]
    if args.rebuild:
        cmd.append("--build")
    p = run(cmd)
    if p.returncode != 0:
        die("docker compose failed to start",
            "Run 'python argus.py start --native' to use a local virtualenv instead.")
    save_state("docker", port)
    return port


def start_native(port: int, args) -> int:
    step("Preparing the local environment")
    py = ensure_venv()
    install_deps(py, force=args.reinstall)

    env = os.environ.copy()
    env.update(load_env_file())
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("PYTHONUNBUFFERED", "1")

    step("Preparing data")
    seed_database(py, env)

    step("Preparing the dashboard")
    if args.rebuild and FRONTEND_DIST.is_dir():
        shutil.rmtree(FRONTEND_DIST, ignore_errors=True)
    build_frontend()

    step("Starting Argus")
    STATE_DIR.mkdir(exist_ok=True)
    log = open(LOG_FILE, "ab")
    cmd = [str(py), "-m", "uvicorn", "backend.api.main:app",
           "--host", args.host, "--port", str(port)]

    creationflags = 0
    preexec = None
    if IS_WINDOWS:
        # New process group so we can signal the whole tree on stop.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        preexec = os.setsid

    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                            stdout=log, stderr=subprocess.STDOUT,
                            creationflags=creationflags, preexec_fn=preexec)
    save_state("native", port, proc.pid)
    info(f"Backend pid {proc.pid}, logging to {LOG_FILE.relative_to(ROOT)}")
    return port


def wait_until_up(port: int, timeout: int = 180) -> bool:
    """Poll /api/v1/health. First boot loads YOLO, so allow generous time."""
    url = f"http://127.0.0.1:{port}/api/v1/health"
    deadline = time.time() + timeout
    spin = "|/-\\"
    i = 0
    while time.time() < deadline:
        if http_ok(url, timeout=2.0):
            if _COLOUR:
                sys.stdout.write("\r")
            print("  " + _c("32", "OK") + "   Server is up" + " " * 30, flush=True)
            return True
        pid = read_pid()
        if pid and not pid_alive(pid):
            print()
            return False
        if _COLOUR:
            sys.stdout.write(f"\r  ..   waiting for the server {spin[i % 4]} ")
            sys.stdout.flush()
        i += 1
        time.sleep(1)
    print()
    return False


def cmd_start(args) -> int:
    print(_c("1;36", "\n  Argus - starting up\n"))

    if sys.version_info < MIN_PYTHON:
        die(f"Python {'.'.join(map(str, MIN_PYTHON))}+ required, found {platform.python_version()}")

    existing = read_state()
    if existing:
        prev_port = existing.get("port", DEFAULT_PORT)
        if http_ok(f"http://127.0.0.1:{prev_port}/api/v1/health"):
            ok(f"Argus is already running at http://localhost:{prev_port}")
            info("Stop it with:  python argus.py stop")
            return 0

    step("Configuration")
    ensure_env_file()

    port = args.port or DEFAULT_PORT
    if port_busy(port):
        new_port = free_port_from(port + 1)
        warn(f"Port {port} is busy - using {new_port} instead")
        port = new_port

    compose = None if args.native else docker_compose_cmd()
    if compose and not args.native:
        port = start_docker(compose, port, args)
    else:
        if args.docker:
            die("Docker was requested but the daemon is not responding",
                "Start Docker Desktop, or drop --docker to use a local virtualenv.")
        port = start_native(port, args)

    if not wait_until_up(port):
        fail("The server did not become healthy in time")
        if LOG_FILE.is_file():
            print("\n  Last lines of the log:\n")
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            for line in tail:
                print(f"    {line}")
        print(f"\n  Full log: {LOG_FILE}")
        print("  Stop with: python argus.py stop\n")
        return 1

    url = f"http://localhost:{port}"
    ui_built = FRONTEND_DIST.joinpath("index.html").is_file()

    print()
    print(_c("1;32", "  Argus is running"))
    print(f"    Dashboard   {url}" + ("" if ui_built else "   (not built - API only)"))
    print(f"    API docs    {url}/docs")
    print(f"    Login       admin / admin123")
    print()
    print(f"    Stop it     python argus.py stop")
    print()

    if not args.no_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    return 0


# ── stop ─────────────────────────────────────────────────────────────────────

def stop_native(quiet: bool = False) -> bool:
    pid = read_pid()
    if pid is None:
        return False
    if not pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return False

    if IS_WINDOWS:
        # taskkill /T ends the whole tree; uvicorn's reloader spawns children.
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Signal the process group so worker children die with the parent.
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        # Graceful first: the shutdown hook flushes state and marks cameras offline.
        for _ in range(100):
            if not pid_alive(pid):
                break
            time.sleep(0.1)
        if pid_alive(pid):
            if not quiet:
                warn("Graceful stop timed out - forcing")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    PID_FILE.unlink(missing_ok=True)
    return True


def cmd_stop(args) -> int:
    print(_c("1;36", "\n  Argus - stopping\n"))
    state = read_state()
    mode = state.get("mode")
    stopped = False

    if mode == "docker" or (mode is None and COMPOSE_FILE.is_file()):
        compose = docker_compose_cmd()
        if compose and mode == "docker":
            p = run(compose + ["-f", str(COMPOSE_FILE), "down"], capture=True)
            if p.returncode == 0:
                ok("Containers stopped")
                stopped = True
            else:
                warn("docker compose down reported a problem")

    if stop_native(quiet=True):
        ok("Backend stopped")
        stopped = True

    port = state.get("port", DEFAULT_PORT)
    if http_ok(f"http://127.0.0.1:{port}/api/v1/health", timeout=1.5):
        warn(f"Something is still answering on port {port}")
        info("It was not started by this launcher - stop it manually")
    elif not stopped:
        info("Nothing was running")
    else:
        ok("Everything is stopped")

    MODE_FILE.unlink(missing_ok=True)
    print()
    return 0


# ── status ───────────────────────────────────────────────────────────────────

def cmd_status(args) -> int:
    print(_c("1;36", "\n  Argus - status\n"))
    state = read_state()
    if not state:
        info("Not started by this launcher")
    else:
        info(f"Mode      {state.get('mode')}")
        info(f"Port      {state.get('port')}")

    port = state.get("port", DEFAULT_PORT)
    pid = read_pid()
    if pid:
        info(f"PID       {pid} ({'alive' if pid_alive(pid) else 'dead'})")

    url = f"http://127.0.0.1:{port}"
    if http_ok(f"{url}/api/v1/health"):
        ok(f"Healthy at http://localhost:{port}")
        try:
            with urllib.request.urlopen(f"{url}/api/v1/health", timeout=3) as r:
                data = json.loads(r.read().decode())
            info(f"Status    {data.get('status')}")
        except Exception:
            pass
    else:
        fail("Not responding")
    print()
    return 0


# ── reset ────────────────────────────────────────────────────────────────────

def cmd_reset(args) -> int:
    print(_c("1;36", "\n  Argus - reset\n"))
    cmd_stop(args)
    targets = [venv_dir(), FRONTEND_DIST, FRONTEND / "node_modules", STATE_DIR]
    if args.all:
        targets.append(ROOT / "data" / "argus.db")
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
            ok(f"Removed {t.relative_to(ROOT)}")
        elif t.is_file():
            t.unlink(missing_ok=True)
            ok(f"Removed {t.relative_to(ROOT)}")
    info("Run 'python argus.py start' to set up again")
    print()
    return 0


# ── cli ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        prog="argus",
        description="Set up, start and stop Argus with one command on any OS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python argus.py start              set up and run\n"
               "  python argus.py start --native     ignore Docker, use a virtualenv\n"
               "  python argus.py start --rebuild    force a fresh dashboard build\n"
               "  python argus.py stop               stop everything\n"
               "  python argus.py doctor             check this machine\n",
    )
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("start", help="set everything up and run it")
    s.add_argument("--port", type=int, default=None, help=f"port (default {DEFAULT_PORT})")
    s.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    s.add_argument("--native", action="store_true", help="force the virtualenv path")
    s.add_argument("--docker", action="store_true", help="require Docker")
    s.add_argument("--rebuild", action="store_true", help="rebuild images / dashboard")
    s.add_argument("--reinstall", action="store_true", help="reinstall dependencies")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser")
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("stop", help="stop everything")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("status", help="show what is running")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("doctor", help="check this machine can run Argus")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("reset", help="stop and delete generated files")
    s.add_argument("--all", action="store_true", help="also delete the database")
    s.set_defaults(func=cmd_reset)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n  Interrupted. Run 'python argus.py stop' to clean up.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
