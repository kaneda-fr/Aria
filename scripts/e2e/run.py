"""Stage current source on Apollo; test through SSH; always stop owned services.

Usage: .venv/bin/python scripts/e2e/run.py --host USER@APOLLO [--mic]
Requires existing Apollo models and Python runtime; never runs sync.sh or start.sh.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import ipaddress
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import wave

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[2]


def command(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=True, text=True, **kwargs)


def audit_network(trace: str) -> dict:
    """Reject observed non-loopback destinations and DNS, even failed attempts."""
    rejected = []
    connections = 0
    for line in trace.splitlines():
        if not re.search(r"\b(connect|sendto|sendmsg)\(", line):
            continue
        addresses = re.findall(r'inet_addr\("([^"]+)"\)', line)
        addresses += re.findall(r'inet_pton\(AF_INET6, "([^"]+)"', line)
        connections += len(addresses)
        if any(not ipaddress.ip_address(a).is_loopback for a in addresses) or re.search(r'htons\(53\)', line):
            rejected.append(line)
    return {"loopback_destinations_seen": connections, "rejected": rejected,
            "passed": connections > 0 and not rejected}


async def wait_output(client: httpx.AsyncClient, after: int, timeout: float = 180,
                      keyword: str | None = None) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = (await client.get("/_e2e/state", params={"after": after})).json()
        events = state["events"]
        failures = [e for e in events if e["event"] in {
            "ARIA.ASR.Error", "ARIA.LLM.Error", "ARIA.LLM.EmptyResponse", "ARIA.TTS.Error",
        }]
        if failures:
            raise RuntimeError(f"Pipeline failed: {failures}")
        completion = events
        if keyword is not None:
            matched = [i for i, e in enumerate(events) if e["event"] == "ARIA.LLM.Request"
                       and keyword in e.get("user", "").lower()]
            completion = events[matched[-1]+1:] if matched else []
        if (any(e["event"] == "ARIA.LLM.Done" for e in completion)
                and any(e["event"] == "E2E.AudioFetched" for e in completion)
                and not state["speech_busy"]):
            return state
        await asyncio.sleep(0.5)
    raise TimeoutError("No completed ASR/LLM/TTS output; inspect collected server log")


async def exercise(base: str, artifacts: Path, microphone: bool, report: dict) -> dict:
    from client_replay import replay
    # Keep successful stages in the caller's report even if a later stage fails.
    report.update({"tests": [], "physical_sonos": False, "device_actions": False,
                   "router": False, "speaker_recognition": False})
    async with httpx.AsyncClient(base_url=base, timeout=150, trust_env=False) as client:
        deadline = time.monotonic() + 120
        while True:
            try:
                state = (await client.get("/_e2e/state", timeout=3)).json()
                break
            except (httpx.HTTPError, ValueError):
                if time.monotonic() > deadline:
                    raise TimeoutError("Test server did not become ready")
                await asyncio.sleep(1)
        assert all(state[k] for k in ("asr", "llm", "tts")), state
        assert state["plugins"] == 0 and not state["router"] and not state["speaker_recognition"], state
        report["ready"] = state
        after = state["next"]
        t0 = time.monotonic()
        response = await client.post("/api/llm/process", json={
            "text": "What is two plus two? Answer with just the number.",
            "use_tools": False, "speak_response": True,
        })
        response.raise_for_status()
        data = response.json()
        assert data["success"] and data["spoken"] and data["tool_calls_made"] == 0, data
        assert re.search(r"\b(4|four)\b", data["response"], re.I), data
        output = await wait_output(client, after)
        report["tests"].append({"name": "text_llm_tts_http", "seconds": time.monotonic()-t0,
                                "response": data, "events": output["events"]})
        print("PASS text -> local LLM -> Piper -> HTTP WAV", flush=True)

        for language, keyword in [("en", "cat"), ("fr", "chat")]:
            response = await client.get(f"/_e2e/fixture/{language}")
            response.raise_for_status()
            fixture = artifacts / f"input-{language}.wav"
            fixture.write_bytes(response.content)
            with wave.open(io.BytesIO(response.content)) as audio:
                assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
                pcm = audio.readframes(audio.getnframes())
            after = (await client.get("/_e2e/state")).json()["next"]
            t0 = time.monotonic()
            async with websockets.connect(base.replace("http://", "ws://") + "/ws/asr", proxy=None) as ws:
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1,
                                         "format": "pcm_s16le", "chunk_ms": 40, "client_vad": True}))
                client_result = await replay(ws, pcm)
                output = await wait_output(client, after, keyword=keyword)
                await ws.send(json.dumps({"type": "stop"}))
            requests = [e.get("user", "") for e in output["events"] if e["event"] == "ARIA.LLM.Request"]
            assert any(keyword in s.lower() for s in requests), requests
            report["tests"].append({"name": f"wav_{language}_mac_client_asr_llm_tts_http",
                                    "client": client_result,
                                    "seconds": time.monotonic()-t0, "events": output["events"]})
            print(f"PASS {language} audio -> Mac client VAD/resampling -> ASR -> LLM -> Piper -> HTTP WAV", flush=True)

        if microphone:
            from mac_audio import microphone_authorization
            permission = microphone_authorization()
            report["microphone_authorization"] = permission
            if permission != "authorized":
                raise RuntimeError(f"macOS microphone permission is {permission}; grant access before --mic")
            print("Testing Mac microphone for 15 seconds; playing the synthetic English fixture locally.", flush=True)
            after = (await client.get("/_e2e/state")).json()["next"]
            env = {**os.environ, "ARIA_CLIENT_VAD": "1", "ARIA_CLIENT_DEBUG": "1", "ARIA_CLIENT_AGC": "0"}
            with (artifacts / "microphone.log").open("w") as log:
                mic = subprocess.Popen([sys.executable, str(ROOT / "client/macos/mic_stream.py"),
                    base.replace("http://", "ws://") + "/ws/asr", "--device", "MacBook Pro Microphone"],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    await asyncio.sleep(3)
                    player = await asyncio.create_subprocess_exec("afplay", str(artifacts / "input-en.wav"))
                    try:
                        await asyncio.wait_for(player.wait(), timeout=9)
                    except asyncio.TimeoutError:
                        player.kill()
                        await player.wait()
                        raise
                    await asyncio.sleep(3)
                finally:
                    mic.send_signal(signal.SIGINT)
                    try:
                        await asyncio.to_thread(mic.wait, timeout=8)
                    except subprocess.TimeoutExpired:
                        mic.kill()
                        mic.wait()
            report["microphone_exit_code"] = mic.returncode
            if mic.returncode != 0:
                raise RuntimeError(f"Mac microphone client exited {mic.returncode}; inspect microphone.log")
            output = await wait_output(client, after, timeout=120, keyword="cat")
            requests = [e.get("user", "") for e in output["events"] if e["event"] == "ARIA.LLM.Request"]
            assert any("cat" in s.lower() for s in requests), requests
            report["tests"].append({"name": "mac_microphone_acoustic", "events": output["events"]})
            print("PASS actual Mac microphone -> client VAD/resampling -> Apollo pipeline", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("ARIA_E2E_SSH", "apollo"))
    parser.add_argument("--remote-root", default="/data/aria/e2e")
    parser.add_argument("--mic", action="store_true", help="Capture the real Mac microphone and play a test fixture")
    parser.add_argument("--capture-dir", type=Path, help="Baseline: replay confirmed saved captures; never open the microphone")
    parser.add_argument("--suite", choices=("smoke", "baseline"), default="smoke")
    args = parser.parse_args()
    if args.capture_dir:
        if args.mic or args.suite != 'baseline':
            parser.error('--capture-dir requires --suite baseline and cannot be combined with --mic')
        from capture_replay import load_capture
        for language in ('en', 'fr'):
            load_capture(args.capture_dir / language)
    if args.host.startswith("-") or not re.fullmatch(r"[\w.@-]+", args.host):
        parser.error("Expected an SSH alias or user@host")
    if not re.fullmatch(r"/[\w/-]+", args.remote_root):
        parser.error("Remote root must be an absolute path without spaces")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(3).hex()
    remote = f"{args.remote_root}/runs/{run_id}"
    unit = "aria-e2e-" + run_id.lower()
    artifacts = Path.home() / "Library/Logs/Aria/e2e" / run_id
    artifacts.mkdir(parents=True, mode=0o700)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host]
    tunnel = None
    report = {"run_id": run_id, "host": args.host, "remote": remote, "unit": unit,
              "suite": args.suite,
              "scope": "isolated conversation pipeline; physical Sonos/actions excluded"}
    if args.capture_dir:
        report['capture_dir'] = str(args.capture_dir.resolve())
    (artifacts / "run.json").write_text(json.dumps(report, indent=2))
    print(f"Artifacts: {artifacts}\nApollo test directory: {remote}", flush=True)
    try:
        command(ssh + [shlex.join(["install", "-d", "-m", "700", remote, remote + "/src"])])
        # Only source and selected tests; no private profiles, recordings, caches, or production scripts.
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "source.tar"
            paths = ["app", "scripts/e2e", "tests/plugins", "tests/test_sonos_http_sink.py",
                     "tests/test_request_queue.py", "tests/test_shared_requests.py",
                     "tests/test_speech_language.py", "tests/test_client_turns.py",
                     "tests/test_capture_replay.py", "client/macos/mic_stream.py"]
            command(["tar", "--no-xattrs", "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.backup",
                     "-cf", str(archive), *paths], cwd=ROOT)
            with archive.open("rb") as source:
                command(ssh + [shlex.join(["tar", "-xf", "-", "-C", remote + "/src"])], stdin=source)
            report["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        report["head"] = command(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True).stdout.strip()
        unit_test = ["timeout", "120s", "env", f"PYTHONPATH={remote}/src:/data/aria/e2e/deps", "/data/aria/dev/.venv/bin/python",
                     "-m", "pytest", remote + "/src/tests/plugins", remote + "/src/tests/test_sonos_http_sink.py",
                     remote + "/src/tests/test_request_queue.py", remote + "/src/tests/test_shared_requests.py",
                     remote + "/src/tests/test_speech_language.py", "-q"]
        result = subprocess.run(ssh + [shlex.join(unit_test)], capture_output=True, text=True)
        (artifacts / "unit-tests.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        print(result.stdout, flush=True)
        start = ["systemd-run", "--user", "--unit", unit, "--collect", "--property=RuntimeMaxSec=900",
                 "env", f"ARIA_E2E_RUN={remote}", f"ARIA_E2E_SUITE={args.suite}", "strace", "-f", "-e", "trace=network", "-s", "256",
                 "-o", remote + "/network.trace", "unshare", "--user", "--map-root-user", "--net",
                 "bash", remote + "/src/scripts/e2e/launch.sh"]
        command(ssh + [shlex.join(start)])
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with (artifacts / "tunnel.log").open("w") as log:
            tunnel = subprocess.Popen(ssh[:-1] + ["-N", "-o", "ExitOnForwardFailure=yes", "-L",
                f"127.0.0.1:{port}:{remote}/aria.sock", args.host], stdout=log, stderr=log)
        selected_exercise = exercise
        if args.suite == "baseline":
            from baseline import exercise_baseline
            selected_exercise = exercise_baseline
        asyncio.run(selected_exercise(f"http://127.0.0.1:{port}", artifacts, args.mic, report))
        report["passed"] = True
        return_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return_code = 1
    finally:
        resources = subprocess.run(ssh + [shlex.join(["systemctl", "--user", "show", unit,
            "-p", "MemoryPeak", "-p", "MemoryCurrent", "-p", "CPUUsageNSec"])], capture_output=True)
        (artifacts / "resources.txt").write_bytes(resources.stdout)
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        subprocess.run(ssh + [shlex.join(["systemctl", "--user", "stop", unit])], capture_output=True)
        for name in ["server.log", "ollama.log", "network.trace", "routes.txt", "models.json"]:
            result = subprocess.run(ssh + [shlex.join(["cat", remote + "/" + name])], capture_output=True)
            if result.returncode == 0:
                (artifacts / name).write_bytes(result.stdout)
        journal = subprocess.run(ssh + [shlex.join(["journalctl", "--user", "-u", unit,
                                                   "--no-pager", "-n", "50"])], capture_output=True)
        (artifacts / "journal.log").write_bytes(journal.stdout)
        trace = artifacts / "network.trace"
        if trace.exists():
            report["network_audit"] = audit_network(trace.read_text(errors="replace"))
            if not report["network_audit"]["passed"]:
                report["passed"] = False
                return_code = 1
        else:
            report["passed"] = False
            report["network_audit"] = {"passed": False, "error": "Missing trace"}
            return_code = 1
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (artifacts / "report.json").write_text(json.dumps(report, indent=2))
        print(f"Report: {artifacts / 'report.json'}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
