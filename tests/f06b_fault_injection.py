#!/usr/bin/env python3
"""F06b real-binary fault receipts. No real data, no power-cut claim."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("F06B_OUT", ROOT / "tmp/F06b/probes"))
OUT.mkdir(parents=True, exist_ok=True)
BIN = ROOT / "target_integration_default/debug/splatsdb"
SHIM = OUT / "f06b_syscalls.so"
DIM = 4

def request(port, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        raw = error.read()
        return error.code, json.loads(raw) if raw else {}

def status(port):
    return request(port, "POST", "/status")[1]

def wait_health(child, port):
    deadline = time.time() + 30
    while time.time() < deadline:
        if child.poll() is not None:
            raise AssertionError(f"server exited {child.returncode}")
        try:
            if request(port, "GET", "/health")[0] == 200:
                return
        except Exception:
            pass
        time.sleep(0.03)
    raise AssertionError("server health timeout")

def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def start(work, mode="process-crash", fail_write="", fail_fsync=False, partial=False):
    port = unused_port()
    trace = work / "syscalls.log"
    arm = work / "armed"
    env = os.environ.copy()
    env.update(LD_PRELOAD=str(SHIM), F06B_TRACE=str(trace), F06B_ARM_FILE=str(arm))
    if fail_write:
        env["F06B_FAIL_WRITE"] = fail_write
    if fail_fsync:
        env["F06B_FAIL_FSYNC"] = "1"
    if partial:
        env["F06B_PARTIAL_WRITE"] = "1"
    log = open(work / "server.log", "wb")
    child = subprocess.Popen([str(BIN), "--data-dir", str(work / "data"), "--dim", str(DIM),
                              "serve", "--host", "127.0.0.1", "--port", str(port), "--wal-mode", mode],
                             stdout=log, stderr=log, env=env)
    wait_health(child, port)
    return child, port, trace, arm, log

def stop(child, log, sig=signal.SIGTERM):
    if child.poll() is None:
        child.send_signal(sig)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    log.close()

def body(name):
    return {"id": name, "text": name, "embedding": [float(i) / 10 for i in range(DIM)]}

def wal_ids(work):
    raw = (work / "data/wal/store.wal").read_bytes()
    found = []
    pos = 0
    while pos + 4 <= len(raw):
        size = int.from_bytes(raw[pos:pos+4], "big")
        if pos + 4 + size > len(raw):
            break
        entry = json.loads(raw[pos+4:pos+4+size])
        if entry.get("operation") == "store":
            found.append(entry["data"]["id"])
        pos += 4 + size
    return found

def scenario(tag, mode, fail_write="", fail_fsync=False, partial=False):
    work = OUT / tag
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    child, port, trace, arm, log = start(work, mode, fail_write, fail_fsync, partial)
    result = {"scenario": tag, "mode": mode}
    try:
        before_wal = (work / "data/wal/store.wal").read_bytes()
        arm.touch()
        code, failure = request(port, "POST", "/store", body("retry-me"))
        result["failure_http"] = code
        result["failure"] = failure
        assert code == 503 and failure["status"] != "stored", result
        assert failure["id"] == "retry-me"
        after_failure_wal = (work / "data/wal/store.wal").read_bytes()
        if fail_fsync:
            assert failure["outcome"] == "unknown" and failure["may_be_persisted"]
            assert failure["crash_durable"] and not failure["power_loss_durable"]
            assert after_failure_wal != before_wal
            assert "WAL_FSYNC" in trace.read_text() and "injected errno=5" in trace.read_text()
            current = status(port)
            assert not current["durable"] and current["wal_pending"] == 1, current
        else:
            assert failure["outcome"] == "not_stored" and not failure["may_be_persisted"]
            assert after_failure_wal == before_wal
            assert "WAL_WRITE" in trace.read_text() and "errno=" in trace.read_text()
            assert not failure["crash_durable"] and not failure["durable"]
        assert status(port)["n_active"] == 0
        arm.unlink()
        code, retried = request(port, "POST", "/store", body("retry-me"))
        assert code == 200 and retried["status"] == "stored", retried
        assert status(port)["n_active"] == 1, status(port)
        assert wal_ids(work) == ["retry-me"], wal_ids(work)
        assert retried["crash_durable"]
        assert retried["power_loss_durable"] == (mode == "power-loss")
        result["retry"] = retried
        result["wal_ids"] = wal_ids(work)
        result["syscalls"] = trace.read_text().splitlines()
    finally:
        stop(child, log)
    return result

def process_crash():
    work = OUT / "process-crash"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    child, port, trace, arm, log = start(work)
    result = {"scenario": "process-crash", "scope": "SIGKILL only; not a power-cut test"}
    try:
        arm.unlink(missing_ok=True)
        before = trace.stat().st_size if trace.exists() else 0
        code, stored = request(port, "POST", "/store", body("survives-kill"))
        assert code == 200 and stored["crash_durable"] and not stored["power_loss_durable"]
        with trace.open("rb") as stream:
            stream.seek(before)
            receipts = stream.read().decode().splitlines()
        assert any("WAL_WRITE" in line for line in receipts), receipts
        assert not any("WAL_FSYNC" in line for line in receipts), receipts
        result["store"] = stored
        result["store_syscalls"] = receipts
    finally:
        stop(child, log, signal.SIGKILL)
    restarted, port2, trace2, arm2, log2 = start(work)
    try:
        result["after_restart"] = status(port2)
        assert result["after_restart"]["n_active"] == 1
        assert wal_ids(work) == ["survives-kill"]
    finally:
        stop(restarted, log2)
    return result

def main():
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", "-o", str(SHIM),
                    str(ROOT / "tests/f06b_syscalls.c")], check=True)
    reports = [
        scenario("write-enospc", "process-crash", fail_write="ENOSPC"),
        scenario("write-eio-partial", "process-crash", fail_write="EIO", partial=True),
        scenario("fsync-eio-power-loss", "power-loss", fail_fsync=True),
        process_crash(),
    ]
    (OUT / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({"scenarios": len(reports), "results": str(OUT / "results.json")}))

if __name__ == "__main__":
    main()
