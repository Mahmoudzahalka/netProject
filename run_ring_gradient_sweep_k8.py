#!/usr/bin/env python3
"""
Automated gradient-size sweep for ring_simulation_mininet_k8.py

Runs the k=8 fat-tree ring all-reduce simulation for each gradient size
in sequence, handling the Mininet CLI interaction and cleanup between runs.

Usage (from /home/mahmoud/Desktop/netProject):
    sudo python3 run_ring_gradient_sweep_k8.py
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "/home/mahmoud/Desktop/netProject"
CONTAINERNET_DIR = "/home/mahmoud/networks/containernet"
VENV_PATH = os.path.join(CONTAINERNET_DIR, "venv")
ORIG_SCRIPT = os.path.join(BASE_DIR, "ring_simulation_mininet_k8.py")
TEMP_SCRIPT = os.path.join(BASE_DIR, "_k8_sweep_temp.py")

WORLD_SIZE = 128  # k=8 fat-tree

# (label, total gradient bytes per host)
GRADIENT_SIZES = [
    ("1KB",   1 * 1024),
    ("100KB", 100 * 1024),
    ("500KB", 500 * 1024),
    ("1MB",   1 * 1024 * 1024),
    ("2MB",   2 * 1024 * 1024),
    ("4MB",   4 * 1024 * 1024),
    ("6MB",   6 * 1024 * 1024),
]

# Seconds to wait after the mininet> prompt appears before sending "exit".
# Gives the background all-reduce time to complete for larger gradients.
CLI_WAIT_SECS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def elems_per_chunk_for(total_bytes):
    """Compute elems_per_chunk so that grad_elems * 4 ~= total_bytes."""
    grad_elems = total_bytes // 4  # 4 bytes per float32
    epc = grad_elems // WORLD_SIZE
    return max(epc, 1)


def create_patched_script(elems_per_chunk, label):
    """Copy the k=8 script, replace elems_per_chunk in both locations,
    and tag per-size output log filenames."""
    shutil.copy2(ORIG_SCRIPT, TEMP_SCRIPT)
    with open(TEMP_SCRIPT, "r") as f:
        content = f.read()

    # Replace all occurrences of `elems_per_chunk = NNN`.
    # The k=8 script has it in setup_and_start_ring and collect_ring_metrics.
    content = re.sub(
        r"elems_per_chunk\s*=\s*\d+",
        f"elems_per_chunk = {elems_per_chunk}",
        content,
    )

    tag = label.lower()  # e.g. "1kb", "100kb", "6mb"
    content = content.replace(
        '"ring_metrics.log"',
        f'"{tag}_ring_metrics.log"',
    )
    content = content.replace(
        '"link_load.log"',
        f'"{tag}_ring_link_load.log"',
    )

    with open(TEMP_SCRIPT, "w") as f:
        f.write(content)


def stream_and_detect(stream, event, label):
    """
    Read process stdout char-by-char, print lines, and set `event` when
    the Mininet CLI prompt is detected. The prompt does not end with a
    newline, so readline() would block forever.
    """
    buf = ""
    while True:
        ch = stream.read(1)
        if not ch:
            if buf:
                print(f"[{label}] {buf}", flush=True)
            break
        buf += ch
        if ch == "\n":
            print(f"[{label}] {buf.rstrip()}", flush=True)
            buf = ""
        elif "containernet>" in buf:
            print(f"[{label}] {buf}", flush=True)
            event.set()
            buf = ""
    stream.close()


def run_one(label, total_bytes):
    """Run a single simulation with the given gradient size."""
    epc = elems_per_chunk_for(total_bytes)
    actual_bytes = epc * WORLD_SIZE * 4
    print(f"\n{'=' * 70}")
    print(f"  K=8 RING GRADIENT SWEEP — {label}  "
          f"(elems_per_chunk={epc}, actual={actual_bytes} bytes)")
    print(f"{'=' * 70}\n", flush=True)

    create_patched_script(epc, label)

    # Build PATH with containernet venv so `python3` resolves to the
    # interpreter that has mininet/containernet installed.
    venv_bin = os.path.join(VENV_PATH, "bin")
    env_path = f"{venv_bin}:{os.environ.get('PATH', '')}"

    cmd = [
        "sudo", "-E", "env", f"PATH={env_path}",
        "python3", TEMP_SCRIPT,
    ]

    cli_ready = threading.Event()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=BASE_DIR,
    )

    reader = threading.Thread(
        target=stream_and_detect,
        args=(proc.stdout, cli_ready, label),
        daemon=True,
    )
    reader.start()

    # Wait for the CLI prompt (generous timeout for setup + routing install)
    if not cli_ready.wait(timeout=600):
        print(f"[{label}] WARNING: never saw containernet> prompt, sending exit anyway",
              flush=True)

    # Give the background all-reduce time to finish
    print(f"[{label}] CLI ready — waiting {CLI_WAIT_SECS}s for all-reduce to complete …",
          flush=True)
    time.sleep(CLI_WAIT_SECS)

    # Send "exit" to the Mininet CLI
    print(f"[{label}] Sending 'exit' to Containernet CLI …", flush=True)
    try:
        proc.stdin.write("exit\n")
        proc.stdin.flush()
    except BrokenPipeError:
        pass

    # Wait for the process to finish (collect_ring_metrics sleeps 60s internally)
    print(f"[{label}] Waiting for metrics collection and shutdown …", flush=True)
    try:
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        print(f"[{label}] WARNING: timed out waiting for shutdown, killing …",
              flush=True)
        proc.kill()
        proc.wait()
    reader.join(timeout=10)

    print(f"[{label}] Simulation finished with return code {proc.returncode}",
          flush=True)

    # Cleanup: mn -c
    print(f"[{label}] Running mn -c cleanup …", flush=True)
    cleanup = subprocess.run(
        ["sudo", "-E", "env", f"PATH={env_path}", "mn", "-c"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if cleanup.returncode != 0:
        print(f"[{label}] mn -c stderr: {cleanup.stderr.strip()}", flush=True)

    # Small pause between runs
    time.sleep(3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.chdir(BASE_DIR)

    if not os.path.exists(ORIG_SCRIPT):
        print(f"ERROR: {ORIG_SCRIPT} not found", flush=True)
        sys.exit(1)

    if not os.path.isdir(VENV_PATH):
        print(f"ERROR: Containernet venv not found at {VENV_PATH}", flush=True)
        sys.exit(1)

    # Prepend venv/bin to PATH so child processes can find python3/mn
    venv_bin = os.path.join(VENV_PATH, "bin")
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
    os.environ["VIRTUAL_ENV"] = VENV_PATH

    for label, total_bytes in GRADIENT_SIZES:
        try:
            run_one(label, total_bytes)
        except Exception as exc:
            print(f"[{label}] FAILED: {exc}", flush=True)
            subprocess.run(
                ["sudo", "mn", "-c"],
                cwd=BASE_DIR,
                capture_output=True,
                timeout=60,
            )
            time.sleep(3)

    # Remove temp script
    if os.path.exists(TEMP_SCRIPT):
        os.remove(TEMP_SCRIPT)

    print("\n" + "=" * 70)
    print("  K=8 SWEEP COMPLETE")
    print("  Check <size>_ring_metrics.log and <size>_ring_link_load.log")
    print("=" * 70)


if __name__ == "__main__":
    main()