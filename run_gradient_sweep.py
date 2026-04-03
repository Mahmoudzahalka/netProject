#!/usr/bin/env python3
"""
Automated gradient-size sweep for simulation_tree_fix_v1.py

Runs the fat-tree hierarchical tree all-reduce simulation for each
gradient size in sequence, handling the Containernet CLI interaction
and cleanup between runs.

Usage (from /home/mahmoud/networks/containernet):
    python3 run_gradient_sweep.py
"""

import os
import re
import subprocess
import sys
import threading
import time
import shutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "/home/mahmoud/networks/containernet"
ORIG_SCRIPT = os.path.join(BASE_DIR, "examples", "simulation_tree_fix_v1.py")
TEMP_SCRIPT = os.path.join(BASE_DIR, "examples", "_sweep_temp.py")

WORLD_SIZE = 16  # 16 hosts in k=4 fat-tree

# (label, total gradient bytes)
GRADIENT_SIZES = [
    ("4MB",   4 * 1024 * 1024),
]

# Seconds to wait after the CLI prompt appears before sending "exit".
# This gives the background all-reduce time to finish.
CLI_WAIT_SECS = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def elems_per_chunk_for(total_bytes):
    """Compute elems_per_chunk so that grad_elems * 4 == total_bytes."""
    grad_elems = total_bytes // 4          # 4 bytes per float32
    epc = grad_elems // WORLD_SIZE
    return max(epc, 1)


def create_patched_script(elems_per_chunk, label):
    """Copy the original script, replace elems_per_chunk, and set per-gradient log filenames."""
    shutil.copy2(ORIG_SCRIPT, TEMP_SCRIPT)
    with open(TEMP_SCRIPT, "r") as f:
        content = f.read()
    content = re.sub(
        r"elems_per_chunk\s*=\s*\d+",
        f"elems_per_chunk = {elems_per_chunk}",
        content,
    )
    tag = label.lower()  # e.g. "1kb", "100kb", "4mb"
    content = content.replace(
        '"tree_metrics.log"',
        f'"{tag}_tree_metrics.log"',
    )
    content = content.replace(
        '"tree_link_load.log"',
        f'"{tag}_tree_link_load.log"',
    )
    content = content.replace(
        '"tree_routes.log"',
        f'"{tag}_tree_routes.log"',
    )
    with open(TEMP_SCRIPT, "w") as f:
        f.write(content)


def stream_and_detect(stream, event, label):
    """
    Read process stdout char-by-char, accumulate lines, print them,
    and set `event` when the Containernet CLI prompt is detected.
    The prompt doesn't end with a newline, so readline() would block forever.
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


def run_one(label, total_bytes, venv_path):
    """Run a single simulation with the given gradient size."""
    epc = elems_per_chunk_for(total_bytes)
    actual_bytes = epc * WORLD_SIZE * 4
    print(f"\n{'='*70}")
    print(f"  GRADIENT SWEEP — {label}  "
          f"(elems_per_chunk={epc}, actual={actual_bytes} bytes)")
    print(f"{'='*70}\n", flush=True)

    create_patched_script(epc, label)

    # Build the PATH that includes the venv
    venv_bin = os.path.join(venv_path, "bin")
    env_path = f"{venv_bin}:{os.environ.get('PATH', '')}"

    cmd = [
        "sudo", "-E", "env", f"PATH={env_path}",
        "python3", f"examples/_sweep_temp.py",
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

    # Wait for the CLI prompt (timeout 300s covers OSPF convergence + startup)
    if not cli_ready.wait(timeout=300):
        print(f"[{label}] WARNING: never saw containernet> prompt, sending exit anyway",
              flush=True)

    # Give the background all-reduce time to finish
    print(f"[{label}] CLI ready — waiting {CLI_WAIT_SECS}s for all-reduce to complete …",
          flush=True)
    time.sleep(CLI_WAIT_SECS)

    # Send "exit" to the Containernet CLI
    print(f"[{label}] Sending 'exit' to Containernet CLI …", flush=True)
    try:
        proc.stdin.write("exit\n")
        proc.stdin.flush()
    except BrokenPipeError:
        pass

    # Wait for the process to finish (collect_tree_metrics sleeps 20s internally)
    print(f"[{label}] Waiting for metrics collection and shutdown …", flush=True)
    proc.wait(timeout=120)
    reader.join(timeout=10)

    print(f"[{label}] Simulation finished with return code {proc.returncode}", flush=True)

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

    # ---- one-time venv setup ----
    venv_path = os.path.join(BASE_DIR, "venv")
    if not os.path.isdir(venv_path):
        print("Creating virtual environment …", flush=True)
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            cwd=BASE_DIR,
            check=True,
        )
    else:
        print("Virtual environment already exists, reusing.", flush=True)

    # Activate = just prepend venv/bin to PATH for child processes
    venv_bin = os.path.join(venv_path, "bin")
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
    os.environ["VIRTUAL_ENV"] = venv_path

    # ---- sweep ----
    for label, total_bytes in GRADIENT_SIZES:
        try:
            run_one(label, total_bytes, venv_path)
        except Exception as exc:
            print(f"[{label}] FAILED: {exc}", flush=True)
            # Still try to clean up
            subprocess.run(
                ["sudo", "-E", "env", f"PATH={os.environ['PATH']}", "mn", "-c"],
                cwd=BASE_DIR,
                capture_output=True,
                timeout=60,
            )
            time.sleep(3)

    # Remove temp script
    if os.path.exists(TEMP_SCRIPT):
        os.remove(TEMP_SCRIPT)

    print("\n" + "=" * 70)
    print("  SWEEP COMPLETE — check <size>_tree_metrics.log, <size>_tree_link_load.log, and <size>_tree_routes.log")
    print("=" * 70)


if __name__ == "__main__":
    main()
