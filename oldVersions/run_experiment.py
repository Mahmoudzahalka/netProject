#!/usr/bin/env python3
import subprocess
import time
import os
import statistics

NUM_HOSTS = 16

# All FatTree(k=4) switch names - manually added by me as a start will change it later
SWITCHES = [
    "c_s0", "c_s1", "c_s2", "c_s3",
    "p0_s0", "p0_s1", "p0_s2", "p0_s3",
    "p1_s0", "p1_s1", "p1_s2", "p1_s3",
    "p2_s0", "p2_s1", "p2_s2", "p2_s3",
    "p3_s0", "p3_s1", "p3_s2", "p3_s3"
]



# -------------------------------
# Run shell command
# -------------------------------
def sh(cmd):
    return subprocess.check_output(cmd, shell=True).decode()

# -------------------------------
# Start tcpdump on all switches (for latency measurement)
# -------------------------------
def start_tcpdump():
    print("Starting tcpdump on all switches...")
    for sw in SWITCHES:
        os.system(f"mnexec -a $(pgrep -f {sw}) tcpdump -i {sw}-eth1 -w {sw}.pcap &")

# -------------------------------
# Capture OVS counters before/after
# -------------------------------
def dump_port_stats(prefix):
    print(f"Dumping port stats: {prefix}")
    for sw in SWITCHES:
        out = sh(f"ovs-ofctl dump-ports {sw}")
        with open(f"{prefix}_{sw}.txt", "w") as f:
            f.write(out)

# -------------------------------
# Launch all hosts' traffic generators
# -------------------------------
def launch_all_to_all():
    print("Launching all-to-all gradient exchange...")

    for rank in range(NUM_HOSTS):
        cmd = f"h{rank} python3 all_to_all_gradients.py {rank} &"
        sh(f"mnexec -a $(pgrep -f 'h{rank}') {cmd}")

# -------------------------------
# Get total experiment time
# -------------------------------
def run_experiment():
    print("\n====== STARTING EXPERIMENT ======\n")

    start = time.time()

    # 1) Capture initial switch load
    dump_port_stats("before")

    # 2) Start tcpdump capture
    start_tcpdump()

    # 3) Launch workload
    launch_all_to_all()

    # 4) Wait until all 7 iterations finish
    print("Waiting for workload to finish...")
    time.sleep(10)

    # 5) Capture final switch load
    dump_port_stats("after")

    end = time.time()
    total_time = end - start

    print("\n====== EXPERIMENT DONE ======")
    print(f"TOTAL TIME = {total_time:.3f} seconds\n")

    return total_time

# -------------------------------
# Extract latency from pcap files
# -------------------------------
def extract_latency():
    print("Extracting latency from pcap files...")

    latencies = []

    for sw in SWITCHES:
        pcap = f"{sw}.pcap"
        if not os.path.exists(pcap):
            continue

        # Extract timestamps of packets
        out = sh(f"tshark -r {pcap} -T fields -e frame.time_epoch")

        times = [float(t) for t in out.splitlines() if t.strip()]

        if len(times) >= 2:
            # latency per hop = interarrival between two appearances
            # (not perfect, but works well for UDP simulation)
            diffs = [
                times[i+1] - times[i]
                for i in range(len(times)-1)
            ]
            latencies.extend(diffs)

    if not latencies:
        print("No packets seen in capture.")
        return None

    avg = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=100)[94]
    return avg, p95

# -------------------------------
# Compute link load / bandwidth utilization
# -------------------------------
def compute_link_load():
    print("Computing link load...")

    results = {}

    for sw in SWITCHES:
        before_file = f"before_{sw}.txt"
        after_file = f"after_{sw}.txt"

        if not (os.path.exists(before_file) and os.path.exists(after_file)):
            continue

        with open(before_file) as f1, open(after_file) as f2:
            before = f1.read()
            after = f2.read()

        # parse TX bytes
        import re

        b_tx = list(map(int, re.findall(r"tx_bytes=(\d+)", before)))
        a_tx = list(map(int, re.findall(r"tx_bytes=(\d+)", after)))

        loads = [a - b for a, b in zip(a_tx, b_tx)]

        results[sw] = loads

    return results


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    total_time = run_experiment()
    latency = extract_latency()
    load = compute_link_load()

    print("\n========== FINAL METRICS ==========\n")

    print(f"Total runtime: {total_time:.3f} seconds")

    if latency:
        avg, p95 = latency
        print(f"Latency (avg):  {avg*1000:.3f} ms")
        print(f"Latency (p95): {p95*1000:.3f} ms")
    else:
        print("Latency data unavailable")

    print("\nLink loads (bytes transmitted per interface):")
    for sw, loads in load.items():
        print(f"  {sw}: {loads}")

    print("\nBandwidth utilization:")
    # assuming 1 Gbps links
    link_speed = 1_000_000_000 / 8  # bytes per second
    for sw, loads in load.items():
        bw = [L / total_time for L in loads]
        util = [round(100 * (b / link_speed), 2) for b in bw]
        print(f"  {sw}: {util} % utilization each interface")

    print("\n====================================\n")

