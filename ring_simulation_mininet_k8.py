#!/usr/bin/env python3
"""
ring_simulation_mininet_k8.py

- Builds a k=8 L3 Fat-Tree using plain Mininet with LinuxRouter nodes.
- Static ECMP routes (no OSPF/FRR needed).
- 128 hosts running TCP-based ring all-reduce.
- Much lighter than Containernet: all nodes are Linux network namespaces.
"""

import os
import re
import time
from collections import defaultdict

from mininet.net import Mininet
from mininet.node import Node
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel

# ---------------------------------------------------------------------------
# Host-side ring all-reduce script, written once to /tmp
# ---------------------------------------------------------------------------

RING_NODE_SCRIPT = r"""#!/usr/bin/env python3
import argparse
import socket
import time
from array import array

def recv_all(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise RuntimeError("Socket closed while expecting data")
        data.extend(chunk)
    return data

def establish_connections(rank, world_size, listen_port, right_ip, right_port, log):

    # Server for left neighbor
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", listen_port))
    server.listen(1)

    # Client to right neighbor, with retry to avoid timing issues
    while True:
        try:
            sock_right = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_right.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock_right.connect((right_ip, right_port))
            break
        except Exception as e:
            log(f"[rank {rank}] connect to right failed ({e}), retrying...")
            time.sleep(0.1)

    # Accept connection from left neighbor
    conn_left, addr_left = server.accept()
    conn_left.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # One-byte handshake to be sure both directions are live
    sock_right.sendall(b"H")
    _ = recv_all(conn_left, 1)

    server.close()
    return conn_left, sock_right

def ring_allreduce(rank, world_size, conn_left, conn_right, grad_elems, log):

    elems_per_chunk = grad_elems // world_size
    assert grad_elems % world_size == 0

    grad = array("f", [float(rank)] * grad_elems)
    chunk_bytes = elems_per_chunk * 4

    def send_chunk(chunk_index):
        start = chunk_index * elems_per_chunk
        out_chunk = grad[start : start + elems_per_chunk]
        conn_right.sendall(out_chunk.tobytes())

    def recv_chunk(chunk_index, do_reduce):
        start = chunk_index * elems_per_chunk
        in_bytes = recv_all(conn_left, chunk_bytes)
        in_chunk = array("f")
        in_chunk.frombytes(in_bytes)

        if do_reduce:
            grad[start : start + elems_per_chunk] = in_chunk
        else:
            grad[start : start + elems_per_chunk] = in_chunk

    # Phase 1: scatter-reduce
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=True)

    # Phase 2: all-gather
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--right-ip", type=str, required=True)
    parser.add_argument("--right-port", type=int, required=True)
    parser.add_argument("--grad-elems", type=int, default=1048576)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        print(line, flush=True)
        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(line + "\n")

    conn_left, conn_right = establish_connections(
        args.rank, args.world_size,
        args.listen_port, args.right_ip, args.right_port,
        log
    )

    t0 = time.time()
    ring_allreduce(args.rank, args.world_size, conn_left, conn_right, args.grad_elems, log)
    t1 = time.time()

    log(f"[rank {args.rank}] TOTAL_TIME_SEC={t1 - t0:.6f}")

    conn_left.close()
    conn_right.close()

if __name__ == "__main__":
    main()
"""

RING_SCRIPT_PATH = "/tmp/ring_allreduce/ring_node.py"
RING_LOG_DIR = "/tmp/ring_allreduce"


# ---------------------------------------------------------------------------
# LinuxRouter: lightweight L3 router using Linux kernel forwarding
# ---------------------------------------------------------------------------

class LinuxRouter(Node):
    """A Node with IP forwarding and L4 ECMP hashing enabled."""

    def config(self, **params):
        super().config(**params)
        self.cmd("sysctl -w net.ipv4.ip_forward=1")
        self.cmd("sysctl -w net.ipv4.fib_multipath_hash_policy=1")

    def terminate(self):
        self.cmd("sysctl -w net.ipv4.ip_forward=0")
        super().terminate()


# ---------------------------------------------------------------------------
# Unique /31 allocator for P2P links
# ---------------------------------------------------------------------------

class P2PAllocator:
    def __init__(self):
        self.n = 0

    def next31(self):
        n = self.n
        self.n += 1
        # 128 /31 pairs per third-octet (256 addresses / 2)
        A = n // 128
        B = n % 128
        ip1 = f"172.16.{A}.{2 * B}/31"
        ip2 = f"172.16.{A}.{2 * B + 1}/31"
        return ip1, ip2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def strip_mask(ip_with_mask):
    """'172.16.0.1/31' -> '172.16.0.1'"""
    return ip_with_mask.split("/")[0]


# ---------------------------------------------------------------------------
# Enforce correct IPs on all interfaces after net.start()
# ---------------------------------------------------------------------------

def enforce_all_ips(edge_agg_links, agg_core_links, host_links):
    """Flush and re-apply IPs on every interface to avoid Mininet auto-assign conflicts."""

    info("*** Enforcing IPs on all P2P interfaces\n")
    for n1, i1, ip1, n2, i2, ip2 in edge_agg_links + agg_core_links:
        n1.cmd(f"ip addr flush dev {i1} && ip addr add {ip1} dev {i1}")
        n2.cmd(f"ip addr flush dev {i2} && ip addr add {ip2} dev {i2}")

    info("*** Enforcing IPs on all host interfaces\n")
    for edge, edge_intf, edge_ip, host, host_intf, host_ip in host_links:
        edge.cmd(f"ip addr flush dev {edge_intf} && ip addr add {edge_ip} dev {edge_intf}")
        host.cmd(f"ip addr flush dev {host_intf} && ip addr add {host_ip} dev {host_intf}")
        gw = strip_mask(edge_ip)
        host.cmd(f"ip route add default via {gw}")


# ---------------------------------------------------------------------------
# Build k=8 Fat-Tree
# ---------------------------------------------------------------------------

def build_fattree(net, k, p2p_alloc):
    core = []
    agg_per_pod = {}
    edge_per_pod = {}
    all_routers = []
    edge_agg_links = []   # (edge, intf, ip, agg, intf, ip)
    agg_core_links = []   # (agg, intf, ip, core, intf, ip)
    host_links = []       # (edge, intf, ip, host, intf, ip)

    half = k // 2

    # --- Core routers ---
    info("*** Creating core routers\n")
    for i in range(half ** 2):
        r = net.addHost(f"c{i}", cls=LinuxRouter)
        core.append(r)
        all_routers.append(r)

    # --- Pods ---
    for p in range(k):
        agg = []
        edge = []

        # Aggregation routers
        for a in range(half, k):
            r = net.addHost(f"p{p}_a{a}", cls=LinuxRouter)
            agg.append(r)
            all_routers.append(r)

        # Edge routers + hosts
        for e in range(half):
            r = net.addHost(f"p{p}_e{e}", cls=LinuxRouter)
            edge.append(r)
            all_routers.append(r)

            for h in range(2, half + 2):
                base = 4 * h
                r_ip = f"10.{p}.{e}.{base + 1}/30"
                h_ip = f"10.{p}.{e}.{base + 2}/30"

                host = net.addHost(
                    f"h_p{p}_e{e}_{h}",
                    ip=h_ip,
                    defaultRoute=f"via {strip_mask(r_ip)}",
                )

                link = net.addLink(
                    r, host,
                    params1={"ip": r_ip},
                    params2={"ip": h_ip},
                    bw=250,
                )
                host_links.append((
                    r, link.intf1.name, r_ip,
                    host, link.intf2.name, h_ip,
                ))

        agg_per_pod[p] = agg
        edge_per_pod[p] = edge

    # --- Edge <-> Agg links ---
    info("*** Creating edge-agg links\n")
    for p in range(k):
        for er in edge_per_pod[p]:
            for ar in agg_per_pod[p]:
                ip1, ip2 = p2p_alloc.next31()
                link = net.addLink(er, ar,
                                   params1={"ip": ip1}, params2={"ip": ip2},
                                   bw=500)
                edge_agg_links.append((
                    er, link.intf1.name, ip1,
                    ar, link.intf2.name, ip2,
                ))

    # --- Agg <-> Core links ---
    info("*** Creating agg-core links\n")
    for p in range(k):
        for idx, ar in enumerate(agg_per_pod[p]):
            for c_r in core[idx * half:(idx + 1) * half]:
                ip1, ip2 = p2p_alloc.next31()
                link = net.addLink(ar, c_r,
                                   params1={"ip": ip1}, params2={"ip": ip2},
                                   bw=1000)
                agg_core_links.append((
                    ar, link.intf1.name, ip1,
                    c_r, link.intf2.name, ip2,
                ))

    return (core, agg_per_pod, edge_per_pod, all_routers,
            edge_agg_links, agg_core_links, host_links)


# ---------------------------------------------------------------------------
# Install static ECMP routes (replaces OSPF)
# ---------------------------------------------------------------------------

def install_static_routes(k, core, agg_per_pod, edge_per_pod,
                          edge_agg_links, agg_core_links):
    """
    Fat-tree routing with static ECMP:
      - Edge  -> default ECMP via all agg routers in pod
      - Agg   -> per-edge /24 routes + default ECMP via core routers
      - Core  -> per-pod /16 route via the connected agg in that pod
    """

    info("*** Installing static routes on edge routers\n")

    # Edge: default ECMP via all connected agg routers
    edge_to_agg_nexthops = defaultdict(list)
    for edge, _, _, agg, _, agg_ip in edge_agg_links:
        edge_to_agg_nexthops[edge.name].append(strip_mask(agg_ip))

    edge_nodes = {}
    for p in range(k):
        for er in edge_per_pod[p]:
            edge_nodes[er.name] = er

    for name, er in edge_nodes.items():
        nhops = edge_to_agg_nexthops[name]
        nexthop_str = " ".join(f"nexthop via {ip}" for ip in nhops)
        er.cmd(f"ip route add default {nexthop_str}")

    info("*** Installing static routes on agg routers\n")

    # Agg: in-pod /24 routes via edge routers + default ECMP via core routers
    agg_to_edge_routes = defaultdict(list)  # agg_name -> [(edge_ip, pod, edge_idx)]
    for edge, _, edge_ip, agg, _, _ in edge_agg_links:
        parts = edge.name.split("_")
        pod = int(parts[0][1:])
        eidx = int(parts[1][1:])
        agg_to_edge_routes[agg.name].append((strip_mask(edge_ip), pod, eidx))

    agg_to_core_nexthops = defaultdict(list)
    for agg, _, _, core_r, _, core_ip in agg_core_links:
        agg_to_core_nexthops[agg.name].append(strip_mask(core_ip))

    agg_nodes = {}
    for p in range(k):
        for ar in agg_per_pod[p]:
            agg_nodes[ar.name] = ar

    for name, ar in agg_nodes.items():
        # In-pod: specific route to each edge's host subnet
        cmds = []
        for edge_ip, pod, eidx in agg_to_edge_routes[name]:
            cmds.append(f"ip route add 10.{pod}.{eidx}.0/24 via {edge_ip}")

        # Cross-pod: default ECMP via core routers
        nhops = agg_to_core_nexthops[name]
        nexthop_str = " ".join(f"nexthop via {ip}" for ip in nhops)
        cmds.append(f"ip route add default {nexthop_str}")

        ar.cmd(" && ".join(cmds))

    info("*** Installing static routes on core routers\n")

    # Core: per-pod /16 route via connected agg router
    core_to_pod_routes = defaultdict(list)  # core_name -> [(agg_ip, pod)]
    for agg, _, agg_ip, core_r, _, _ in agg_core_links:
        parts = agg.name.split("_")
        pod = int(parts[0][1:])
        core_to_pod_routes[core_r.name].append((strip_mask(agg_ip), pod))

    for c_r in core:
        cmds = []
        for agg_ip, pod in core_to_pod_routes[c_r.name]:
            cmds.append(f"ip route add 10.{pod}.0.0/16 via {agg_ip}")
        c_r.cmd(" && ".join(cmds))

    info("*** Static route installation complete\n")


# ---------------------------------------------------------------------------
# Ring all-reduce: setup and launch
# ---------------------------------------------------------------------------

def build_ring_order(k):
    """
    Build ring order interleaving pods — adjacent ring neighbors are in
    different pods to exercise cross-pod fat-tree paths.
    Pattern: for each (edge, host_index), cycle through all pods.
    """
    half = k // 2
    ordered = []
    for e in range(half):
        for h in range(2, half + 2):
            for p in range(k):
                ordered.append(f"h_p{p}_e{e}_{h}")
    return ordered


def setup_and_start_ring(host_links, k):
    """
    Write ring_node.py to shared filesystem, build ring order,
    and launch one ring process per host.
    """

    # Collect unique hosts and their IPs
    host_info = {}
    for _, _, _, host, _, host_ip in host_links:
        ip = strip_mask(host_ip)
        host_info[host.name] = (host, ip)

    ordered_names = build_ring_order(k)
    ring = [host_info[name] for name in ordered_names]
    world_size = len(ring)
    info(f"*** Ring hosts (world_size={world_size})\n")

    # Gradient size (same elems_per_chunk as v1)
    elems_per_chunk = 1600
    grad_elems = world_size * elems_per_chunk

    base_port = 5000

    # Write ring_node.py once to shared filesystem
    info("*** Writing ring_node.py to shared filesystem\n")
    os.makedirs(RING_LOG_DIR, exist_ok=True)
    with open(RING_SCRIPT_PATH, "w") as f:
        f.write(RING_NODE_SCRIPT)
    os.chmod(RING_SCRIPT_PATH, 0o755)

    # Clear old log files
    for rank in range(world_size):
        log_file = os.path.join(RING_LOG_DIR, f"ring_rank{rank}.log")
        if os.path.exists(log_file):
            os.remove(log_file)

    # Start one ring process per host
    info("*** Starting ring all-reduce on all hosts\n")
    for rank, (host, ip) in enumerate(ring):
        _, right_ip = ring[(rank + 1) % world_size]

        listen_port = base_port + rank
        right_port = base_port + ((rank + 1) % world_size)
        log_file = os.path.join(RING_LOG_DIR, f"ring_rank{rank}.log")

        cmd = (
            f"python3 {RING_SCRIPT_PATH} "
            f"--rank {rank} --world-size {world_size} "
            f"--listen-port {listen_port} "
            f"--right-ip {right_ip} --right-port {right_port} "
            f"--grad-elems {grad_elems} "
            f"--log-file {log_file} "
            "&"
        )
        host.cmd(cmd)

    info("*** Ring all-reduce processes launched (background in each host)\n")


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_ring_metrics(host_links, k):
    """
    Parse per-rank ring logs and compute:
      - total latency  = max rank latency (slowest rank)
      - total throughput = logical gradient bytes / total latency
      - max tail = max_latency - min_latency
    """

    info("*** Sleeping 60s for all-reduce completion (128 hosts)\n")
    time.sleep(60)

    ordered_names = build_ring_order(k)
    world_size = len(ordered_names)

    elems_per_chunk = 1600
    grad_elems = world_size * elems_per_chunk
    gradient_bytes = grad_elems * 4.0

    latencies = []

    for rank in range(world_size):
        log_file = os.path.join(RING_LOG_DIR, f"ring_rank{rank}.log")
        try:
            with open(log_file, "r") as f:
                content = f.read()
        except FileNotFoundError:
            info(f"*** WARNING: no log file for rank {rank}\n")
            continue

        matches = re.findall(r"TOTAL_TIME_SEC=([0-9.]+)", content)
        if not matches:
            info(f"*** WARNING: no TOTAL_TIME_SEC in rank {rank} log\n")
            continue

        latency = float(matches[-1])
        latencies.append(latency)

    if not latencies:
        info("*** Ring metrics: no latency data collected from logs\n")
        return

    fastest = min(latencies)
    slowest = max(latencies)
    max_tail = slowest - fastest
    total_latency = slowest

    total_throughput_bytes_per_sec = gradient_bytes / total_latency
    total_throughput_mib_per_sec = total_throughput_bytes_per_sec / (1024.0 * 1024.0)
    gradient_mib = gradient_bytes / (1024.0 * 1024.0)

    info("*** Ring all-reduce metrics:\n")
    info(f"    ranks with data:            {len(latencies)}/{world_size}\n")
    info(f"    total latency (slowest):    {total_latency:.6f} s\n")
    info(f"    max tail (slowest-fastest): {max_tail:.6f} s\n")
    info(f"    logical gradient size:      {gradient_mib:.3f} MiB\n")
    info(f"    total throughput:           {total_throughput_mib_per_sec:.3f} MiB/s\n")

    try:
        with open("ring_metrics.log", "a") as f:
            f.write("Ring all-reduce metrics (k=8, 128 hosts):\n")
            f.write(f"  ranks_with_data={len(latencies)}/{world_size}\n")
            f.write(f"  total_latency_s={total_latency:.6f}\n")
            f.write(f"  max_tail_s={max_tail:.6f}\n")
            f.write(f"  gradient_mib={gradient_mib:.3f}\n")
            f.write(f"  total_throughput_mib_s={total_throughput_mib_per_sec:.3f}\n")
            f.write("\n")
    except Exception as e:
        info(f"*** WARNING: failed to write ring_metrics.log: {e}\n")


# ---------------------------------------------------------------------------
# Per-link measurements
# ---------------------------------------------------------------------------

_ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def read_if_counters(node, ifname):
    """Read tx_bytes and rx_bytes for a given interface inside a Mininet node."""
    base = f"/sys/class/net/{ifname}/statistics"
    out = node.cmd(
        f"cat {base}/tx_bytes {base}/rx_bytes 2>&1 || echo '0 0'"
    ).strip()
    clean = _ansi.sub("", out).replace("\r", "")
    nums = re.findall(r"\b\d+\b", clean)

    try:
        if len(nums) >= 2:
            return int(nums[-2]), int(nums[-1])
    except ValueError:
        pass
    return 0, 0


def snapshot_all_link_counters(all_p2p_links, host_links):
    """Take a snapshot of tx/rx counters for all interfaces."""
    snap = {}

    for n1, if1, _, n2, if2, _ in all_p2p_links:
        snap[(n1.name, if1)] = read_if_counters(n1, if1)
        snap[(n2.name, if2)] = read_if_counters(n2, if2)

    for n1, if1, _, n2, if2, _ in host_links:
        snap[(n1.name, if1)] = read_if_counters(n1, if1)
        snap[(n2.name, if2)] = read_if_counters(n2, if2)

    return snap


def report_raw_link_load(before_snap, all_p2p_links, host_links):
    """Compare current counters to before_snap and log byte deltas per interface."""
    info("*** Raw link load (tx_bytes, rx_bytes per interface during ring) ***\n")

    seen = set()

    def report_for(node, ifname, role):
        key = (node.name, ifname)
        if key in seen:
            return
        seen.add(key)

        tx0, rx0 = before_snap.get(key, (0, 0))
        tx1, rx1 = read_if_counters(node, ifname)
        dtx = tx1 - tx0
        drx = rx1 - rx0

        try:
            with open("link_load.log", "a") as f:
                f.write(f"  {role} {node.name}:{ifname}  Δtx={dtx} B  Δrx={drx} B\n")
        except Exception as e:
            info(f"*** WARNING: failed to write link_load.log: {e}\n")

    for n1, if1, _, n2, if2, _ in all_p2p_links:
        report_for(n1, if1, "R-R")
        report_for(n2, if2, "R-R")

    for n1, if1, _, n2, if2, _ in host_links:
        report_for(n1, if1, "R-H")
        report_for(n2, if2, "R-H")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    setLogLevel("info")

    k = 8

    net = Mininet(controller=None, link=TCLink, autoSetMacs=True)
    p2p_alloc = P2PAllocator()

    (core, agg_per_pod, edge_per_pod, all_routers,
     edge_agg_links, agg_core_links, host_links) = build_fattree(net, k, p2p_alloc)

    net.start()

    # Enforce correct IPs on all interfaces
    enforce_all_ips(edge_agg_links, agg_core_links, host_links)

    # Install static ECMP routes (replaces OSPF — no convergence wait needed)
    install_static_routes(k, core, agg_per_pod, edge_per_pod,
                          edge_agg_links, agg_core_links)

    # Small sleep to let interfaces settle
    info("*** Sleeping 3s for interfaces to settle\n")
    time.sleep(3)

    # Take baseline link counters before starting the ring
    all_p2p_links = edge_agg_links + agg_core_links
    link_counters_before = snapshot_all_link_counters(all_p2p_links, host_links)

    # Start ring all-reduce
    setup_and_start_ring(host_links, k)

    # Drop into CLI while ring traffic is running
    info("*** Starting CLI (ring all-reduce is running in background)\n")
    CLI(net)

    # After exiting CLI, collect metrics
    info("*** Collecting ring all-reduce metrics\n")
    collect_ring_metrics(host_links, k)

    # Report raw link load
    info("*** Collecting raw link load\n")
    report_raw_link_load(link_counters_before, all_p2p_links, host_links)

    net.stop()


if __name__ == "__main__":
    run()
