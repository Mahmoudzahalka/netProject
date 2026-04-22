#!/usr/bin/env python3
"""
tree_simulation_mininet_k8.py

- Builds a k=8 L3 Fat-Tree using plain Mininet with LinuxRouter nodes.
- Static ECMP routes (no OSPF/FRR needed).
- 128 hosts running TCP-based hierarchical tree all-reduce.
- Tree structure mirrors the physical fat-tree: pod -> edge -> host.
- Topology: 16 core + 32 agg + 32 edge routers + 128 hosts = 208 nodes.
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
# Host-side hierarchical tree all-reduce script
# ---------------------------------------------------------------------------

TREE_NODE_SCRIPT = r"""#!/usr/bin/env python3
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


def parse_children(children_str):
    if not children_str:
        return []
    out = []
    for token in children_str.split(","):
        token = token.strip()
        if token:
            rank_str, ip, port_str = token.split(":")
            out.append((int(rank_str), ip, int(port_str)))
    return out


def establish_tree_connections(rank, listen_port, parent_ip, parent_port, children, log):
    expected_children = len(children)
    child_rank_set = {child_rank for child_rank, _, _ in children}

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", listen_port))
    server.listen(max(1, expected_children))

    parent_sock = None
    if parent_ip:
        while True:
            try:
                parent_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                parent_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                parent_sock.connect((parent_ip, parent_port))
                parent_sock.sendall(rank.to_bytes(4, byteorder="big", signed=False))
                _ = recv_all(parent_sock, 1)
                break
            except Exception as e:
                log(f"[rank {rank}] connect to parent failed ({e}), retrying...")
                time.sleep(0.1)

    child_socks = []
    while len(child_socks) < expected_children:
        conn, _ = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        child_rank = int.from_bytes(recv_all(conn, 4), byteorder="big", signed=False)
        if child_rank not in child_rank_set:
            conn.close()
            raise RuntimeError(f"Unexpected child rank {child_rank} for rank {rank}")
        conn.sendall(b"H")
        child_socks.append((child_rank, conn))

    child_socks.sort(key=lambda x: x[0])
    server.close()
    return parent_sock, [sock for _, sock in child_socks]


def recv_full_grad(sock, grad_elems):
    in_bytes = recv_all(sock, grad_elems * 4)
    grad = array("f")
    grad.frombytes(in_bytes)
    return grad


def send_full_grad(sock, grad):
    sock.sendall(grad.tobytes())


def tree_barrier(parent_sock, child_socks):
    # Phase 1: gather — leaves send 'R' up, parents wait for all children then send up
    for child_sock in child_socks:
        recv_all(child_sock, 1)
    if parent_sock is not None:
        parent_sock.sendall(b"R")
    # Phase 2: broadcast — root sends 'G' down, children forward down
    if parent_sock is not None:
        recv_all(parent_sock, 1)
    for child_sock in child_socks:
        child_sock.sendall(b"G")


def tree_allreduce(rank, parent_sock, child_socks, grad_elems, log):
    grad = array("f", [float(rank)] * grad_elems)

    for child_sock in child_socks:
        child_grad = recv_full_grad(child_sock, grad_elems)
        grad = child_grad

    if parent_sock is not None:
        send_full_grad(parent_sock, grad)
        grad = recv_full_grad(parent_sock, grad_elems)

    for child_sock in child_socks:
        send_full_grad(child_sock, grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--parent-ip", type=str, default="")
    parser.add_argument("--parent-port", type=int, default=0)
    parser.add_argument("--children", type=str, default="")
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

    children = parse_children(args.children)
    parent_sock, child_socks = establish_tree_connections(
        args.rank,
        args.listen_port,
        args.parent_ip,
        args.parent_port,
        children,
        log,
    )

    # Barrier: wait until all ranks have finished connection setup before starting
    tree_barrier(parent_sock, child_socks)

    t0 = time.time()
    tree_allreduce(args.rank, parent_sock, child_socks, args.grad_elems, log)
    t1 = time.time()

    log(f"[rank {args.rank}] TOTAL_TIME_SEC={t1 - t0:.6f}")

    if parent_sock is not None:
        parent_sock.close()
    for child_sock in child_socks:
        child_sock.close()


if __name__ == "__main__":
    main()
"""

TREE_SCRIPT_PATH = "/tmp/tree_allreduce/tree_node.py"
TREE_LOG_DIR = "/tmp/tree_allreduce"


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
        A = n // 128
        B = n % 128
        ip1 = f"172.16.{A}.{2 * B}/31"
        ip2 = f"172.16.{A}.{2 * B + 1}/31"
        return ip1, ip2


def strip_mask(ip_with_mask):
    return ip_with_mask.split("/")[0]


# ---------------------------------------------------------------------------
# Enforce correct IPs on all interfaces after net.start()
# ---------------------------------------------------------------------------

def enforce_all_ips(edge_agg_links, agg_core_links, host_links):
    info("*** Enforcing IPs on all P2P interfaces\n")
    for n1, i1, ip1, n2, i2, ip2 in edge_agg_links + agg_core_links:
        n1.cmd(f"ip addr flush dev {i1} && ip addr add {ip1} dev {i1}")
        n2.cmd(f"ip addr flush dev {i2} && ip addr add {ip2} dev {i2}")

    info("*** Enforcing IPs on all host interfaces\n")
    for edge, edge_intf, edge_ip, host, host_intf, host_ip in host_links:
        edge.cmd(f"ip addr flush dev {edge_intf} && ip addr add {edge_ip} dev {edge_intf}")
        host.cmd(f"ip addr flush dev {host_intf} && ip addr add {host_ip} dev {host_intf}")
        gw = strip_mask(edge_ip)
        host.cmd(f"ip route replace default via {gw}")

    info("*** Clearing stale routes on all routers\n")
    seen = set()
    for n1, _, _, n2, _, _ in edge_agg_links + agg_core_links:
        for node in (n1, n2):
            if node.name not in seen:
                seen.add(node.name)
                node.cmd("ip route del default 2>/dev/null || true")


# ---------------------------------------------------------------------------
# Build Fat-Tree
# ---------------------------------------------------------------------------

def build_fattree(net, k, p2p_alloc):
    core = []
    agg_per_pod = {}
    edge_per_pod = {}
    all_routers = []
    edge_agg_links = []
    agg_core_links = []
    host_links = []

    half = k // 2

    info(f"*** Creating {half ** 2} core routers\n")
    for i in range(half ** 2):
        r = net.addHost(f"c{i}", cls=LinuxRouter)
        core.append(r)
        all_routers.append(r)

    for p in range(k):
        agg = []
        edge = []

        for a in range(half, k):
            r = net.addHost(f"p{p}_a{a}", cls=LinuxRouter)
            agg.append(r)
            all_routers.append(r)

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
# Install static ECMP routes
# ---------------------------------------------------------------------------

def install_static_routes(k, core, agg_per_pod, edge_per_pod,
                          edge_agg_links, agg_core_links):
    info("*** Installing static routes on edge routers\n")

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
        er.cmd(f"ip route del default 2>/dev/null || true")
        er.cmd(f"ip route add default {nexthop_str}")

    info("*** Installing static routes on agg routers\n")

    agg_to_edge_routes = defaultdict(list)
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
        cmds = []
        for edge_ip, pod, eidx in agg_to_edge_routes[name]:
            cmds.append(f"ip route replace 10.{pod}.{eidx}.0/24 via {edge_ip}")

        nhops = agg_to_core_nexthops[name]
        nexthop_str = " ".join(f"nexthop via {ip}" for ip in nhops)
        cmds.append(f"ip route del default 2>/dev/null || true")
        cmds.append(f"ip route add default {nexthop_str}")

        ar.cmd(" ; ".join(cmds))

    info("*** Installing static routes on core routers\n")

    core_to_pod_routes = defaultdict(list)
    for agg, _, agg_ip, core_r, _, _ in agg_core_links:
        parts = agg.name.split("_")
        pod = int(parts[0][1:])
        core_to_pod_routes[core_r.name].append((strip_mask(agg_ip), pod))

    for c_r in core:
        cmds = []
        for agg_ip, pod in core_to_pod_routes[c_r.name]:
            cmds.append(f"ip route replace 10.{pod}.0.0/16 via {agg_ip}")
        c_r.cmd(" && ".join(cmds))

    info("*** Static route installation complete\n")


# ---------------------------------------------------------------------------
# Verify L3 routing
# ---------------------------------------------------------------------------

def verify_l3_routing(k, core, agg_per_pod, edge_per_pod, host_links):
    info("\n")
    info("=" * 70 + "\n")
    info("  L3 ROUTING VERIFICATION\n")
    info("=" * 70 + "\n\n")

    sample_edge = edge_per_pod[0][0]
    sample_agg = agg_per_pod[0][0]
    sample_core = core[0]

    info("*** [1/4] IP forwarding check on sample routers:\n")
    for r in [sample_edge, sample_agg, sample_core]:
        fwd = r.cmd("cat /proc/sys/net/ipv4/ip_forward").strip()
        ecmp = r.cmd("cat /proc/sys/net/ipv4/fib_multipath_hash_policy").strip()
        info(f"    {r.name:12s}  ip_forward={fwd}  fib_multipath_hash_policy={ecmp}\n")
    info("\n")

    info(f"*** [2/4] Routing table: EDGE router {sample_edge.name}\n")
    info(f"    (expect: default ECMP via {k // 2} agg routers)\n")
    info(sample_edge.cmd("ip route show") + "\n")

    info(f"*** [2/4] Routing table: AGG router {sample_agg.name}\n")
    info(f"    (expect: /24 routes to edge subnets + default ECMP via {k // 2} cores)\n")
    info(sample_agg.cmd("ip route show") + "\n")

    info(f"*** [2/4] Routing table: CORE router {sample_core.name}\n")
    info(f"    (expect: {k} per-pod /16 routes, one per pod)\n")
    info(sample_core.cmd("ip route show") + "\n")

    host_info = {}
    for _, _, _, host, _, host_ip in host_links:
        host_info[host.name] = (host, strip_mask(host_ip))

    src_name = "h_p0_e0_2"
    dst_name = f"h_p{k - 1}_e0_2"
    src_host, src_ip = host_info[src_name]
    _, dst_ip = host_info[dst_name]

    info(f"*** [3/4] Cross-pod ping: {src_name} ({src_ip}) -> {dst_name} ({dst_ip})\n")
    result = src_host.cmd(f"ping -c 3 -W 2 {dst_ip}")
    info(result + "\n")

    info(f"*** [4/4] Traceroute: {src_name} -> {dst_name}\n")
    info(f"    (expect 5 L3 hops: edge->agg->core->agg->edge)\n")
    result = src_host.cmd(f"traceroute -n -m 10 -w 2 {dst_ip}")
    info(result + "\n")

    info("=" * 70 + "\n")
    info("  L3 VERIFICATION COMPLETE\n")
    info("=" * 70 + "\n\n")


# ---------------------------------------------------------------------------
# Tree topology — pod-aware hierarchical all-reduce
# ---------------------------------------------------------------------------

def build_tree_structure(k):
    """
    Build a hierarchical tree that mirrors the fat-tree physical topology:
      - Root: h_p0_e0_2 (pod 0, edge 0, host 2)
      - Root's children:
          * same-edge siblings (leaves): h_p0_e0_{3..half+1}
          * other-edge roots in pod 0: h_p0_e{1..half-1}_2
          * other-pod roots: h_p{1..k-1}_e0_2
      - Pod root's children: same-edge siblings + other-edge roots in same pod
      - Edge root's children: same-edge siblings (leaves)
    """
    half = k // 2
    children_by_name = {}

    def leaves_of(p, e):
        return [f"h_p{p}_e{e}_{h}" for h in range(3, half + 2)]

    # Root: h_p0_e0_2
    root_children = []
    root_children.extend(leaves_of(0, 0))

    # Other edges in pod 0 -> edge roots
    for e in range(1, half):
        edge_root = f"h_p0_e{e}_2"
        root_children.append(edge_root)
        children_by_name[edge_root] = leaves_of(0, e)

    # Other pods -> pod roots
    for p in range(1, k):
        pod_root = f"h_p{p}_e0_2"
        root_children.append(pod_root)

        pod_root_children = []
        pod_root_children.extend(leaves_of(p, 0))
        for e in range(1, half):
            edge_root = f"h_p{p}_e{e}_2"
            pod_root_children.append(edge_root)
            children_by_name[edge_root] = leaves_of(p, e)
        children_by_name[pod_root] = pod_root_children

    children_by_name["h_p0_e0_2"] = root_children
    return children_by_name


def build_host_order(k):
    """All host names in a stable deterministic order (for rank assignment)."""
    half = k // 2
    ordered = []
    for e in range(half):
        for h in range(2, half + 2):
            for p in range(k):
                ordered.append(f"h_p{p}_e{e}_{h}")
    return ordered


# ---------------------------------------------------------------------------
# Tree all-reduce setup and launch
# ---------------------------------------------------------------------------

def setup_and_start_tree(host_links, k):
    host_info = {}
    for _, _, _, host, _, host_ip in host_links:
        ip = strip_mask(host_ip)
        host_info[host.name] = (host, ip)

    ordered_names = build_host_order(k)
    tree_hosts = [host_info[name] for name in ordered_names]
    world_size = len(tree_hosts)
    info(f"*** Tree hosts (world_size={world_size})\n")

    elems_per_chunk = 1600  # 1600 * 128 * 4 = 819,200 bytes (~800 KiB per host)
    grad_elems = world_size * elems_per_chunk

    base_port = 5000
    ranks = {name: rank for rank, name in enumerate(ordered_names)}

    children_by_name = build_tree_structure(k)
    parent_by_name = {}
    for parent_name, child_names in children_by_name.items():
        for child_name in child_names:
            parent_by_name[child_name] = parent_name

    # Sanity check: every non-root host must have a parent
    missing = [n for n in ordered_names if n != "h_p0_e0_2" and n not in parent_by_name]
    if missing:
        info(f"*** WARNING: {len(missing)} hosts missing a parent: {missing[:5]}...\n")

    # Write tree_node.py once to shared filesystem
    info("*** Writing tree_node.py to shared filesystem\n")
    os.makedirs(TREE_LOG_DIR, exist_ok=True)
    with open(TREE_SCRIPT_PATH, "w") as f:
        f.write(TREE_NODE_SCRIPT)
    os.chmod(TREE_SCRIPT_PATH, 0o755)

    # Clear old log files
    for rank in range(world_size):
        log_file = os.path.join(TREE_LOG_DIR, f"tree_rank{rank}.log")
        if os.path.exists(log_file):
            os.remove(log_file)

    info(f"*** Starting hierarchical tree all-reduce on all {world_size} hosts\n")
    for rank, (host, ip) in enumerate(tree_hosts):
        host_name = ordered_names[rank]
        listen_port = base_port + rank
        log_file = os.path.join(TREE_LOG_DIR, f"tree_rank{rank}.log")

        parent_name = parent_by_name.get(host_name)
        if parent_name is None:
            parent_ip = ""
            parent_port = 0
        else:
            _, parent_ip = host_info[parent_name]
            parent_port = base_port + ranks[parent_name]

        child_specs = []
        for child_name in children_by_name.get(host_name, []):
            child_rank = ranks[child_name]
            _, child_ip = host_info[child_name]
            child_port = base_port + child_rank
            child_specs.append(f"{child_rank}:{child_ip}:{child_port}")
        children_arg = ",".join(child_specs)

        cmd = (
            f"python3 {TREE_SCRIPT_PATH} "
            f"--rank {rank} --world-size {world_size} "
            f"--listen-port {listen_port} "
            f"--parent-ip '{parent_ip}' --parent-port {parent_port} "
            f"--children '{children_arg}' "
            f"--grad-elems {grad_elems} "
            f"--log-file {log_file} "
            "&"
        )
        host.cmd(cmd)

    info("*** Tree all-reduce processes launched (background in each host)\n")


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_tree_metrics(host_links, k):
    info("*** Sleeping 60s for tree all-reduce completion\n")
    time.sleep(60)

    ordered_names = build_host_order(k)
    world_size = len(ordered_names)

    elems_per_chunk = 1600
    grad_elems = world_size * elems_per_chunk
    gradient_bytes = grad_elems * 4.0

    latencies = []

    for rank in range(world_size):
        log_file = os.path.join(TREE_LOG_DIR, f"tree_rank{rank}.log")
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

        latencies.append(float(matches[-1]))

    if not latencies:
        info("*** Tree metrics: no latency data collected from logs\n")
        return

    fastest = min(latencies)
    slowest = max(latencies)
    max_tail = slowest - fastest
    total_latency = slowest

    total_throughput_bytes_per_sec = gradient_bytes / total_latency
    total_throughput_mib_per_sec = total_throughput_bytes_per_sec / (1024.0 * 1024.0)
    gradient_mib = gradient_bytes / (1024.0 * 1024.0)

    info("*** Hierarchical tree all-reduce metrics:\n")
    info(f"    ranks with data:            {len(latencies)}/{world_size}\n")
    info(f"    total latency (slowest):    {total_latency:.6f} s\n")
    info(f"    max tail (slowest-fastest): {max_tail:.6f} s\n")
    info(f"    logical gradient size:      {gradient_mib:.3f} MiB\n")
    info(f"    total throughput:           {total_throughput_mib_per_sec:.3f} MiB/s\n")

    try:
        with open("tree_metrics.log", "a") as f:
            f.write("Hierarchical tree all-reduce metrics (k=8, 128 hosts):\n")
            f.write(f"  ranks_with_data={len(latencies)}/{world_size}\n")
            f.write(f"  total_latency_s={total_latency:.6f}\n")
            f.write(f"  max_tail_s={max_tail:.6f}\n")
            f.write(f"  gradient_mib={gradient_mib:.3f}\n")
            f.write(f"  total_throughput_mib_s={total_throughput_mib_per_sec:.3f}\n")
            f.write("\n")
    except Exception as e:
        info(f"*** WARNING: failed to write tree_metrics.log: {e}\n")


# ---------------------------------------------------------------------------
# Per-link measurements
# ---------------------------------------------------------------------------

_ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def read_if_counters(node, ifname):
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
    snap = {}

    for n1, if1, _, n2, if2, _ in all_p2p_links:
        snap[(n1.name, if1)] = read_if_counters(n1, if1)
        snap[(n2.name, if2)] = read_if_counters(n2, if2)

    for n1, if1, _, n2, if2, _ in host_links:
        snap[(n1.name, if1)] = read_if_counters(n1, if1)
        snap[(n2.name, if2)] = read_if_counters(n2, if2)

    return snap


def report_raw_link_load(before_snap, all_p2p_links, host_links):
    info("*** Raw link load (tx_bytes, rx_bytes per interface during tree all-reduce) ***\n")

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
            with open("tree_link_load.log", "a") as f:
                f.write(f"  {role} {node.name}:{ifname}  Δtx={dtx} B  Δrx={drx} B\n")
        except Exception as e:
            info(f"*** WARNING: failed to write tree_link_load.log: {e}\n")

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

    enforce_all_ips(edge_agg_links, agg_core_links, host_links)

    install_static_routes(k, core, agg_per_pod, edge_per_pod,
                          edge_agg_links, agg_core_links)

    info("*** Sleeping 3s for interfaces to settle\n")
    time.sleep(3)

    verify_l3_routing(k, core, agg_per_pod, edge_per_pod, host_links)

    all_p2p_links = edge_agg_links + agg_core_links
    link_counters_before = snapshot_all_link_counters(all_p2p_links, host_links)

    setup_and_start_tree(host_links, k)

    info("*** Starting CLI (hierarchical tree all-reduce is running in background)\n")
    CLI(net)

    info("*** Collecting hierarchical tree all-reduce metrics\n")
    collect_tree_metrics(host_links, k)

    info("*** Collecting raw link load\n")
    report_raw_link_load(link_counters_before, all_p2p_links, host_links)

    net.stop()


if __name__ == "__main__":
    run()
