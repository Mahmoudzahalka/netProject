#!/usr/bin/env python3
"""
fattree_containernet_frr_ring.py

- Builds k=4 L3 Fat-Tree with FRR/OSPF/ECMP (same as your current script).
- Uses Ubuntu-based hosts (HOST_IMG).
- Automatically runs a TCP-based ring all-reduce across all hosts.
- Ring code has an explicit Phase 0 where only TCP connections are established,
  and *only then* do we start exchanging gradient chunks.
"""

import os
from time import sleep
import re 

from mininet.net import Containernet
from mininet.node import Docker
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel

# ---------------------------------------------------------------------------
# Host-side ring all-reduce script, injected into each host container
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
    

    #log(f"[rank {rank}] Phase 0: setting up sockets (listen_port={listen_port}, "
        #f"right={right_ip}:{right_port})")

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
    #log(f"[rank {rank}] accepted connection from left neighbor {addr_left}")

    # One-byte handshake to be sure both directions are live
    sock_right.sendall(b"H")
    _ = recv_all(conn_left, 1)

    #log(f"[rank {rank}] Phase 0: connections established to left and right.")
    server.close()
    return conn_left, sock_right

def ring_allreduce(rank, world_size, conn_left, conn_right, grad_elems, log):
    

    # Make grad_elems divisible by world_size (enforced in caller)
    elems_per_chunk = grad_elems // world_size
    assert grad_elems % world_size == 0

    # Local gradient initialized with rank (just for sanity/testing)
    grad = array("f", [float(rank)] * grad_elems)

    # Each float32 is 4 bytes
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
            # scatter-reduce: sum into local gradient chunk
            #for i in range(elems_per_chunk): 
                #grad[start + i] += in_chunk[i]
            grad[start : start + elems_per_chunk] = in_chunk #this compute is a major bottleneck, difference of 0.7 seconds for 4mb gradients, for bigger gradients it will just get bigger
        else:
            # all-gather: overwrite with fully reduced chunk
            grad[start : start + elems_per_chunk] = in_chunk

    # -------------------------
    # Phase 1: scatter-reduce
    # -------------------------
    #log(f"[rank {rank}] Phase 1: scatter-reduce starting "
        #f"(grad_elems={grad_elems}, chunks={world_size}, chunk_bytes={chunk_bytes})")

    for step in range(world_size - 1):
        # Index of chunk we send this step
        send_index = (rank - step) % world_size
        # Index of chunk we receive and reduce this step
        recv_index = (rank - step - 1) % world_size

        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=True)

        #log(f"[rank {rank}] Phase 1 step {step+1}/{world_size-1}: "
            #f"sent chunk {send_index}, reduced chunk {recv_index}")

    # -------------------------
    # Phase 2: all-gather
    # -------------------------
    #log(f"[rank {rank}] Phase 2: all-gather starting")

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size

        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=False)

        #log(f"[rank {rank}] Phase 2 step {step+1}/{world_size-1}: "
            #f"sent chunk {send_index}, received fully-reduced chunk {recv_index}")

    #log(f"[rank {rank}] Ring all-reduce completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--right-ip", type=str, required=True)
    parser.add_argument("--right-port", type=int, required=True)
    # Total gradient length (float32 elements). Must be divisible by world_size.
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

    #log(f"[rank {args.rank}] Starting ring node with world_size={args.world_size}, "
        #f"listen_port={args.listen_port}, right={args.right_ip}:{args.right_port}, "
        #f"grad_elems={args.grad_elems}")

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
    #log(f"[rank {args.rank}] Connections closed, exiting.")

if __name__ == "__main__":
    main()
"""

# ---------------------------------------------------------------------------
# Unique /31 allocator for P2P links
# ---------------------------------------------------------------------------

class P2PAllocator:
    def __init__(self):
        self.n = 0

    def next31(self):
        n = self.n
        self.n += 1
        A = n // 256
        B = n % 256
        ip1 = f"172.16.{A}.{2 * B}/31"
        ip2 = f"172.16.{A}.{2 * B + 1}/31"
        return ip1, ip2


# ---------------------------------------------------------------------------
# Detect router-router (P2P) interfaces by IP subnet
# ---------------------------------------------------------------------------

def get_p2p_intfs(router):
    p2p = []
    for intf in router.intfNames():
        out = router.cmd(f"ip -o -4 addr show dev {intf}").strip()
        if "inet 172.16." in out and "/31" in out:
            p2p.append(intf)
    return p2p


# ---------------------------------------------------------------------------
# Force correct /31 IPs on router-router links
# ---------------------------------------------------------------------------

def enforce_p2p_ips(p2p_links):
    for n1, i1, ip1, n2, i2, ip2 in p2p_links:
        n1.cmd(f"ip addr flush dev {i1}")
        n1.cmd(f"ip addr add {ip1} dev {i1}")
        n2.cmd(f"ip addr flush dev {i2}")
        n2.cmd(f"ip addr add {ip2} dev {i2}")


# ---------------------------
# Force correct /30 IPs on edge-router <-> host links
# ---------------------------
def enforce_host_ips(host_links):
    """
    Ensure that host-facing links get the intended /30 addresses on the correct interfaces.
    This eliminates interface-order ambiguity (which edge ethX got which host link).
    """
    for edge, edge_intf, edge_ip, host, host_intf, host_ip in host_links:
        edge.cmd(f"ip addr flush dev {edge_intf}")
        edge.cmd(f"ip addr add {edge_ip} dev {edge_intf}")

        host.cmd(f"ip addr flush dev {host_intf}")
        host.cmd(f"ip addr add {host_ip} dev {host_intf}")
        gw = edge_ip.split("/")[0] #Mahmoud : addded back the default route
        host.cmd(f"ip route add default via {gw}")

# ---------------------------------------------------------------------------
# Generate ospfd.conf
# ---------------------------------------------------------------------------

def generate_ospfd_conf(router_name, role, rid_suffix, p2p_intfs):
    base = f"""hostname {router_name}
password zebra
log file /var/log/frr/ospfd.log

router ospf
 ospf router-id 1.1.1.{rid_suffix}
 maximum-paths 8
 passive-interface default
"""

    # All routers: run OSPF on all 172.16.* P2P links
    base += " network 172.16.0.0/16 area 0\n"

    # Edge routers: also run OSPF on the 10.* host-facing links
    if role == "edge":
        base += " network 10.0.0.0/8 area 0\n"
        # Optional: redistribute connected (not strictly needed, but fine)
        base += " redistribute connected\n"

    # Un-passive the P2P router-router interfaces
    for intf in p2p_intfs:
        base += f" no passive-interface {intf}\n"

    base += "\n"

    # Set P2P network type on router-router links
    for intf in p2p_intfs:
        base += f"""interface {intf}
 ip ospf network point-to-point

"""

    return base


# ---------------------------------------------------------------------------
# Start FRR
# ---------------------------------------------------------------------------

def start_frr_ospf(router, rid_suffix):
    name = router.name

    router.cmd("mkdir -p /etc/frr /var/log/frr /var/run/frr")
    router.cmd("chown -R frr:frr /etc/frr /var/log/frr /var/run/frr || true")
    router.cmd("sysctl -w net.ipv4.ip_forward=1")

    zebra_conf = f"""hostname {name}
password zebra
log file /var/log/frr/zebra.log
"""
    router.cmd(f"printf '%s\n' '{zebra_conf}' > /etc/frr/zebra.conf")

    if "_e" in name:
        role = "edge"
    elif "_a" in name:
        role = "agg"
    elif name.startswith("c"):
        role = "core"
    else:
        raise RuntimeError(f"Unknown router role for {name}")

    p2p_intfs = get_p2p_intfs(router)
    ospf_conf = generate_ospfd_conf(name, role, rid_suffix, p2p_intfs)
    router.cmd(f"printf '%s\n' '{ospf_conf}' > /etc/frr/ospfd.conf")

    router.cmd("touch /etc/frr/vtysh.conf")
    router.cmd("chown -R frr:frr /etc/frr /var/log/frr /var/run/frr || true")

    router.cmd("sed -i 's/^zebra=no/zebra=yes/' /etc/frr/daemons || true")
    router.cmd("sed -i 's/^ospfd=no/ospfd=yes/' /etc/frr/daemons || true")

    router.cmd("/usr/lib/frr/watchfrr -d zebra ospfd")


# ---------------------------------------------------------------------------
# Build k=4 Fat-Tree
# ---------------------------------------------------------------------------

def build_fattree_k4(net, p2p_alloc, ROUTER_IMG, HOST_IMG):
    k = 4

    core = []
    agg_per_pod = {}
    edge_per_pod = {}
    all_routers = []
    p2p_links = []
    host_links = []

    info("*** Creating core routers\n")
    for i in range((k // 2) ** 2):
        r = net.addDocker(f"c{i}", dimage=ROUTER_IMG, privileged=True,
                          cap_add=["NET_ADMIN", "SYS_ADMIN"])
        core.append(r)
        all_routers.append(r)

    for p in range(k):
        agg = []
        edge = []

        for a in range(k // 2, k):
            r = net.addDocker(f"p{p}_a{a}", dimage=ROUTER_IMG,
                              privileged=True, cap_add=["NET_ADMIN", "SYS_ADMIN"])
            agg.append(r)
            all_routers.append(r)

        for e in range(k // 2):
            r = net.addDocker(f"p{p}_e{e}", dimage=ROUTER_IMG,
                              privileged=True, cap_add=["NET_ADMIN", "SYS_ADMIN"])
            edge.append(r)
            all_routers.append(r)

            for h in range(2, (k // 2) + 2):
                # give each host its own /30 P2P subnet
                base = 4 * h          # h=2,3 → bases 8,12
                r_ip = f"10.{p}.{e}.{base+1}/30"
                h_ip = f"10.{p}.{e}.{base+2}/30"

                host = net.addDocker(
                    f"h_p{p}_e{e}_{h}",
                    dimage=HOST_IMG,
                    ip=h_ip,
                    defaultRoute=f"via 10.{p}.{e}.{base+1}",
                    cap_add=["NET_ADMIN"],
                )

                link = net.addLink(
                    r,
                    host,
                    params1={"ip": r_ip},
                    params2={"ip": h_ip},
                    bw=250,  # 1 Gbps bandwidth
                )

                # record the *actual* interface names created for this link
                host_links.append((r, link.intf1.name, r_ip, host, link.intf2.name, h_ip))

        agg_per_pod[p] = agg
        edge_per_pod[p] = edge

    # Edge <-> Agg p2p links
    for p in range(k):
        for er in edge_per_pod[p]:
            for ar in agg_per_pod[p]:
                ip1, ip2 = p2p_alloc.next31()
                link = net.addLink(er, ar, params1={"ip": ip1}, params2={"ip": ip2},bw=500)
                p2p_links.append((er, link.intf1.name, ip1,
                                  ar, link.intf2.name, ip2))

    # Agg <-> Core p2p links
    half = k // 2
    for p in range(k):
        for idx, ar in enumerate(agg_per_pod[p]):
            for c_r in core[idx * half:(idx + 1) * half]:
                ip1, ip2 = p2p_alloc.next31()
                link = net.addLink(ar, c_r, params1={"ip": ip1}, params2={"ip": ip2},bw=1000)
                p2p_links.append((ar, link.intf1.name, ip1,
                                  c_r, link.intf2.name, ip2))

    return core, agg_per_pod, edge_per_pod, all_routers, p2p_links, host_links


# ---------------------------------------------------------------------------
# Inject ring_node.py into hosts and start ring all-reduce
# ---------------------------------------------------------------------------

def setup_and_start_ring(host_links):
    """
    - Build a stable ring order over all hosts (sorted by name).
    - Push ring_node.py into each host under /app/.
    - Start one ring process per host, with explicit Phase 0 (socket setup) inside.
    """

    # Collect unique hosts and their IPs
    host_info = {}
    for _, _, _, host, _, host_ip in host_links:
        ip = host_ip.split("/")[0]
        host_info[host.name] = (host, ip)

    #Mahmoud: here we define the order of hosts inside the logical ring
    # Stable ring order: lexicographic by host name
    #ordered_names = sorted(host_info.keys())
    ordered_names = [
    # --- Pod 0 ---
    "h_p0_e0_2",
    "h_p0_e0_3",  # intra-edge
    "h_p0_e1_2",  # intra-pod, cross-edge
    "h_p0_e1_3",  # intra-edge
    # --- Pod 1 ---
    "h_p1_e0_2",  # cross-pod
    "h_p1_e0_3",  # intra-edge
    "h_p1_e1_2",  # intra-pod, cross-edge
    "h_p1_e1_3",  # intra-edge
    # --- Pod 2 ---
    "h_p2_e0_2",  # cross-pod
    "h_p2_e0_3",  # intra-edge
    "h_p2_e1_2",  # intra-pod, cross-edge
    "h_p2_e1_3",  # intra-edge
    # --- Pod 3 ---
    "h_p3_e0_2",  # cross-pod
    "h_p3_e0_3",  # intra-edge
    "h_p3_e1_2",  # intra-pod, cross-edge
    "h_p3_e1_3",  # intra-edge (wraps to h_p0_e0_2, cross-pod)
    ]
    ring = [host_info[name] for name in ordered_names]
    world_size = len(ring)
    info(f"*** Ring hosts (world_size={world_size}): "
         f"{', '.join(ordered_names)}\n")

    # Gradient size: world_size chunks, each 65536 floats (256 KB). 524288 
    # Total per-host gradient ~ 4 MB for world_size=16.
    elems_per_chunk = 1600  # 8000 floats × 4 bytes × 16 hosts = 512,000 bytes (500 KB) per host  # 16 floats × 4 bytes × 16 hosts = 1024 bytes (1 KB) per host
    grad_elems = world_size * elems_per_chunk

    base_port = 5000  # base TCP port for ring connections

    # Push ring_node.py into each host
    info("*** Distributing ring_node.py to hosts\n") #Mahmoud : need to make sure this works correctly, as past experiments had issues with file empty
    for host, _ in ring:                             #Mahmoud: may need to change to printf like before
        host.cmd("mkdir -p /app")
        """
        host.cmd(
            "bash -c 'cat > /app/ring_node.py << \"EOF\"\n"
            + RING_NODE_SCRIPT +
            "\nEOF'"
        )
        """
        host.cmd(f"printf '%s\n' '{RING_NODE_SCRIPT}' > /app/ring_node.py")
        host.cmd("chmod +x /app/ring_node.py")

    # Start one ring process per host
    info("*** Starting ring all-reduce on all hosts\n")
    for rank, (host, ip) in enumerate(ring):
        _, right_ip = ring[(rank + 1) % world_size]

        listen_port = base_port + rank
        right_port = base_port + ((rank + 1) % world_size)
        log_file = f"/app/ring_rank{rank}.log"

        cmd = (
            "python3 /app/ring_node.py "
            f"--rank {rank} --world-size {world_size} "
            f"--listen-port {listen_port} "
            f"--right-ip {right_ip} --right-port {right_port} "
            f"--grad-elems {grad_elems} "
            f"--log-file {log_file} "
            "&"
        )
        host.cmd(cmd)

    info("*** Ring all-reduce processes launched (background in each host)\n")


#--------------------------------Measurments-----------------------------------

def collect_ring_metrics(host_links):
    """
    Parse per-rank ring logs on each host and compute:
      - total latency  = max rank latency (slowest rank)
      - total throughput = logical gradient bytes / total latency (MiB/s)
      - max tail      = max_latency - min_latency
    """

    info("*** Sleeping 20s for All-Reduce completion\n")
    sleep(20)


    # Rebuild the same host_info mapping used in setup_and_start_ring
    host_info = {}
    for _, _, _, host, _, host_ip in host_links:
        ip = host_ip.split("/")[0]
        host_info[host.name] = (host, ip)

    # Use the SAME logical ring order as in setup_and_start_ring (duplicated on purpose)
    ordered_names = [
        # --- Pod 0 ---
        "h_p0_e0_2",
        "h_p0_e0_3",  # intra-edge
        "h_p0_e1_2",  # intra-pod, cross-edge
        "h_p0_e1_3",  # intra-edge
        # --- Pod 1 ---
        "h_p1_e0_2",  # cross-pod
        "h_p1_e0_3",  # intra-edge
        "h_p1_e1_2",  # intra-pod, cross-edge
        "h_p1_e1_3",  # intra-edge
        # --- Pod 2 ---
        "h_p2_e0_2",  # cross-pod
        "h_p2_e0_3",  # intra-edge
        "h_p2_e1_2",  # intra-pod, cross-edge
        "h_p2_e1_3",  # intra-edge
        # --- Pod 3 ---
        "h_p3_e0_2",  # cross-pod
        "h_p3_e0_3",  # intra-edge
        "h_p3_e1_2",  # intra-pod, cross-edge
        "h_p3_e1_3",  # intra-edge (wraps to h_p0_e0_2, cross-pod)
    ]
    ring = [host_info[name] for name in ordered_names]
    world_size = len(ring)

    # Same gradient size logic as in setup_and_start_ring
    elems_per_chunk = 1600  # 8000 floats × 4 bytes × 16 hosts = 512,000 bytes (500 KB) per host
    grad_elems = world_size * elems_per_chunk
    gradient_bytes = grad_elems * 4.0  # float32

    latencies = []

    for rank, (host, _) in enumerate(ring):
        log_file = f"/app/ring_rank{rank}.log"
        # Grab the last TOTAL_TIME_SEC= line (if any)
        cmd = (
            f"grep 'TOTAL_TIME_SEC=' {log_file} | tail -n 1 2>/dev/null || true"
        )
        out = host.cmd(cmd)
        line = out.strip()
        if not line:
            info(f"*** WARNING: no TOTAL_TIME_SEC line for rank {rank} ({host.name})\n")
            continue

        try:
            # Expect something like: "... TOTAL_TIME_SEC=0.123456"
            #token = line.split("TOTAL_TIME_SEC=")[-1]
            #token= line.split("=", 1)[1].strip() 
            #latency = float(token)
            #latencies.append(latency)
            
            matches = re.findall(r"TOTAL_TIME_SEC=([0-9.]+)", line)
            if not matches:
                info(f"*** WARNING: could not find TOTAL_TIME_SEC in rank {rank} log\n")
                continue
            # Take the last one (the final completion time)
            latency_str = matches[-1]
            latency = float(latency_str)
            latencies.append(latency)
        except ValueError:
            info(f"*** WARNING: could not parse latency from rank {rank} line: {line}\n") #this here happens
            #info(f"*** DEBUG rank {rank} line={repr(line)} token={repr(token)}\n")

    if not latencies:
        info("*** Ring metrics: no latency data collected from logs\n")
        return

    fastest = min(latencies)
    slowest = max(latencies)
    max_tail = slowest - fastest
    total_latency = slowest

    # Logical gradient throughput (one all-reduced gradient per run)
    total_throughput_bytes_per_sec = gradient_bytes / total_latency
    total_throughput_mib_per_sec = total_throughput_bytes_per_sec / (1024.0 * 1024.0)
    gradient_mib = gradient_bytes / (1024.0 * 1024.0)

    info("*** Ring all-reduce metrics (from host logs):\n")
    info(f"    ranks with data:          {len(latencies)}/{world_size}\n")
    info(f"    total latency (slowest):  {total_latency:.6f} s\n")
    info(f"    max tail (slowest-fastest): {max_tail:.6f} s\n")
    info(f"    logical gradient size:    {gradient_mib:.3f} MiB\n")
    info(f"    total throughput:         {total_throughput_mib_per_sec:.3f} MiB/s\n")


    # Also append metrics to a local file on the controller
    try:
        with open("ring_metrics.log", "a") as f:
            f.write("Ring all-reduce metrics:\n")
            f.write(f"  ranks_with_data={len(latencies)}/{world_size}\n")
            f.write(f"  total_latency_s={total_latency:.6f}\n")
            f.write(f"  max_tail_s={max_tail:.6f}\n")
            f.write(f"  gradient_mib={gradient_mib:.3f}\n")
            f.write(f"  total_throughput_mib_s={total_throughput_mib_per_sec:.3f}\n")
            f.write("\n")
    except Exception as e:
        info(f"*** WARNING: failed to write ring_metrics.log: {e}\n")



#-------------------Per-link Measurments-----------------------
def read_if_counters(node, ifname):
    """
    Read tx_bytes and rx_bytes for a given interface inside a Mininet/Containernet node.
    Returns a tuple (tx_bytes, rx_bytes) as integers.
    """

    _ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # local definition
    # Paths inside the container namespace
    base = f"/sys/class/net/{ifname}/statistics"
    tx = node.cmd(f"cat {base}/tx_bytes 2>/dev/null || echo 0").strip()
    #rx = node.cmd(f"cat {base}/rx_bytes 2>/dev/null || echo 0").strip()
    out = node.cmd(
        f"cat {base}/tx_bytes {base}/rx_bytes 2>&1 || echo '0 0'"
    ).strip()
    clean = _ansi.sub("", out).replace("\r", "")
    nums = re.findall(r"\b\d+\b", clean)

    try:
        if len(nums) >= 2:
            tx = int(nums[-2])
            rx = int(nums[-1])
            return tx, rx
    except ValueError:
        #info(f"*** DEBUG rx={rx} \n")
        info(f"*** GOTCHAAAAAA\n")
        return 0, 0
    
    return 0, 0

    # Expect two tokens: tx rx
    #parts = out.split()
    #if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
    #    return int(parts[0]), int(parts[1])

    # Debug if unexpected output
    #info(f"*** DEBUG bad counter read node={node.name} if={ifname} out={out!r}\n")
    #if rx == "" :
     #   info(f"*** DEBUG EMPTY RX\n")
    #if( tx == "" ):
     #   info(f"*** DEBUG EMPTY TX\n")
    #info(f"*** DEBUG tx={tx} rx={rx} \n")
    #try:
    #    return int(tx), int(rx)
    #except ValueError:
    #    #info(f"*** DEBUG rx={rx} \n")
    #    info(f"*** GOTCHAAAAAA\n")
    #    return 0, 0

def snapshot_all_link_counters(p2p_links, host_links):
    """
    Take a snapshot of tx/rx counters for all router-router and router-host interfaces.
    Returns a dict keyed by (node.name, ifname).
    """
    snap = {}

    # Router <-> router links
    for n1, if1, _, n2, if2, _ in p2p_links:
        snap[(n1.name, if1)] = read_if_counters(n1, if1)
        snap[(n2.name, if2)] = read_if_counters(n2, if2)

    # Router <-> host links
    for n1, if1, _, n2, if2, _ in host_links:
        snap[(n1.name, if1)] = read_if_counters(n1, if1)
        snap[(n2.name, if2)] = read_if_counters(n2, if2)

    return snap


def report_raw_link_load(before_snap, p2p_links, host_links):
    """
    Compare current counters to 'before_snap' and print raw byte deltas per interface.
    """
    info("*** Raw link load (tx_bytes, rx_bytes per interface during ring) ***\n")

    seen = set()

    def report_for(node, ifname, role):
        key = (node.name, ifname)
        if key in seen:
            return
        seen.add(key)

        tx0, rx0 = before_snap.get(key, (0, 0)) #TODO: check about the default argument
        tx1, rx1 = read_if_counters(node, ifname)
        dtx = tx1 - tx0
        drx = rx1 - rx0
        #info(f"  {role} {node.name}:{ifname}  Δtx={dtx} B  Δrx={drx} B\n") #change this to write to a file
        try:
            with open("link_load.log", "a") as f:
                f.write("Raw link load for one ring run:\n") #TODO: can make this bit cleaner later though
                f.write(f"  {role} {node.name}:{ifname}  Δtx={dtx} B  Δrx={drx} B\n")
                #f.write("\n")
        except Exception as e:
            info(f"*** WARNING: failed to write link_load.log: {e}\n")

    # Router <-> router
    for n1, if1, _, n2, if2, _ in p2p_links:
        report_for(n1, if1, "R-R")
        report_for(n2, if2, "R-R")

    # Router <-> host
    for n1, if1, _, n2, if2, _ in host_links:
        report_for(n1, if1, "R-H")
        report_for(n2, if2, "R-H")

#-------------------Per-link Measurments-----------------------
#--------------------------------Measurments-----------------------------------







# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    setLogLevel("info")

    ROUTER_IMG = "frrouting/frr:latest"
    # IMPORTANT: build this image yourself from ubuntu:22.04 with python3 installed
    #HOST_IMG = "ring-host:22.04" 
    HOST_IMG = "network-multitool-python:latest" 


    net = Containernet(controller=None, link=TCLink, autoSetMacs=True)
    p2p_alloc = P2PAllocator()

    core, agg_per_pod, edge_per_pod, all_routers, p2p_links, host_links = \
        build_fattree_k4(net, p2p_alloc, ROUTER_IMG, HOST_IMG)

    net.start()

    info("*** Enforcing host /30 IPs\n")
    enforce_host_ips(host_links)

    info("*** Enforcing P2P /31 IPs\n")
    enforce_p2p_ips(p2p_links)

    info("*** Starting FRR on all routers\n")
    rid = 10
    for r in all_routers:
        start_frr_ospf(r, rid)
        rid += 1

    # Allow OSPF time to converge
    info("*** Sleeping 10s for OSPF convergence\n")
    sleep(10)


    # Take baseline link counters before starting the ring - Per-link measurements
    link_counters_before = snapshot_all_link_counters(p2p_links, host_links)

    # Start ring all-reduce across hosts (includes Phase 0 socket setup)
    setup_and_start_ring(host_links)

    # Drop into CLI while ring traffic is running in background
    info("*** Starting CLI (ring all-reduce is running in background)\n")
    CLI(net)


    # After exiting the CLI, collect ring metrics from host logs
    info("*** Collecting ring all-reduce metrics\n")
    collect_ring_metrics(host_links)

    # Now report raw link load based on byte deltas - Per-link measurements
    info("*** Collecting raw link load\n")
    report_raw_link_load(link_counters_before, p2p_links, host_links)

    net.stop()


if __name__ == "__main__":
    run()
