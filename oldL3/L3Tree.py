#!/usr/bin/env python3
"""
fattree_containernet_frr.py

Containernet k=4 (16 hosts) Fat-Tree with:
- NO L2 switches (no OVS)
- All "switches" are L3 routers implemented as Docker containers running FRR:
  edge (ToR), aggregation, core
- OSPF area 0 everywhere; Linux kernel ECMP used for equal-cost multipath

Run (inside venv, in repo root):
    source venv/bin/activate
    sudo -E env PATH=$PATH python3 fattree_containernet_frr.py
"""

import os
from time import sleep

from mininet.net import Containernet
from mininet.node import Docker
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel


# ---------------------------
# Unique /31 allocator for P2P links
# ---------------------------
class P2PAllocator:
    def __init__(self):
        self.n = 0

    def next31(self):
        """
        For link index n:
          A = n // 256
          B = n % 256
          subnet = 172.16.A.(2*B)/31

        Returns (ip1, ip2) as strings with /31 prefix.
        """
        n = self.n
        self.n += 1
        A = n // 256
        B = n % 256
        ip1 = f"172.16.{A}.{2 * B}/31"
        ip2 = f"172.16.{A}.{2 * B + 1}/31"
        return ip1, ip2


# ---------------------------
# Start FRR (zebra + ospfd) inside a router container
# ---------------------------
def start_frr_ospf(router, rid_suffix):
    """
    Start FRR (zebra + ospfd) inside an frrouting/frr container using watchfrr.

    - Uses /etc/frr for configs
    - Ensures /var/log/frr and /var/run/frr exist and are owned by frr:frr
    - Enables zebra and ospfd in /etc/frr/daemons
    - Starts /usr/lib/frr/watchfrr to supervise them
    """
    name = router.name
    frr_dir = "/etc/frr"

    # Make sure config + log + run dirs exist
    router.cmd("mkdir -p /etc/frr /var/log/frr /var/run/frr")

    # FRR daemons normally drop to user 'frr'
    # Ensure that user can write config/log/run dirs
    router.cmd("chown -R frr:frr /etc/frr /var/log/frr /var/run/frr || true")

    # Enable IP forwarding
    router.cmd("sysctl -w net.ipv4.ip_forward=1")

    # Build zebra.conf INSIDE the container
    zebra_conf = f"""hostname {name}
password zebra
log file /var/log/frr/zebra.log
"""
    router.cmd(
        f"bash -c 'cat > {frr_dir}/zebra.conf << \"EOF\"\\n{zebra_conf}EOF'"
    )

    #router.cmd(f"""bash -c 'cat > {frr_dir}/zebra.conf << EOF{zebra_conf}EOF'""")
    router.cmd(
    f"printf '%s\n' '{zebra_conf}' > /etc/frr/zebra.conf"
    )

    # Build ospfd.conf INSIDE the container
    ospf_conf = f"""hostname {name}
password zebra
log file /var/log/frr/ospfd.log

router ospf
 ospf router-id 1.1.1.{rid_suffix}
 maximum-paths 8
 network 0.0.0.0/0 area 0
"""
    router.cmd(
        f"bash -c 'cat > {frr_dir}/ospfd.conf << \"EOF\"\\n{ospf_conf}EOF'"
    )

    router.cmd(
    f"printf '%s\n' '{ospf_conf}' > /etc/frr/ospfd.conf"
    )

    router.cmd("touch /etc/frr/vtysh.conf") #Mahmoud both of these added now
    router.cmd("chown -R frr:frr /etc/frr /var/log/frr /var/run/frr || true")


    # Enable daemons in /etc/frr/daemons (this file is part of the image)
    router.cmd("sed -i 's/^zebra=no/zebra=yes/' /etc/frr/daemons || true")
    router.cmd("sed -i 's/^ospfd=no/ospfd=yes/' /etc/frr/daemons || true")

    # Start watchfrr to supervise zebra + ospfd
    # In the official image, watchfrr lives in /usr/lib/frr
    router.cmd("/usr/lib/frr/watchfrr -d zebra ospfd")



# ---------------------------
# Build the k=4 Fat-Tree in Containernet
# ---------------------------
def build_fattree_k4(net, p2p_alloc, ROUTER_IMG, HOST_IMG):
    """
    Build k=4 fat-tree:

    - 4 cores
    - 4 pods
      - each pod: 2 agg, 2 edge, 4 hosts (2 per edge)

    All routers and hosts are Docker containers.
    IPs:
      host: 10.P.E.H/24, gw 10.P.E.1
      edge-host: router side = 10.P.E.1/24
      p2p router-router: 172.16.A.X/31 via P2PAllocator
    """
    k = 4

    core = []
    agg_per_pod = {}
    edge_per_pod = {}
    all_routers = []

    # ---------------- CORE ROUTERS ----------------
    info("*** Creating core routers (4)\n")
    for i in range((k // 2) ** 2):  # 4
        r = net.addDocker(
            f"c{i}",
            dimage=ROUTER_IMG,
            privileged=True,
            cap_add=["NET_ADMIN", "SYS_ADMIN"],
        )
        core.append(r)
        all_routers.append(r)

    info("*** Creating pods with aggregation + edge routers and hosts\n")

    # ---------------- PODS ----------------
    for p in range(k):  # pods 0..3
        agg = []
        edge = []

        # Aggregation routers in pod
        info(f"*** Pod {p}: aggregation routers\n")
        for a in range(k // 2, k):  # 2,3
            rname = f"p{p}_a{a}"
            r = net.addDocker(
                rname,
                dimage=ROUTER_IMG,
                privileged=True,
                cap_add=["NET_ADMIN", "SYS_ADMIN"],
            )
            agg.append(r)
            all_routers.append(r)

        # Edge routers + hosts in pod
        info(f"*** Pod {p}: edge routers + hosts\n")
        for e in range(k // 2):  # 0,1
            rname = f"p{p}_e{e}"
            r = net.addDocker(
                rname,
                dimage=ROUTER_IMG,
                privileged=True,
                cap_add=["NET_ADMIN", "SYS_ADMIN"],
            )
            edge.append(r)
            all_routers.append(r)

            # Two hosts per edge (H=2,3)
            for h in range(2, (k // 2) + 2):
                hip = f"10.{p}.{e}.{h}/24"
                gw = f"10.{p}.{e}.1"
                host = net.addDocker(
                    f"h_p{p}_e{e}_{h}",
                    dimage=HOST_IMG,
                    ip=hip,
                    defaultRoute=f"via {gw}",
                    cap_add=["NET_ADMIN"],
                )
                # Link edge router ↔ host
                net.addLink(
                    r,
                    host,
                    params1={"ip": f"{gw}/24"},
                    params2={},
                )

        agg_per_pod[p] = agg
        edge_per_pod[p] = edge

    # ---------------- EDGE ↔ AGG LINKS ----------------
    info("*** Wiring edge↔aggregation links (per pod)\n")
    for p in range(k):
        agg = agg_per_pod[p]
        edge = edge_per_pod[p]
        for er in edge:
            for ar in agg:
                ip1, ip2 = p2p_alloc.next31()
                net.addLink(
                    er,
                    ar,
                    params1={"ip": ip1},
                    params2={"ip": ip2},
                )

    # ---------------- AGG ↔ CORE LINKS ----------------
    info("*** Wiring aggregation↔core links\n")
    half = k // 2  # 2
    for p in range(k):
        agg = agg_per_pod[p]
        for idx, ar in enumerate(agg):
            group = idx  # 0 or 1
            start = group * half
            end = (group + 1) * half
            for c_r in core[start:end]:
                ip1, ip2 = p2p_alloc.next31()
                net.addLink(
                    ar,
                    c_r,
                    params1={"ip": ip1},
                    params2={"ip": ip2},
                )

    return core, agg_per_pod, edge_per_pod, all_routers


# ---------------------------
# Main runner
# ---------------------------
def run():
    setLogLevel("info")

    # Adjust images to what you actually have:
    ROUTER_IMG = "frrouting/frr:latest"
    HOST_IMG = "praqma/network-multitool:latest"

    info("*** Creating Containernet instance (no controller)\n")
    net = Containernet(
        controller=None,
        link=TCLink,
        autoSetMacs=True,
    )

    p2p_alloc = P2PAllocator()

    info("*** Building k=4 FRR-based L3 Fat-Tree\n")
    core, agg_per_pod, edge_per_pod, all_routers = build_fattree_k4(
        net, p2p_alloc, ROUTER_IMG, HOST_IMG
    )

    info("*** Starting network\n")
    net.start()

    # Start FRR on all routers (core + agg + edge)
    info("*** Starting FRR (zebra+ospfd) on all routers\n")
    rid = 10
    for r in all_routers:
        start_frr_ospf(r, rid)
        rid += 1

    info("*** Waiting a bit for OSPF to converge\n")
    sleep(10)

    info("\n*** Quick sanity checks:\n")
    info("  containernet> nodes\n")
    info("  containernet> h_p0_e0_2 ping -c 3 10.0.0.1        # ToR GW\n")
    info("  containernet> h_p0_e0_2 ping -c 3 h_p3_e1_3       # Cross-pod\n")
    info("  containernet> p0_e0 bash\n")
    info("    # vtysh -c 'show ip ospf neighbor'\n")
    info("    # vtysh -c 'show ip route'\n\n")

    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    run()


