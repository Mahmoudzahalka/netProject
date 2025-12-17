#!/usr/bin/env python3
"""
fattree_containernet_frr.py
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
        n = self.n
        self.n += 1
        A = n // 256
        B = n % 256
        ip1 = f"172.16.{A}.{2 * B}/31"
        ip2 = f"172.16.{A}.{2 * B + 1}/31"
        return ip1, ip2


# ---------------------------
# Detect router-router (P2P) interfaces by IP subnet
# ---------------------------
def get_p2p_intfs(router):
    p2p = []
    for intf in router.intfNames():
        out = router.cmd(f"ip -o -4 addr show dev {intf}").strip()
        if "inet 172.16." in out and "/31" in out:
            p2p.append(intf)
    return p2p


# ---------------------------
# Force correct /31 IPs on router-router links
# ---------------------------
def enforce_p2p_ips(p2p_links):
    for n1, i1, ip1, n2, i2, ip2 in p2p_links:
        n1.cmd(f"ip addr flush dev {i1}")
        n1.cmd(f"ip addr add {ip1} dev {i1}")
        n2.cmd(f"ip addr flush dev {i2}")
        n2.cmd(f"ip addr add {ip2} dev {i2}")


# ---------------------------
# Generate ospfd.conf
# ---------------------------
def generate_ospfd_conf(router_name, role, rid_suffix, p2p_intfs):
    base = f"""hostname {router_name}
password zebra
log file /var/log/frr/ospfd.log

router ospf
 ospf router-id 1.1.1.{rid_suffix}
 maximum-paths 8
 passive-interface default
"""

    for intf in p2p_intfs:
        base += f" no passive-interface {intf}\n"

    base += "\n"

    for intf in p2p_intfs:
        base += f"""interface {intf}
 ip ospf area 0
 ip ospf network point-to-point

"""
    return base


# ---------------------------
# Start FRR
# ---------------------------
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


# ---------------------------
# Build k=4 Fat-Tree
# ---------------------------
def build_fattree_k4(net, p2p_alloc, ROUTER_IMG, HOST_IMG):
    k = 4

    core = []
    agg_per_pod = {}
    edge_per_pod = {}
    all_routers = []
    p2p_links = []

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
                hip = f"10.{p}.{e}.{h}/24"
                gw = f"10.{p}.{e}.1"
                host = net.addDocker(
                    f"h_p{p}_e{e}_{h}",
                    dimage=HOST_IMG,
                    ip=hip,
                    defaultRoute=f"via {gw}",
                    cap_add=["NET_ADMIN"],
                )
                net.addLink(r, host, params1={"ip": f"{gw}/24"})

        agg_per_pod[p] = agg
        edge_per_pod[p] = edge

    for p in range(k):
        for er in edge_per_pod[p]:
            for ar in agg_per_pod[p]:
                ip1, ip2 = p2p_alloc.next31()
                link = net.addLink(er, ar, params1={"ip": ip1}, params2={"ip": ip2})
                p2p_links.append((er, link.intf1.name, ip1,
                                  ar, link.intf2.name, ip2))

    half = k // 2
    for p in range(k):
        for idx, ar in enumerate(agg_per_pod[p]):
            for c_r in core[idx * half:(idx + 1) * half]:
                ip1, ip2 = p2p_alloc.next31()
                link = net.addLink(ar, c_r, params1={"ip": ip1}, params2={"ip": ip2})
                p2p_links.append((ar, link.intf1.name, ip1,
                                  c_r, link.intf2.name, ip2))

    return core, agg_per_pod, edge_per_pod, all_routers, p2p_links


# ---------------------------
# Main
# ---------------------------
def run():
    setLogLevel("info")

    ROUTER_IMG = "frrouting/frr:latest"
    HOST_IMG = "praqma/network-multitool:latest"

    net = Containernet(controller=None, link=TCLink, autoSetMacs=True)
    p2p_alloc = P2PAllocator()

    core, agg_per_pod, edge_per_pod, all_routers, p2p_links = \
        build_fattree_k4(net, p2p_alloc, ROUTER_IMG, HOST_IMG)

    net.start()

    info("*** Enforcing P2P /31 IPs\n")
    enforce_p2p_ips(p2p_links)

    info("*** Starting FRR\n")
    rid = 10
    for r in all_routers:
        start_frr_ospf(r, rid)
        rid += 1

    sleep(10)
    CLI(net)
    net.stop()


if __name__ == "__main__":
    run()
