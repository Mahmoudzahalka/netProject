#!/usr/bin/env python3
"""
FRR-based L3 Fat-Tree topology for Mininet (k=4), fully routed.

Design:
- All "switches" are FRR routers (FRRRouter).
- Every router-router and host-router link is a /31 point-to-point link.
- No L2 switching anywhere.
- OSPF (ospfd) + zebra run in each router namespace.
"""

import os

from mininet.topo import Topo
from mininet.node import Node
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel, info

# Absolute paths to FRR binaries (Ubuntu default – adjust if different on your system)
FRR_ZEBRA = "/usr/lib/frr/zebra"
FRR_OSPFD = "/usr/lib/frr/ospfd"


# ================================================================
#   Router Node Class (FRR: zebra + ospfd)
# ================================================================
class FRRRouter(Node):
    """Mininet router running FRR (zebra + ospfd) inside its namespace."""

    def config(self, **params):
        super(FRRRouter, self).config(**params)

        # Enable IPv4 forwarding
        self.cmd("sysctl -w net.ipv4.ip_forward=1")

        # Per-router FRR directory
        self.frr_dir = f"/tmp/frr-{self.name}"
        os.makedirs(self.frr_dir, exist_ok=True)
        self.cmd(f"chown -R frr:frr {self.frr_dir}")

        zebra_conf = f"{self.frr_dir}/zebra.conf"
        ospf_conf = f"{self.frr_dir}/ospfd.conf"

        # ---------------- zebra.conf ----------------
        zebra_cfg = f"""
hostname {self.name}
password zebra
enable password zebra
log file {self.frr_dir}/zebra.log
"""
        with open(zebra_conf, "w") as f:
            f.write(zebra_cfg)

        # ---------------- ospfd.conf ----------------
        ospf_cfg = f"""
hostname {self.name}
password zebra
log file {self.frr_dir}/ospfd.log

interface default
 ip ospf network point-to-point

router ospf
 ospf router-id 1.1.1.{self.pid}
 network 0.0.0.0/0 area 0
 redistribute connected
"""
        with open(ospf_conf, "w") as f:
            f.write(ospf_cfg)
    def startFRR(self):
        """Start zebra + ospfd AFTER Mininet has brought interfaces up."""

        zebra_conf = f"{self.frr_dir}/zebra.conf"
        ospf_conf  = f"{self.frr_dir}/ospfd.conf"

        zebra_pid = f"{self.frr_dir}/zebra.pid"
        ospf_pid  = f"{self.frr_dir}/ospfd.pid"

        zebra_out = f"{self.frr_dir}/zebra.out"
        zebra_err = f"{self.frr_dir}/zebra.err"
        ospf_out  = f"{self.frr_dir}/ospfd.out"
        ospf_err  = f"{self.frr_dir}/ospfd.err"

        info(f"*** Starting FRR on {self.name}\n")

        # ---- Start zebra as user frr ----
        self.cmd(
            f"{FRR_ZEBRA} -d "
            f"-f {zebra_conf} "
            f"-z {self.frr_dir}/zebra.sock "
            f"-i {zebra_pid} "
            f"--user frr --group frr "
            f"> {zebra_out} 2> {zebra_err} &"
        )

        # ---- Start ospfd as user frr ----
        self.cmd(
            f"{FRR_OSPFD} -d "
            f"-f {ospf_conf} "
            f"-z {self.frr_dir}/zebra.sock "
            f"-i {ospf_pid} "
            f"--user frr --group frr "
            f"> {ospf_out} 2> {ospf_err} &"
        )



    def terminate(self):
        """Stop FRR daemons on node shutdown."""
        self.cmd("pkill ospfd || true")
        self.cmd("pkill zebra || true")
        super(FRRRouter, self).terminate()


# ================================================================
#   Fat-Tree Topology (k=4) with FRR Routers and /31 everywhere
# ================================================================
class FatTreeFRR(Topo):
    """
    Full L3 Fat-Tree (k=4) with FRR routers.

    Addressing:

    1) Host-router links (point-to-point, /31):
         router: 10.P.E.2/31 , host: 10.P.E.3/31
         router: 10.P.E.4/31 , host: 10.P.E.5/31

       (Avoids 10.x.x.0/31)

    2) Router-router links (edge<->agg, agg<->core):
       Each gets a unique /31 in 172.16.0.0/16.
    """

    def __init__(self, k=4):
        assert k == 4, "This implementation supports only k=4."
        super(FatTreeFRR, self).__init__()

        self.k = k
        self.p2p_count = 0  # for router-router P2P links

        self.build_fattree()

    # ---------------------------
    # Helper: /31 router-router link
    # ---------------------------
    def addP2PLink(self, r1, r2):
        n = self.p2p_count
        self.p2p_count += 1

        A = n // 256
        B = n % 256
        ip1 = f"172.16.{A}.{2 * B}/31"
        ip2 = f"172.16.{A}.{2 * B + 1}/31"

        self.addLink(
            r1,
            r2,
            params1={"ip": ip1},
            params2={"ip": ip2},
        )

    # ---------------------------
    # Build the Fat-Tree
    # ---------------------------
    def build_fattree(self):
        k = self.k

        # ---------- CORE ROUTERS ----------
        core_routers = []
        num_core = (k // 2) ** 2  # for k=4: 4 core routers
        for i in range(num_core):
            c = self.addNode(f"c{i}", cls=FRRRouter, ip=None)
            core_routers.append(c)


        # ---------- PODS ----------
        for p in range(k):  # pod index: 0..3
            agg_routers = []
            edge_routers = []

            # ----- Aggregation routers -----
            for a in range(k // 2, k):  # a=2,3 for k=4
                name = f"p{p}_a{a}"
                r = self.addNode(name, cls=FRRRouter, ip=None)
                agg_routers.append(r)


            # ----- Edge routers + hosts -----
            for e in range(k // 2):  # e=0,1 for k=4
                er_name = f"p{p}_e{e}"
                er = self.addNode(er_name, cls=FRRRouter, ip=None)
                edge_routers.append(er)


                # 2 hosts per edge router → always two /31s
                # host link #1 → 10.p.e.[2]/31 ↔ 10.p.e.[3]/31
                # host link #2 → 10.p.e.[4]/31 ↔ 10.p.e.[5]/31
                for h_idx in range(2):  # 0,1
                    base = 2 * (h_idx + 1)      # 2, 4
                    router_ip = f"10.{p}.{e}.{base}/31"
                    host_ip   = f"10.{p}.{e}.{base + 1}/31"

                    router_ip_plain = router_ip.split("/")[0]

                    host = self.addHost(
                        f"h_p{p}_e{e}_{h_idx+2}",
                        ip=host_ip,
                        defaultRoute=f"via {router_ip_plain}",
                    )

                    # Router interface towards this host
                    self.addLink(
                        er,
                        host,
                        params1={"ip": router_ip},
                        params2={},
                    )

            # ----- Edge ↔ Aggregation links -----
            for er in edge_routers:
                for ar in agg_routers:
                    self.addP2PLink(er, ar)

            # ----- Aggregation ↔ Core links -----
            half = k // 2
            for a_index, ar in enumerate(agg_routers):
                group = a_index  # 0 or 1 for k=4
                start = group * half
                end = (group + 1) * half
                for c_r in core_routers[start:end]:
                    self.addP2PLink(ar, c_r)


# ================================================================
#   Runner
# ================================================================
def run():
    topo = FatTreeFRR(k=4)

    # No OpenFlow controller: routing done purely by FRR in routers
    net = Mininet(
        topo=topo,
        controller=None,
        autoSetMacs=True,
        autoStaticArp=False
    )

    net.start()
    info("\n*** Mininet topology started. Starting FRR on all routers... ***\n")

    # Start FRR on ALL FRRRouter nodes
    for node in net.nameToNode.values():
        if isinstance(node, FRRRouter):
            node.startFRR()

    info("\n*** Give OSPF 3–5 seconds to converge, then use vtysh. ***\n")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
