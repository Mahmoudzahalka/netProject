#!/usr/bin/env python3
"""
fatTree_frr.py
A full FRR-based L3 Fat-Tree topology for Mininet (k=4).

Core idea:
- Every "switch" in the fat-tree is replaced with a real Linux router
  implemented as a Mininet Node (FRRRouter).
- Each router runs FRR daemons (zebra + ospfd) inside its own namespace.
- OSPF automatically discovers all adjacencies and installs ECMP routes.
- Hosts use their TOR router (edge router) as gateway.

Requirements on the host:
- Mininet installed
- FRR installed (binaries `zebra` and `ospfd` in PATH)
"""

import os

from mininet.topo import Topo
from mininet.node import Node
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel, info


# ================================================================
#   Router Node Class (Runs FRR: zebra + ospfd)
# ================================================================
class FRRRouter(Node):
    """A Mininet router running FRR (OSPF + Zebra) in its namespace."""

    def config(self, **params):
        super(FRRRouter, self).config(**params)

        # Enable Linux IP forwarding
        self.cmd("sysctl -w net.ipv4.ip_forward=1")

        # Directory for this router's FRR config & logs
        self.frr_dir = f"/tmp/frr-{self.name}"
        os.makedirs(self.frr_dir, exist_ok=True)

        # --- Generate zebra.conf ---
        zebra_cfg = f"""
hostname {self.name}
password zebra
enable password zebra
log file {self.frr_dir}/zebra.log
"""
        with open(f"{self.frr_dir}/zebra.conf", "w") as f:
            f.write(zebra_cfg)

        # --- Generate ospfd.conf ---
        # router-id is arbitrary but must be unique per router;
        # here we just derive it from the router's PID.
        ospf_cfg = f"""
hostname {self.name}
password zebra
log file {self.frr_dir}/ospfd.log

router ospf
 ospf router-id 1.1.1.{self.pid}
 network 0.0.0.0/0 area 0
"""
        with open(f"{self.frr_dir}/ospfd.conf", "w") as f:
            f.write(ospf_cfg)

        # --- Start FRR daemons (inside namespace) ---
        # NOTE: assumes `zebra` and `ospfd` are in PATH.
        self.cmd(
            f"zebra -d -f {self.frr_dir}/zebra.conf "
            f"-z {self.frr_dir}/zebra.sock"
        )
        self.cmd(
            f"ospfd -d -f {self.frr_dir}/ospfd.conf "
            f"-z {self.frr_dir}/zebra.sock"
        )

    def terminate(self):
        """Stop FRR daemons on node shutdown."""
        # Scoped to this namespace, so should not kill host FRR
        self.cmd("pkill ospfd || true")
        self.cmd("pkill zebra || true")
        super(FRRRouter, self).terminate()


# ================================================================
#   Fat-Tree Topology (k=4) with FRR Routers
# ================================================================
class FatTreeFRR(Topo):
    """
    Full L3 Fat-Tree (k=4) where all "switches" are FRR routers.

    Addressing scheme:

      1) Host subnets:
         Host at pod P, edge E, host index H:
             IP = 10.P.E.H/24
             GW = 10.P.E.1

         Edge router interface towards hosts:
             IP = 10.P.E.1/24 on that interface.

      2) Router-router links (edge<->agg, agg<->core):
         Each router-router link gets a unique /31 subnet:

             172.16.A.B/31

         One side gets the ".0" address, the other the ".1" address.
    """

    def __init__(self, k=4):
        assert k == 4, "This implementation currently supports k=4 only."
        super(FatTreeFRR, self).__init__()

        self.k = k

        # Counter for allocating unique /31 subnets for router-router links
        self.p2p_count = 0

        self.build_fattree()

    # ---------------------------
    # Helper: create a P2P link with /31 IPs
    # ---------------------------
    def addP2PLink(self, r1, r2):
        """
        Create a point-to-point link between routers r1 and r2, assigning
        each side a unique /31 IP from 172.16.0.0/16 space.

        For link index n:
          A = n // 256
          B = n % 256
          subnet = 172.16.A.(2*B)/31

          r1 gets 172.16.A.(2*B)/31
          r2 gets 172.16.A.(2*B+1)/31
        """
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
    # Build the fat-tree routers
    # ---------------------------
    def build_fattree(self):
        k = self.k

        # --- CORE ROUTERS ---
        core_routers = []
        num_core = (k // 2) ** 2  # for k=4: 4 core routers
        for i in range(num_core):
            c = self.addNode(f"c{i}", cls=FRRRouter)
            core_routers.append(c)

        # --- PODS ---
        for p in range(k):  # pod number: 0..3
            agg_routers = []
            edge_routers = []

            # --- Aggregation routers (upper layer in pod) ---
            # For k=4: indices 2,3
            for a in range(k // 2, k):
                name = f"p{p}_a{a}"
                r = self.addNode(name, cls=FRRRouter)
                agg_routers.append(r)

            # --- Edge routers (ToR) + Hosts ---
            # For k=4: edge indices 0,1; 2 hosts per edge
            for e in range(k // 2):
                er_name = f"p{p}_e{e}"
                er = self.addNode(er_name, cls=FRRRouter)
                edge_routers.append(er)

                # Hosts under this edge router
                # H = 2,3 -> host IPs 10.P.E.2 and 10.P.E.3
                for h in range(2, (k // 2) + 2):
                    host_ip = f"10.{p}.{e}.{h}/24"
                    gw_ip = f"10.{p}.{e}.1"

                    host = self.addHost(
                        f"h_p{p}_e{e}_{h}",
                        ip=host_ip,
                        defaultRoute=f"via {gw_ip}",
                    )

                    # Router interface towards host gets GW address /24
                    self.addLink(
                        er,
                        host,
                        params1={"ip": f"{gw_ip}/24"},
                        params2={},
                    )

            # --- Edge ↔ Aggregation links ---
            # Every edge router connects to every agg router in its pod.
            for er in edge_routers:
                for ar in agg_routers:
                    self.addP2PLink(er, ar)

            # --- Aggregation ↔ Core links ---
            # Standard fat-tree wiring pattern:
            # each aggregation router connects to k/2 distinct core routers.
            half = k // 2
            for a_index, ar in enumerate(agg_routers):
                # For k=4, agg_routers has 2 routers (index 0,1)
                # We split the core routers into 2 groups of size half (=2):
                #
                #   agg[0] -> core[0:2]
                #   agg[1] -> core[2:4]
                #
                group = a_index  # 0 or 1
                start = group * half
                end = (group + 1) * half
                for c_r in core_routers[start:end]:
                    self.addP2PLink(ar, c_r)


# ================================================================
#   Runner
# ================================================================
def run():
    "Create and run the FRR-based FatTree topology."
    topo = FatTreeFRR(k=4)

    # No OpenFlow controller: all routing is done by FRR in the routers.
    net = Mininet(topo=topo, controller=None, autoSetMacs=True, autoStaticArp=False)

    net.start()
    info("\n*** Network started. Waiting briefly for OSPF to converge...\n\n")

    # Optional: you can sleep a bit here if you want:
    # import time; time.sleep(5)

    # Drop into Mininet CLI for manual testing:
    # - use `h_p0_e0_2 ping h_p1_e1_3`
    # - or `h_p0_e0_2 traceroute h_p2_e1_3`
    # - or from a router: `r = net['p0_e0']; r.cmd('vtysh -c \"show ip ospf neighbor\"')`
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
