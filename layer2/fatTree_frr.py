"""
fatTree_frr.py
A full FRR-based L3 Fat-Tree topology for Mininet.

Core idea:
- Every "switch" in the fat-tree is replaced with a real Linux router
  implemented as a Mininet Node.
- Each router runs FRR daemons (zebra + ospfd) inside its own namespace.
- OSPF automatically discovers all adjacencies and installs ECMP routes.
- Hosts use their TOR router (edge router) as gateway.
"""

from mininet.topo import Topo
from mininet.node import Node
import os


# ================================================================
#   Router Node Class (Runs FRR: zebra + ospfd)
# ================================================================
class FRRRouter(Node):
    """A Mininet router running FRR (OSPF + Zebra) in its namespace."""

    def config(self, **params):
        super(FRRRouter, self).config(**params)

        # Enable Linux forwarding
        self.cmd("sysctl -w net.ipv4.ip_forward=1")

        # Directory for FRR config files
        self.frr_dir = f"/tmp/frr-{self.name}"
        os.makedirs(self.frr_dir, exist_ok=True)

        #
        # --- Generate zebra.conf ---
        #
        zebra_cfg = f"""
hostname {self.name}
password zebra
enable password zebra
log file {self.frr_dir}/zebra.log
        """
        with open(f"{self.frr_dir}/zebra.conf", "w") as f:
            f.write(zebra_cfg)

        #
        # --- Generate ospfd.conf ---
        #
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

        #
        # --- Start FRR daemons (inside namespace) ---
        #
        self.cmd(f"zebra -d -f {self.frr_dir}/zebra.conf "
                 f"-z {self.frr_dir}/zebra.sock")

        self.cmd(f"ospfd -d -f {self.frr_dir}/ospfd.conf "
                 f"-z {self.frr_dir}/zebra.sock")

    def terminate(self):
        """Stop FRR daemons on node shutdown."""
        self.cmd("pkill ospfd")
        self.cmd("pkill zebra")
        super(FRRRouter, self).terminate()


# ================================================================
#   Fat-Tree Topology (k=4)
# ================================================================
class FatTreeFRR(Topo):
    """
    Full L3 Fat-Tree (k=4) where all switches are replaced by FRR routers.

    Addressing scheme:
      Host at pod P, edge E, host index H:
         IP = 10.P.E.H /24
         GW = 10.P.E.1

      Routers get interface IPs automatically:
         For every link created: assign incremental /31 addresses
    """

    def __init__(self, k=4):
        assert k == 4, "This implementation currently supports k=4 only."
        super(FatTreeFRR, self).__init__()

        self.k = k
        self.ip_counter = {}   # Track /31 assignments per router interface

        self.build_fattree()

    # ---------------------------
    # Assign incremental /31 addresses
    # ---------------------------
    def assign_router_iface_ip(self, rname, ifindex):
        """
        Returns an IP address for a router's interface:
            172.(router_id).(iface).0/31
        These are point-to-point routed links.
        """
        if rname not in self.ip_counter:
            self.ip_counter[rname] = 0

        base = self.ip_counter[rname]
        self.ip_counter[rname] += 1

        # Unique /31: 172.<routerID>.<base>.0/31
        return f"172.{hash(rname) % 250}.{base}.0/31"

    # ---------------------------
    # Build the fat-tree routers
    # ---------------------------
    def build_fattree(self):
        k = self.k

        #
        # --- CORE ROUTERS ---
        #
        core_routers = []
        num_core = (k // 2) ** 2
        for i in range(num_core):
            c = self.addNode(f"c{i}", cls=FRRRouter)
            core_routers.append(c)

        #
        # --- PODS ---
        #
        for p in range(k):  # pod number
            agg_routers = []
            edge_routers = []

            #
            # --- Aggregation routers ---
            #
            for a in range(k // 2, k):
                name = f"p{p}_a{a}"
                r = self.addNode(name, cls=FRRRouter)
                agg_routers.append(r)

            #
            # --- Edge routers ---
            #
            for e in range(k // 2):
                name = f"p{p}_e{e}"
                r = self.addNode(name, cls=FRRRouter)
                edge_routers.append(r)

                #
                # --- Hosts under this edge router ---
                #
                for h in range(2, (k // 2) + 2):
                    host_ip = f"10.{p}.{e}.{h}/24"
                    gw = f"10.{p}.{e}.1"

                    host = self.addHost(
                        f"h_p{p}_e{e}_{h}",
                        ip=host_ip,
                        defaultRoute=f"via {gw}"
                    )
                    self.addLink(r, host)

            #
            # --- Edge ↔ Aggregation links ---
            #
            for e_r in edge_routers:
                for a_r in agg_routers:
                    self.addLink(e_r, a_r)

            #
            # --- Aggregation ↔ Core links ---
            #
            half = k // 2
            for a_index, a_r in enumerate(agg_routers):
                group = a_index - half  # -2, -1 for k=4
                start = group * half
                end = (group + 1) * half
                for c_r in core_routers[start:end]:
                    self.addLink(a_r, c_r)
