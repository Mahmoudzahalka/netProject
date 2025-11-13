#!/usr/bin/env python3
"""
POX controller that builds a deterministic logical ring over all FatTree hosts
WITHOUT sniffing packets or learning.

Works directly from FatTree parameters (k).
"""

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import EthAddr


log = core.getLogger()

# --------- Must match your FatTree topo file ---------
def location_to_dpid(core=None, pod=None, switch=None):
    if core is not None:
        return int('0000000010%02x0000' % core, 16)
    else:
        return int('000000002000%02x%02x' % (pod, switch), 16)

def host_to_ip(p, s, h):
    return f"10.{p}.{s}.{h}"

def ip_to_mac(ip):
    parts = list(map(int, ip.split(".")))
    pod, sw, host = parts[1], parts[2], parts[3]
    return "00:00:00:%02x:%02x:%02x" % (pod, sw, host)

# ------------------------------------------------------

class HostRingStatic(object):
    def __init__(self, k):
        self.k = k

        core.openflow.addListeners(self)

        # Precompute ring
        self.hosts = self.compute_all_hosts(k)
        log.info(f"Computed {len(self.hosts)} hosts from FatTree(k={k})")

        self.ring = list(self.hosts.keys())  # MAC order = natural lexicographic
        log.info("Logical host ring created:")
        for i, mac in enumerate(self.ring):
            nxt = self.ring[(i+1) % len(self.ring)]
            log.info(f"  {mac}  ->  {nxt}")

    # ------------------------------------------------------------
    # Build all hosts deterministically (NO learning)
    # ------------------------------------------------------------
    def compute_all_hosts(self, k):
        """
        Your FatTree defines:
           pods          = k
           lower switches = k/2 per pod
           hosts/lower_sw = k/2 (host IDs start at 2)
        """

        hosts = {}

        for pod in range(k):
            for sw in range(k // 2):              # only lower switches have hosts
                for hostID in range(2, (k//2)+2): # hostID = 2,3,...,(k/2+1)
                    ip  = host_to_ip(pod, sw, hostID)
                    mac = ip_to_mac(ip)
                    dpid = location_to_dpid(pod=pod, switch=sw)
                    port = hostID                # same as in your topo addLink()

                    key = mac
                    hosts[key] = {
                        "pod": pod,
                        "sw": sw,
                        "hostID": hostID,
                        "mac": mac,
                        "ip": ip,
                        "dpid": dpid,
                        "port": hostID
                    }
        return hosts

    # ------------------------------------------------------------
    # Install static ring forwarding rules
    # ------------------------------------------------------------
    def _handle_ConnectionUp(self, event):
        dpid = event.dpid

        # For each host attached to this switch, install rule: H_i -> H_(i+1)
        for i, mac_src in enumerate(self.ring):
            h_src = self.hosts[mac_src]

            if h_src["dpid"] != dpid:
                continue

            mac_dst = self.ring[(i+1) % len(self.ring)]   # next in ring
            h_dst = self.hosts[mac_dst]

            # Send traffic from H_src to the port of H_dst
            msg = of.ofp_flow_mod()
            #msg.match.dl_src = mac_src   #replaced to avoid POX bug
            msg.match.dl_src = EthAddr(mac_src)
            msg.actions.append(of.ofp_action_output(port=h_dst["port"]))
            event.connection.send(msg)

            log.info(f"Switch {dpid}: {mac_src} -> {mac_dst} via port {h_dst['port']}")

def launch(k=4):
    k = int(k)
    core.registerNew(HostRingStatic, k)

