from mininet.net import Mininet
from mininet.log import setLogLevel, info
from mininet.node import OVSController
from time import sleep, time
import re

from fatTree import FatTreeTopo


def get_port_stats(sw):
    """Return a dictionary {port: (rx_bytes, tx_bytes, rx_pkts, tx_pkts)}"""
    out = sw.cmd("ovs-ofctl dump-ports {} 2>/dev/null".format(sw.name))
    stats = {}
    for line in out.split("\n"):
        m = re.search(r"port\s+(\d+):.*rx pkts=(\d+), bytes=(\d+).*tx pkts=(\d+), bytes=(\d+)", line)
        if m:
            port, rxp, rxb, txp, txb = m.groups()
            stats[int(port)] = (int(rxb), int(txb), int(rxp), int(txp))
    return stats


def onepacket_ring_allreduce(net):

    hosts = sorted(net.hosts, key=lambda h: h.name)
    N = len(hosts)

    # ------------------------------
    # ENABLE STATIC ARP
    # ------------------------------
    info("*** Setting static ARP...\n")
    net.staticArp()

    # ------------------------------
    # SAVE INITIAL SWITCH PORT STATS
    # ------------------------------
    info("*** Capturing initial switch statistics...\n")
    sw_stats_before = {}
    for sw in net.switches:
        sw_stats_before[sw.name] = get_port_stats(sw)

    # ------------------------------
    # START TCPDUMP ON RECEIVERS
    # ------------------------------
    info("*** Starting packet capture on all hosts...\n")
    for h in hosts:
        h.cmd("pkill tcpdump")
        h.cmd("rm -f /tmp/%s.log" % h.name)
        h.cmd("tcpdump -i %s-eth0 -nn -tt -vv port 5001 > /tmp/%s.log 2>&1 &" %
              (h.name, h.name))

    sleep(1)

    info("*** Starting ring-allreduce...\n")
    start_time = time()

    # iperf TCP uses port 5001 by default
    for step in range(N - 1):
        info(f"--- Step {step} ---\n")

        for i in range(N):
            src = hosts[i]
            dst = hosts[(i + 1) % N]
            dst_ip = dst.IP()

            # Use iperf with proper TCP settings
            cmd = f"iperf -c {dst_ip} -b 1M -l 1400 -t 0.5 -x C 2>&1"
            result = src.cmd(cmd)
            info(f"{src.name} -> {dst.name}: {result[:100]}\n")  # Show first 100 chars of output

        sleep(0.4)

    end_time = time()
    total_time = end_time - start_time

    info(f"\n=== TOTAL COMMUNICATION TIME: {total_time:.4f} seconds ===\n")

    # ------------------------------
    # STOP TCPDUMP & MEASURE LATENCY
    # ------------------------------
    info("*** Stopping tcpdump...\n")
    for h in hosts:
        h.cmd("pkill tcpdump")
        sleep(0.1)

    info("*** Computing latency...\n")
    latencies = []

    for h in hosts:
        try:
            with open(f"/tmp/{h.name}.log", "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    info(f"{h.name}: captured {len(lines)} lines\n")
                    # Extract only tcpdump timestamp lines (format: HH:MM:SS.ffffff)
                    for line in lines:
                        # tcpdump format: "HH:MM:SS.ffffff IP ..."
                        m = re.search(r"^(\d{2}):(\d{2}):(\d{2})\.(\d+)", line)
                        if m:
                            hours, mins, secs, usecs = m.groups()
                            t_recv = int(secs) + int(usecs) / 1e6
                            latencies.append(t_recv)
        except FileNotFoundError:
            info(f"Warning: {h.name} tcpdump log not found\n")

    if len(latencies) > 1:
        latencies.sort()
        diffs = [latencies[i+1] - latencies[i] for i in range(len(latencies)-1)]
        avg_latency = sum(diffs) / len(diffs) if diffs else 0
    else:
        avg_latency = 0

    info(f"=== AVG ONE-WAY LATENCY (approx): {avg_latency*1000:.3f} ms ===\n")
    info(f"Total packets captured: {len(latencies)}\n")

    # ------------------------------
    # SWITCH STATISTICS AFTER
    # ------------------------------
    info("*** Gathering final switch stats...\n")
    sw_stats_after = {}
    for sw in net.switches:
        sw_stats_after[sw.name] = get_port_stats(sw)

    # ------------------------------
    # COMPUTE LINK LOAD & BANDWIDTH
    # ------------------------------
    info("\n=== LINK LOAD / BANDWIDTH UTILIZATION ===\n")
    for sw in net.switches:
        info(f"\nSwitch {sw.name}:\n")
        before = sw_stats_before[sw.name]
        after = sw_stats_after[sw.name]

        for port in after:
            if port in before:
                rxb1, txb1, rxp1, txp1 = before[port]
                rxb2, txb2, rxp2, txp2 = after[port]
                drxb = rxb2 - rxb1
                dtxb = txb2 - txb1
                drxp = rxp2 - rxp1
                dtxp = txp2 - txp1

                bw_rx = drxb * 8 / total_time / 1e6  # Mbps
                bw_tx = dtxb * 8 / total_time / 1e6

                info(f"  Port {port}: RX {drxp} pkts, {drxb} bytes "
                     f"(BW {bw_rx:.2f} Mbps) | "
                     f"TX {dtxp} pkts, {dtxb} bytes "
                     f"(BW {bw_tx:.2f} Mbps)\n")


if __name__ == "__main__":
    setLogLevel("info")

    topo = FatTreeTopo(4)
    net = Mininet(topo=topo, controller=None, autoSetMacs=True)

    net.start()

    onepacket_ring_allreduce(net)

    net.stop()
