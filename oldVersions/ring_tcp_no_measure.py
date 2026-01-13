from mininet.net import Mininet
from mininet.log import setLogLevel, info
from time import sleep

# Import your topology
from fatTree import FatTreeTopo


def onepacket_ring_allreduce(net):
    hosts = sorted(net.hosts, key=lambda h: h.name)
    N = len(hosts)

    info("*** Hosts in ring order:\n")
    for h in hosts:
        info(f"{h.name} ")
    info("\n\n")

    info("=== ONE-PACKET RING ALL-REDUCE ===\n")
    info("(Only Phase 1 is needed because each gradient is 1 packet.)\n\n")

    # ------------------------------------------
    # FIX: Start TCP iperf servers on all hosts
    # ------------------------------------------
    info("*** Starting TCP iperf servers on all hosts...\n")
    for h in hosts:
        h.cmd("pkill iperf")     # clean any old processes
        h.cmd("iperf -s &")      # start TCP server
    sleep(1)                     # allow servers to bind

    # ------------------------------------------
    # Perform N-1 scatter-reduce steps
    # ------------------------------------------
    for step in range(N - 1):
        info(f"--- Step {step} ---\n")

        for i in range(N):
            src = hosts[i]
            dst = hosts[(i + 1) % N]
            dst_ip = dst.IP()

            # Send exactly one TCP flow
            cmd = f"iperf -c {dst_ip} -b 1M -l 1400 -t 0.5 &"
            src.cmd(cmd)

            info(f"{src.name} -> {dst.name} (TCP gradient packet)\n")

        sleep(0.3)  # give TCP flows time to complete

    info("\n=== DONE ===\n")
    info("All hosts have sent their gradient packet to the next host.\n")
    info("Phase 2 is not needed because we only have 1 chunk.\n")


if __name__ == "__main__":
    setLogLevel("info")

    topo = FatTreeTopo(4)

    # No controller → switches run OVS learning
    net = Mininet(topo=topo, controller=None, autoSetMacs=True)

    net.start()

    info("*** Setting static ARP...\n")
    net.staticArp()

    onepacket_ring_allreduce(net)

    net.stop()
