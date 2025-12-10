from mininet.net import Mininet
from mininet.log import setLogLevel, info
from mininet.node import OVSController
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

    '''
    # Warm-up ARP (important!)
    info("*** Warming ARP...\n")
    for h in hosts:
        for other in hosts:
            if h != other:
                h.cmd(f"ping -c1 -W1 {other.IP()} >/dev/null 2>&1")
    info("ARP warm-up finished.\n\n")
    '''


    # Perform N-1 scatter-reduce steps
    for step in range(N - 1):
        info(f"--- Step {step} ---\n")

        for i in range(N):
            src = hosts[i]
            dst = hosts[(i + 1) % N]
            dst_ip = dst.IP()

            # Send exactly one TCP packet
            cmd = f"iperf -c {dst_ip} -b 1M -l 1400 -t 0.5 &" #here it worked with -u just wrong measurments
            src.cmd(cmd)

            info(f"{src.name} -> {dst.name} (gradient packet)\n")

        sleep(0.3)  # allow propagation

    info("\n=== DONE ===\n")
    info("All hosts have seen every gradient packet (Phase 2 not needed).\n")


if __name__ == "__main__":
    setLogLevel("info")

    topo = FatTreeTopo(4)
    ##net = Mininet(topo=topo, controller=OVSController)
    net = Mininet(topo=topo, controller=None, autoSetMacs=True)


    net.start()
    info("*** Setting static ARP...\n")
    net.staticArp()


    onepacket_ring_allreduce(net)

    net.stop()

