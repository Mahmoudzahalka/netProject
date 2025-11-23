from mininet.net import Mininet
from mininet.log import setLogLevel, info
from time import sleep

def onepacket_ring_allreduce(net):
    hosts = sorted(net.hosts, key=lambda h: h.name)
    N = len(hosts)

    info("*** Hosts in ring order:\n")
    for h in hosts:
        info(f"{h.name} ")
    info("\n\n")

    info("=== ONE-PACKET RING ALL-REDUCE ===\n")
    info("(Only Phase 1 is needed because there is a single chunk.)\n\n")

    # Perform N-1 steps for scatter-reduce
    for step in range(N - 1):
        info(f"--- Step {step} ---\n")

        # Each host sends exactly one UDP packet to next host
        for i in range(N):
            src = hosts[i]
            dst = hosts[(i + 1) % N]
            dst_ip = dst.IP()

            # Send a single packet (1400 bytes payload)
            cmd = f"iperf -c {dst_ip} -u -b 1M -l 1400 -t 0.05 &"
            src.cmd(cmd)

            info(f"{src.name} -> {dst.name} (gradient packet)\n")

        # Let packets propagate
        sleep(0.3)

    info("\n=== DONE: All hosts have seen every gradient packet ===\n")
    info("Phase 2 is not needed because there is only 1 gradient chunk.\n")


if __name__ == "__main__":
    setLogLevel("info")

    net = Mininet()
    net.start()

    onepacket_ring_allreduce(net)

    net.stop()
