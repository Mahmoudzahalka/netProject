#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from time import time, sleep

NUM_HOSTS = 16

def main():
    setLogLevel("info")

    info("\n*** Creating FatTree Mininet network\n")

    # Import your topology
    from fatTree import FatTreeTopo
    topo = FatTreeTopo(k=4)

    # Create Mininet with:
    # - your topology
    # - tc links
    # - OVS switches (required!)
    # - no default controller (we manually specify RemoteController)
    net = Mininet(
        topo=topo,
        controller=None,
        link=TCLink,
        switch=OVSSwitch,
        autoSetMacs=True
    )

    info("\n*** Adding remote controller (your POX)\n")
    net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6634
    )

    info("\n*** Starting network\n")
    net.start()

    info("\n*** Testing connectivity\n")
    net.pingAll()

    # -----------------------------------------
    # RUN YOUR EXPERIMENT
    # -----------------------------------------

    info("\n*** Launching gradient scripts on all hosts\n")
    start_time = time()

    for rank in range(NUM_HOSTS):
        host = net.get(f"h{rank}")
        info(f" h{rank}: launching gradient script\n")
        host.cmd(f"python3.9 all_to_all_gradients.py {rank} &")  # here made it 3.9 

    info("\n*** Waiting for gradient exchange to finish...\n")
    sleep(10)   # Adjust if needed

    end_time = time()
    total = end_time - start_time

    info(f"\n*** Experiment COMPLETE\nTotal Time = {total:.3f} seconds\n")

    # -----------------------------------------

    info("*** Stopping network\n")
    net.stop()

if __name__ == "__main__":
    main()
