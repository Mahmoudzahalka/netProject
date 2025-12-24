from mininet.net import Mininet
from mininet.log import setLogLevel, info
from time import sleep, time

from fatTree import FatTreeTopo


def onepacket_ring_allreduce(net):
    hosts = sorted(net.hosts, key=lambda h: h.name)
    N = len(hosts)

    info("*** Hosts in ring order:\n")
    for h in hosts:
        info(f"{h.name} ")
    info("\n\n")

    info("=== ONE-PACKET RING ALL-REDUCE ===\n")
    info("(Phase 1 + Phase 2 implemented)\n\n")

    # ----------------------------------------------------------
    # Start TCP iperf servers on all hosts
    # ----------------------------------------------------------
    info("*** Starting TCP iperf servers on all hosts...\n")
    for h in hosts:
        h.cmd("pkill iperf")
        h.cmd("iperf -s &")
    sleep(1)

    # ----------------------------------------------------------
    # MEASUREMENT: lists to store results
    # ----------------------------------------------------------
    iteration_times = []
    iteration_tail_fcts = []

    total_start = time()

    # ==========================================================
    # PHASE 1 — REDUCE-SCATTER (existing code)
    # ==========================================================
    info("=== PHASE 1: REDUCE-SCATTER ===\n")

    for step in range(N - 1):
        info(f"--- Step {step} ---\n")

        step_start = time()
        flow_fcts = []

        for i in range(N):
            src = hosts[i]
            dst = hosts[(i + 1) % N]

            dst_ip = dst.IP()

            # -------------------------
            # Flow FCT measurement
            # -------------------------
            flow_start = time()
            src.cmd(f"iperf -c {dst_ip} -b 1M -l 1400 -t 0.5 &")
            flow_end = time()

            fct = flow_end - flow_start
            flow_fcts.append(fct)

            info(f"{src.name} -> {dst.name}  FCT={fct*1000:.2f} ms\n")

        step_end = time()
        iteration_time = step_end - step_start
        iteration_times.append(iteration_time)

        tail_fct = max(flow_fcts)
        iteration_tail_fcts.append(tail_fct)

        info(f"Step {step}: iteration_time={iteration_time:.4f}s, "
             f"TAIL_FCT={tail_fct*1000:.2f}ms\n")

        sleep(0.3)

    # ==========================================================
    # PHASE 2 — ALL-GATHER  (NEW)
    # ==========================================================
    info("\n=== PHASE 2: ALL-GATHER ===\n")
    info("Broadcasting the reduced result around the ring...\n")

    for step in range(N - 1):
        info(f"--- Step {step} ---\n")

        step_start = time()
        flow_fcts = []

        for i in range(N):
            src = hosts[i]
            dst = hosts[(i + 1) % N]

            dst_ip = dst.IP()

            flow_start = time()
            src.cmd(f"iperf -c {dst_ip} -b 1M -l 1400 -t 0.5 &")
            flow_end = time()

            fct = flow_end - flow_start
            flow_fcts.append(fct)

            info(f"{src.name} -> {dst.name}  FCT={fct*1000:.2f} ms\n")

        step_end = time()
        iteration_time = step_end - step_start
        iteration_times.append(iteration_time)

        tail_fct = max(flow_fcts)
        iteration_tail_fcts.append(tail_fct)

        info(f"Phase2 Step {step}: iter_time={iteration_time:.4f}s, "
             f"TAIL_FCT={tail_fct*1000:.2f}ms\n")

        sleep(0.3)

    # ==========================================================
    # PRINT FINAL RESULTS
    # ==========================================================
    total_end = time()
    total_time = total_end - total_start

    info("\n=== DONE (Phase 1 + Phase 2) ===\n")
    info(f"TOTAL COMMUNICATION TIME = {total_time:.4f} seconds\n\n")

    info("=== PER-STEP METRICS ===\n")
    for i in range(len(iteration_times)):
        info(f"Step {i}: iter_time={iteration_times[i]:.4f}s, "
             f"tail_fct={iteration_tail_fcts[i]*1000:.2f}ms\n")

    info("\nAll hosts now hold the final reduced gradient.\n")


if __name__ == "__main__":
    setLogLevel("info")

    topo = FatTreeTopo(4)

    net = Mininet(topo=topo, controller=None, autoSetMacs=True)

    net.start()

    info("*** Setting static ARP...\n")
    net.staticArp()

    onepacket_ring_allreduce(net)

    net.stop()
