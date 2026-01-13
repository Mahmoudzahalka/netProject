import os
import time

# IMPORTANT: adjust this if your project folder path is different
PROJECT_DIR = "/home/msa/netProject-main"
TREE_NODE_LOCAL = os.path.join(PROJECT_DIR, "tree_node.py")

def _read_local_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _push_script_to_host(host, script_text, dst="/tmp/tree_node.py"):
    """
    Put tree_node.py code inside the host container as /tmp/tree_node.py
    (Hosts are containers; they usually cannot see your VM files directly.)
    """
    host.cmd("cat > {dst} <<'PYEOF'\n{code}\nPYEOF\nchmod +x {dst}".format(
        dst=dst, code=script_text
    ))

def run_tree_allreduce(net,
                       arity=2,
                       base_port=6000,
                       grad_bytes=100 * 1024 * 1024,
                       block_bytes=4 * 1024 * 1024,
                       timeout_sec=300):
    """
    Launch Tree All-Reduce on ALL hosts named h_*.

    - ranks: assigned by sorted(host.name)
    - listen port per rank: base_port + rank
    - parent of rank r (r>0): (r-1)//arity
    - copies tree_node.py into each host at /tmp/tree_node.py
    - runs it and collects PASS + TOTAL_TIME_SEC
    """

    # 1) Pick hosts (only end-hosts, not routers)
    hosts = [h for h in net.hosts if h.name.startswith("h_")]
    hosts = sorted(hosts, key=lambda x: x.name)
    world = len(hosts)
    if world == 0:
        print("[tree_runner] No hosts found starting with h_")
        return

    # 2) Read tree_node.py from your VM filesystem
    if not os.path.exists(TREE_NODE_LOCAL):
        print("[tree_runner] ERROR: cannot find tree_node.py at:", TREE_NODE_LOCAL)
        print("Make sure tree_node.py exists in your project folder.")
        return
    tree_node_code = _read_local_file(TREE_NODE_LOCAL)

    # 3) Print mapping rank -> host -> ip -> ports
    print("[tree_runner] world={}, arity={}, grad_bytes={}, block_bytes={}".format(
        world, arity, grad_bytes, block_bytes
    ))
    for rank, h in enumerate(hosts):
        print("  rank {:2d} -> {:12s} ip={} listen_port={}".format(
            rank, h.name, h.IP(), base_port + rank
        ))

    # 4) Cleanup old processes/logs
    for h in hosts:
        h.cmd("pkill -f /tmp/tree_node.py >/dev/null 2>&1 || true")
        h.cmd("rm -rf /tmp/tree_logs && mkdir -p /tmp/tree_logs")

    # 5) Copy tree_node.py into each host container
    for h in hosts:
        _push_script_to_host(h, tree_node_code)

    # 6) Launch all ranks
    for rank, h in enumerate(hosts):
        listen_port = base_port + rank
        logf = "/tmp/tree_logs/rank{}.log".format(rank)

        if rank == 0:
            # root has no parent
            cmd = (
                "python3 /tmp/tree_node.py "
                "--rank {rank} --world-size {world} --arity {arity} "
                "--listen-port {lp} "
                "--grad-bytes {gb} --block-bytes {bb} "
                "--log-file {log} "
                "> {log} 2>&1 &"
            ).format(rank=rank, world=world, arity=arity,
                     lp=listen_port, gb=grad_bytes, bb=block_bytes, log=logf)
        else:
            parent_rank = (rank - 1) // arity
            parent_ip = hosts[parent_rank].IP()
            parent_port = base_port + parent_rank

            cmd = (
                "python3 /tmp/tree_node.py "
                "--rank {rank} --world-size {world} --arity {arity} "
                "--listen-port {lp} "
                "--parent-ip {pip} --parent-port {pp} "
                "--grad-bytes {gb} --block-bytes {bb} "
                "--log-file {log} "
                "> {log} 2>&1 &"
            ).format(rank=rank, world=world, arity=arity,
                     lp=listen_port, pip=parent_ip, pp=parent_port,
                     gb=grad_bytes, bb=block_bytes, log=logf)

        h.cmd(cmd)

    # 7) Wait for completion
    deadline = time.time() + timeout_sec
    done = set()

    while time.time() < deadline and len(done) < world:
        for rank, h in enumerate(hosts):
            if rank in done:
                continue
            out = h.cmd("grep -E 'PASS=|TOTAL_TIME_SEC=' /tmp/tree_logs/rank{r}.log | tail -n 5".format(r=rank))
            if ("PASS=" in out) and ("TOTAL_TIME_SEC=" in out):
                done.add(rank)
        time.sleep(1)

    # 8) Summary
    print("[tree_runner] Completed {}/{}".format(len(done), world))
    for rank, h in enumerate(hosts):
        tail = h.cmd("grep -E 'PASS=|TOTAL_TIME_SEC=' /tmp/tree_logs/rank{r}.log | tail -n 3".format(r=rank))
        print("rank {:2d} {:12s}: {}".format(rank, h.name, tail.strip()))

    if len(done) < world:
        print("[tree_runner] TIMEOUT. Debug one rank like:")
        print("  <host> tail -n 40 /tmp/tree_logs/rankX.log")

# When you run via: containernet> py exec(open('tree_runner.py').read())
# Mininet provides "net" in globals(), so we auto-run.
if "net" in globals():
    run_tree_allreduce(net)
else:
    print("Run inside containernet> with: py exec(open('tree_runner.py').read())")
