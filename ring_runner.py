import os
import time

PROJECT_DIR = "/home/msa/netProject-main"
INNER_LOCAL = os.path.join(PROJECT_DIR, "innerScript.py")

def _read_local_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _push_script_to_host(host, script_text, dst="/tmp/ring_node.py"):
    host.cmd("cat > {dst} <<'PYEOF'\n{code}\nPYEOF\nchmod +x {dst}".format(
        dst=dst, code=script_text
    ))

def run_ring_allreduce(net,
                       base_port=5000,
                       grad_bytes=100 * 1024 * 1024,
                       timeout_sec=300,
                       sock_buf=0,
                       connect_timeout=60.0,
                       host_order=None):
    """
    Launch ring all-reduce on ALL hosts named h_*.

    - ranks: assigned by sorted(host.name) unless host_order is provided
    - listen port per rank: base_port + rank
    - right neighbor of rank r: rank (r+1 mod world)
    - copies innerScript.py into each host at /tmp/ring_node.py
    - runs it and collects PASS + TOTAL_TIME_SEC
    """

    # 1) Select hosts
    hosts_all = [h for h in net.hosts if h.name.startswith("h_")]
    if not hosts_all:
        print("[ring_runner] No hosts found starting with h_")
        return

    if host_order is None:
        hosts = sorted(hosts_all, key=lambda x: x.name)
    else:
        # host_order is list of names, e.g. ["h_p0_e0_2", "h_p0_e0_3", ...]
        name_to_host = {h.name: h for h in hosts_all}
        hosts = []
        for name in host_order:
            if name not in name_to_host:
                raise ValueError("host_order contains unknown host name: {}".format(name))
            hosts.append(name_to_host[name])

    world = len(hosts)

    # 2) Compute grad_elems
    if grad_bytes % 4 != 0:
        raise ValueError("grad_bytes must be divisible by 4 (float32)")
    grad_elems = grad_bytes // 4
    if grad_elems % world != 0:
        raise ValueError("grad_elems={} not divisible by world={}. Choose different grad_bytes.".format(grad_elems, world))

    # 3) Read innerScript.py
    if not os.path.exists(INNER_LOCAL):
        # fallback: try current working directory
        alt = os.path.join(os.getcwd(), "innerScript.py")
        if os.path.exists(alt):
            script_path = alt
        else:
            print("[ring_runner] ERROR: cannot find innerScript.py at:")
            print("  ", INNER_LOCAL)
            print("and also not found in current directory.")
            return
    else:
        script_path = INNER_LOCAL

    code = _read_local_file(script_path)

    # 4) Print mapping
    print("[ring_runner] world={}, grad_bytes={}, grad_elems={}, sock_buf={}, connect_timeout={}".format(
        world, grad_bytes, grad_elems, sock_buf, connect_timeout
    ))
    for rank, h in enumerate(hosts):
        right = hosts[(rank + 1) % world]
        print("  rank {:2d} -> {:12s} ip={} listen_port={}  right={:12s} right_ip={} right_port={}".format(
            rank, h.name, h.IP(), base_port + rank, right.name, right.IP(), base_port + ((rank + 1) % world)
        ))

    # 5) Cleanup old runs
    for h in hosts:
        h.cmd("pkill -f /tmp/ring_node.py >/dev/null 2>&1 || true")
        h.cmd("rm -rf /tmp/ring_logs && mkdir -p /tmp/ring_logs")

    # 6) Push script into each host
    for h in hosts:
        _push_script_to_host(h, code)

    # 7) Launch all ranks
    for rank, h in enumerate(hosts):
        right = hosts[(rank + 1) % world]
        listen_port = base_port + rank
        right_port = base_port + ((rank + 1) % world)
        logf = "/tmp/ring_logs/rank{}.log".format(rank)

        cmd = (
            "python3 /tmp/ring_node.py "
            "--rank {rank} --world-size {world} "
            "--listen-port {lp} --right-ip {rip} --right-port {rp} "
            "--grad-elems {ge} --log-file {log} "
            "--connect-timeout {ct} "
        ).format(rank=rank, world=world, lp=listen_port, rip=right.IP(), rp=right_port,
                 ge=grad_elems, log=logf, ct=connect_timeout)

        if sock_buf and int(sock_buf) > 0:
            cmd += "--sock-buf {} ".format(int(sock_buf))

        cmd += "> {log} 2>&1 &".format(log=logf)
        h.cmd(cmd)

    # 8) Wait for completion
    deadline = time.time() + timeout_sec
    done = set()
    results = {}

    while time.time() < deadline and len(done) < world:
        for rank, h in enumerate(hosts):
            if rank in done:
                continue
            out = h.cmd("grep -E 'PASS=|TOTAL_TIME_SEC=' /tmp/ring_logs/rank{r}.log | tail -n 10".format(r=rank))
            if ("PASS=" in out) and ("TOTAL_TIME_SEC=" in out):
                done.add(rank)
                results[rank] = out
        time.sleep(1)

    # 9) Summary
    print("[ring_runner] Completed {}/{}".format(len(done), world))
    for rank, h in enumerate(hosts):
        tail = h.cmd("grep -E 'PASS=|TOTAL_TIME_SEC=' /tmp/ring_logs/rank{r}.log | tail -n 5".format(r=rank))
        print("rank {:2d} {:12s}: {}".format(rank, h.name, tail.strip()))

    if len(done) < world:
        print("[ring_runner] TIMEOUT. Debug one rank like:")
        print("  <host> tail -n 50 /tmp/ring_logs/rankX.log")

# auto-run when executed via containernet CLI: py exec(open('ring_runner.py').read())
if "net" in globals():
    run_ring_allreduce(net)
else:
    print("Run inside containernet> with: py exec(open('/home/msa/netProject-main/ring_runner.py').read())")
