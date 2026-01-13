#!/usr/bin/env python3
import argparse
import socket
import struct
import time
from array import array
from typing import Optional, Dict, List

def recv_all(sock: socket.socket, nbytes: int) -> bytes:
    data = bytearray()
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise RuntimeError("Socket closed early")
        data.extend(chunk)
    return bytes(data)

def log_line(log_file: Optional[str], msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = "{} {}".format(ts, msg)
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")

def children_of(rank: int, world: int, arity: int) -> List[int]:
    kids = []
    for i in range(1, arity + 1):
        c = arity * rank + i
        if c < world:
            kids.append(c)
    return kids

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--arity", type=int, default=2)
    ap.add_argument("--listen-port", type=int, required=True)

    ap.add_argument("--parent-ip", type=str, default=None)
    ap.add_argument("--parent-port", type=int, default=None)

    ap.add_argument("--grad-bytes", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--block-bytes", type=int, default=4 * 1024 * 1024)

    ap.add_argument("--log-file", type=str, default=None)
    ap.add_argument("--connect-timeout", type=float, default=60.0)
    args = ap.parse_args()

    rank = args.rank
    world = args.world_size
    arity = args.arity

    if world <= 0:
        raise ValueError("world-size must be > 0")
    if not (0 <= rank < world):
        raise ValueError("rank must be in [0, world-size)")
    if (args.grad_bytes % 4) != 0 or (args.block_bytes % 4) != 0:
        raise ValueError("grad-bytes and block-bytes must be divisible by 4 (float32)")

    grad_elems = args.grad_bytes // 4
    block_elems = args.block_bytes // 4
    expected = (world * (world - 1)) / 2.0

    parent_rank = (rank - 1) // arity if rank != 0 else None
    kids = children_of(rank, world, arity)

    log_line(args.log_file,
             "[rank {}] TREE start world={} arity={} parent={} kids={} listen={} grad_bytes={} block_bytes={}".format(
                 rank, world, arity, parent_rank, kids, args.listen_port, args.grad_bytes, args.block_bytes))

    # Listen for children
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", args.listen_port))
    server.listen(max(1, len(kids)))

    # Connect to parent (if not root)
    parent_sock = None  # type: Optional[socket.socket]
    if rank != 0:
        if not args.parent_ip or not args.parent_port:
            raise ValueError("Non-root must provide --parent-ip and --parent-port")

        deadline = time.time() + args.connect_timeout
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((args.parent_ip, args.parent_port))
                s.sendall(struct.pack("!I", rank))  # identify to parent
                parent_sock = s
                break
            except Exception as e:
                if time.time() > deadline:
                    raise RuntimeError("[rank {}] connect parent timeout: {}".format(rank, e))
                time.sleep(0.1)

        log_line(args.log_file, "[rank {}] Connected to parent {}:{}".format(rank, args.parent_ip, args.parent_port))

    # Accept children connections
    child_socks = {}  # type: Dict[int, socket.socket]
    for _ in range(len(kids)):
        conn, addr = server.accept()
        cr = struct.unpack("!I", recv_all(conn, 4))[0]
        child_socks[cr] = conn
        log_line(args.log_file, "[rank {}] Accepted child {} from {}".format(rank, cr, addr))

    server.close()

    # Run streaming reduce + broadcast
    t0 = time.time()
    ok = True

    offset = 0
    block_idx = 0
    while offset < grad_elems:
        n = min(block_elems, grad_elems - offset)
        nbytes = n * 4

        # initialize local block = rank (more memory-friendly than a huge python list)
        block = array("f", [float(rank)]) * n

        # Reduce: receive from children and sum
        for cr in sorted(child_socks.keys()):
            data = recv_all(child_socks[cr], nbytes)
            arr = array("f")
            arr.frombytes(data)
            for i in range(n):
                block[i] += arr[i]

        # send to parent
        if parent_sock is not None:
            parent_sock.sendall(block.tobytes())

        # Broadcast: receive final from parent, then send to children
        if parent_sock is not None:
            data = recv_all(parent_sock, nbytes)
            block = array("f")
            block.frombytes(data)

        for cr in sorted(child_socks.keys()):
            child_socks[cr].sendall(block.tobytes())

        # correctness sampling (few blocks only)
        if block_idx in (0, 1) or (offset + n) >= grad_elems:
            sample = [block[0], block[n // 2], block[-1]]
            mn = min(block)
            mx = max(block)
            pass_block = (abs(mn - expected) < 1e-3) and (abs(mx - expected) < 1e-3) and all(abs(x - expected) < 1e-3 for x in sample)
            ok = ok and pass_block
            log_line(args.log_file,
                     "[rank {}] CHECK block={} expected={} sample={} min={} max={} PASS={}".format(
                         rank, block_idx, expected, sample, mn, mx, pass_block))

        offset += n
        block_idx += 1

    total = time.time() - t0
    log_line(args.log_file, "[rank {}] PASS={}".format(rank, ok))
    log_line(args.log_file, "[rank {}] TOTAL_TIME_SEC={:.6f}".format(rank, total))
    log_line(args.log_file, "[rank {}] DONE".format(rank))

    # cleanup
    for s in child_socks.values():
        try:
            s.close()
        except Exception:
            pass
    if parent_sock:
        try:
            parent_sock.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
