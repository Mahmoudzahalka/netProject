#!/usr/bin/env python3
import argparse
import socket
import time
from array import array

def recv_all(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise RuntimeError("Socket closed while expecting data")
        data.extend(chunk)
    return data

def _apply_sock_opts(s, sock_buf, nodelay=True):
    try:
        if nodelay:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    if sock_buf and int(sock_buf) > 0:
        sb = int(sock_buf)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sb)
        except Exception:
            pass
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, sb)
        except Exception:
            pass

def establish_connections(rank, world_size, listen_port, right_ip, right_port, log,
                          sock_buf=0, connect_timeout=60.0):
    log("[rank {}] Phase 0: setting up sockets (listen_port={}, right={}:{})".format(
        rank, listen_port, right_ip, right_port))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _apply_sock_opts(server, sock_buf, nodelay=False)
    server.bind(("", listen_port))
    server.listen(1)

    deadline = time.time() + float(connect_timeout)
    sock_right = None
    while True:
        try:
            sock_right = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _apply_sock_opts(sock_right, sock_buf, nodelay=True)
            sock_right.connect((right_ip, right_port))
            break
        except Exception as e:
            if time.time() > deadline:
                raise RuntimeError("[rank {}] connect to right timed out: {}".format(rank, e))
            log("[rank {}] connect to right failed ({}), retrying...".format(rank, e))
            time.sleep(0.1)

    conn_left, addr_left = server.accept()
    _apply_sock_opts(conn_left, sock_buf, nodelay=True)
    log("[rank {}] accepted connection from left neighbor {}".format(rank, addr_left))

    sock_right.sendall(b"H")
    _ = recv_all(conn_left, 1)

    log("[rank {}] Phase 0: connections established to left and right.".format(rank))
    server.close()
    return conn_left, sock_right

def ring_allreduce(rank, world_size, conn_left, conn_right, grad_elems, log):
    if grad_elems % world_size != 0:
        raise ValueError("grad_elems must be divisible by world_size")

    elems_per_chunk = grad_elems // world_size
    chunk_bytes = elems_per_chunk * 4

    # memory-friendly init (no huge python list)
    grad = array("f", [float(rank)]) * grad_elems

    def send_chunk(chunk_index):
        start = chunk_index * elems_per_chunk
        out_chunk = grad[start:start + elems_per_chunk]
        conn_right.sendall(out_chunk.tobytes())

    def recv_chunk(chunk_index, do_reduce):
        start = chunk_index * elems_per_chunk
        in_bytes = recv_all(conn_left, chunk_bytes)
        in_chunk = array("f")
        in_chunk.frombytes(in_bytes)

        if do_reduce:
            for i in range(elems_per_chunk):
                grad[start + i] += in_chunk[i]
        else:
            grad[start:start + elems_per_chunk] = in_chunk

    log("[rank {}] Phase 1: scatter-reduce starting".format(rank))
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=True)

    log("[rank {}] Phase 2: all-gather starting".format(rank))
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=False)

    log("[rank {}] Ring all-reduce completed".format(rank))
    return grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--right-ip", type=str, required=True)
    parser.add_argument("--right-port", type=int, required=True)
    parser.add_argument("--grad-elems", type=int, default=1048576)
    parser.add_argument("--log-file", type=str, default=None)

    parser.add_argument("--sock-buf", type=int, default=0)
    parser.add_argument("--connect-timeout", type=float, default=60.0)

    args = parser.parse_args()

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = "{} {}".format(ts, msg)
        print(line, flush=True)
        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(line + "\n")

    log("[rank {}] Starting ring node world_size={}, listen_port={}, right={}:{}, grad_elems={}".format(
        args.rank, args.world_size, args.listen_port, args.right_ip, args.right_port, args.grad_elems))

    if args.grad_elems % args.world_size != 0:
        raise ValueError("grad-elems must be divisible by world-size")

    conn_left, conn_right = establish_connections(
        args.rank, args.world_size,
        args.listen_port, args.right_ip, args.right_port,
        log,
        sock_buf=args.sock_buf,
        connect_timeout=args.connect_timeout
    )

    t0 = time.time()
    grad = ring_allreduce(args.rank, args.world_size, conn_left, conn_right, args.grad_elems, log)
    t1 = time.time()

    total = t1 - t0
    log("[rank {}] TOTAL_TIME_SEC={:.6f}".format(args.rank, total))

    # correctness check (after timing)
    expected = (args.world_size * (args.world_size - 1)) / 2.0
    elems_per_chunk = args.grad_elems // args.world_size

    sample_idx = [0, args.grad_elems // 2, args.grad_elems - 1]
    for c in range(args.world_size):
        sample_idx.append(c * elems_per_chunk)

    ok = True
    for idx in sample_idx:
        if abs(grad[idx] - expected) > 1e-3:
            ok = False
            break

    log("[rank {}] PASS={} expected={}".format(args.rank, ok, expected))

    mb = (args.grad_elems * 4) / (1024.0 * 1024.0)
    thr = mb / total if total > 0 else 0.0
    log("[rank {}] RESULT algo=ring world={} bytes={} time={:.6f} thr_MBps={:.2f}".format(
        args.rank, args.world_size, args.grad_elems * 4, total, thr
    ))

    conn_left.close()
    conn_right.close()
    log("[rank {}] DONE".format(args.rank))

if __name__ == "__main__":
    main()
