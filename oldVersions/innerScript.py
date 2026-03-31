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

def establish_connections(rank, world_size, listen_port, right_ip, right_port, log):
    
    '''Phase 0: Only create the TCP connections, no gradient traffic yet.

    Each rank:
      - Binds and listens on (0.0.0.0:listen_port).
      - Connects to its right neighbor (right_ip:right_port) with retry.
      - Accepts a connection from its left neighbor.
      - Performs a 1-byte handshake to ensure both directions are up.
    '''
    log(f"[rank {rank}] Phase 0: setting up sockets (listen_port={listen_port}, "
        f"right={right_ip}:{right_port})")

    # Server for left neighbor
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", listen_port))
    server.listen(1)

    # Client to right neighbor, with retry to avoid timing issues
    while True:
        try:
            sock_right = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_right.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock_right.connect((right_ip, right_port))
            break
        except Exception as e:
            log(f"[rank {rank}] connect to right failed ({e}), retrying...")
            time.sleep(0.1)

    # Accept connection from left neighbor
    conn_left, addr_left = server.accept()
    conn_left.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    log(f"[rank {rank}] accepted connection from left neighbor {addr_left}")

    # One-byte handshake to be sure both directions are live
    sock_right.sendall(b"H")
    _ = recv_all(conn_left, 1)

    log(f"[rank {rank}] Phase 0: connections established to left and right.")
    server.close()
    return conn_left, sock_right

def ring_allreduce(rank, world_size, conn_left, conn_right, grad_elems, log):
    
    '''Standard ring all-reduce with two phases:
      - Phase 1: scatter-reduce
      - Phase 2: all-gather

    Gradient is a float32 array of length grad_elems.
    We partition into 'world_size' contiguous chunks.
    Each rank starts with its own local gradient initialized to 'rank'.
    '''

    # Make grad_elems divisible by world_size (enforced in caller)
    elems_per_chunk = grad_elems // world_size
    assert grad_elems % world_size == 0

    # Local gradient initialized with rank (just for sanity/testing)
    grad = array("f", [float(rank)] * grad_elems)

    # Each float32 is 4 bytes
    chunk_bytes = elems_per_chunk * 4

    def send_chunk(chunk_index):
        start = chunk_index * elems_per_chunk
        out_chunk = grad[start : start + elems_per_chunk]
        conn_right.sendall(out_chunk.tobytes())

    def recv_chunk(chunk_index, do_reduce):
        start = chunk_index * elems_per_chunk
        in_bytes = recv_all(conn_left, chunk_bytes)
        in_chunk = array("f")
        in_chunk.frombytes(in_bytes)

        if do_reduce:
            # scatter-reduce: sum into local gradient chunk
            for i in range(elems_per_chunk):
                grad[start + i] += in_chunk[i]
        else:
            # all-gather: overwrite with fully reduced chunk
            grad[start : start + elems_per_chunk] = in_chunk

    # -------------------------
    # Phase 1: scatter-reduce
    # -------------------------
    log(f"[rank {rank}] Phase 1: scatter-reduce starting "
        f"(grad_elems={grad_elems}, chunks={world_size}, chunk_bytes={chunk_bytes})")

    for step in range(world_size - 1):
        # Index of chunk we send this step
        send_index = (rank - step) % world_size
        # Index of chunk we receive and reduce this step
        recv_index = (rank - step - 1) % world_size

        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=True)

        log(f"[rank {rank}] Phase 1 step {step+1}/{world_size-1}: "
            f"sent chunk {send_index}, reduced chunk {recv_index}")

    # -------------------------
    # Phase 2: all-gather
    # -------------------------
    log(f"[rank {rank}] Phase 2: all-gather starting")

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size

        send_chunk(send_index)
        recv_chunk(recv_index, do_reduce=False)

        log(f"[rank {rank}] Phase 2 step {step+1}/{world_size-1}: "
            f"sent chunk {send_index}, received fully-reduced chunk {recv_index}")

    log(f"[rank {rank}] Ring all-reduce completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--right-ip", type=str, required=True)
    parser.add_argument("--right-port", type=int, required=True)
    # Total gradient length (float32 elements). Must be divisible by world_size.
    parser.add_argument("--grad-elems", type=int, default=1048576)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"{ts} {msg}"
        print(line, flush=True)
        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(line + "\n")

    log(f"[rank {args.rank}] Starting ring node with world_size={args.world_size}, "
        f"listen_port={args.listen_port}, right={args.right_ip}:{args.right_port}, "
        f"grad_elems={args.grad_elems}")

    conn_left, conn_right = establish_connections(
        args.rank, args.world_size,
        args.listen_port, args.right_ip, args.right_port,
        log
    )

    t0 = time.time()
    ring_allreduce(args.rank, args.world_size, conn_left, conn_right, args.grad_elems, log)
    t1 = time.time()

    log(f"[rank {args.rank}] TOTAL_TIME_SEC={t1 - t0:.6f}")

    conn_left.close()
    conn_right.close()
    log(f"[rank {args.rank}] Connections closed, exiting.")

if __name__ == "__main__":
    main()



