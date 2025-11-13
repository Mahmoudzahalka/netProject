#!/usr/bin/env python3
import socket
import time
import sys


#initial exchange simulation using UDP, might change the algorithm or the protocol later


# for this script check the case that when running it with pox ring controller 
# if we have to change anything in that case and if not why




NUM_HOSTS = 16
PACKET_SIZE = 1400            # bytes (simulating 1 gradient packet)
ITERATIONS = 7
SEND_PORT = 5001

# --------------------------------------------------
# Get IP for host rank using your deterministic FatTree IP rules
# --------------------------------------------------
def get_ip(rank):
    pod = rank // 4
    sw  = (rank % 4) // 2
    host = 2 + (rank % 2)      # host IDs are 2 and 3
    return f"10.{pod}.{sw}.{host}"

# --------------------------------------------------
# Main traffic generation
# --------------------------------------------------
def main(my_rank):
    my_ip = get_ip(my_rank)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_sock.bind((my_ip, 0))  # bind to this host’s IP

    payload = b'x' * PACKET_SIZE

    print(f"[Host {my_rank}] Starting all-to-all gradient exchange...")

    for it in range(ITERATIONS):
        print(f"[Host {my_rank}] Iteration {it+1}/{ITERATIONS}")

        start = time.time()

        for dst_rank in range(NUM_HOSTS):
            if dst_rank == my_rank:
                continue

            dst_ip = get_ip(dst_rank)
            send_sock.sendto(payload, (dst_ip, SEND_PORT))

        end = time.time()
        print(f"[Host {my_rank}] Iteration {it+1} done in {end-start:.6f} seconds")

    print(f"[Host {my_rank}] Completed all 7 iterations.")

if __name__ == "__main__":
    my_rank = int(sys.argv[1])
    main(my_rank)
