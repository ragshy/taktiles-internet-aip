import numpy as np
import socket 
import time

a = np.array([0,1,0,1,1,1,1,0,0,0],dtype=np.int32).tobytes()

print(a)
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
# Enable broadcasting mode
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
# Set a timeout so the socket does not block
# indefinitely when trying to receive data.
server.settimeout(0.2)
while True:
    server.sendto(a, ('<broadcast>', 9998))
    print("message sent!")
    time.sleep(1)