import socket
import sys

HOST, PORT = "localhost", 9999
data = b"Hello, world"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(data)
    data_received = s.recv(1024)

print(f"Sent:     {data}")
print(f"Received from server: {data_received!r}")