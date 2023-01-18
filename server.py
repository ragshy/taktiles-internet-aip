import socket

HOST, PORT = '10.181.173.23', 9999

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            print(f"Received from client: {data}")
            data = data +b'berrryy'
            if data:
                conn.sendall(data)