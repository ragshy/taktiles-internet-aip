import socket


class Server():
  def __init__(self,HOST="localhost", PORT =  9999) -> None:
    self.HOST = HOST
    self.PORT = PORT
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.bind((HOST, PORT))
    self.s.listen(5)
    self.conn, addr = self.s.accept()
    
  def send(self,data):
    self.conn.sendall(data)

  def receive(self):
    data = self.conn.recv(1024)
    return data


class Client():
  def __init__(self,HOST="localhost", PORT =  9999) -> None:
    self.HOST = HOST
    self.PORT = PORT
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.connect((self.HOST, self.PORT))

  def send(self,data):
    self.s.sendall(data)

  def receive(self):
    data = self.s.recv(1024)
    return data
