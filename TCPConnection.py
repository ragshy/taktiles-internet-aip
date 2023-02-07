import socket


class Server():
  """ class that holds the TCP server that sends data (angles,position) to other user (client) 
  and receives data from the client
  """
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
  """ class that holds the TCP client that sends data (angles,position) to other user (server) 
  and receives data from the server
  """
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
