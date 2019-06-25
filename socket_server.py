import socket
import sys

# AF_INET = IPV4 and SOCK_STREAM = TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# IP = "192.168.43.33"

# vishal lappy ip IP = "10.194.35.37"
# bindal lappy ip
IP = "192.168.43.33"

try:
    s.bind((IP, 1234))
except socket.error as err:
    print("Bind failed, Error Code" + str(err.args[0]) + ", message: " + err.args[1])
    sys.exit()

print("Socket server build successfully")
s.listen(5)  # listen to maximum of 5 people after adding them to queue

while True:
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been extablished")
    # clientsocket.send(bytes("Welcome to the socket"))
    # clientsocket.close()
    burf = clientsocket.recv(64)
    print(burf)
s.close()
