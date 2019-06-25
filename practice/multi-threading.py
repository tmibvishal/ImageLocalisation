import time
import threading
import socket
import sys

# with definations
'''
def calc_square(numbers):
    print("calculating squares")
    for n in numbers:
        time.sleep(0.2)
        print("square: ", n * n)


def calc_cube(numbers):
    print("calculating cubes")
    for n in numbers:
        time.sleep(0.2)
        print("cube: ", n * n * n)


arr = [i for i in range(10)]

t = time.time()

t1 = threading.Thread(target=calc_square, args=(arr,))
t2 = threading.Thread(target=calc_cube, args=(arr,))

t1.start()
t2.start()

t1.join()
t2.join()

print("done in: ", time.time() - t)
print("yo i am done with the work")
'''

# AF_INET = IPV4 and SOCK_STREAM = TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Vishal laptop ip in SIT IP = "10.194.35.37"
# Vishal laptop ip when connected using hotspot IP = "192.168.43.33"
# Bindal laptop ip in SIT IP = "10.194.55.238"
IP = "192.168.43.33"

try:
    s.bind((IP, 1234))
except socket.error as err:
    print("Bind failed, Error Code" + str(err.args[0]) + ", message: " + err.args[1])
    sys.exit()

print("Socket server build successfully")
s.listen(5)  # listen to maximum of 5 people after adding them to queue


# with classes
class Hello(threading.Thread):
    def run(self):
        for i in range(500):
            time.sleep(0.2)
            print("Hello")


class SocketServer(threading.Thread):
    def run(self):
        while True:
            client_socket, address = s.accept()
            print(f"Connection from {address} has been established")
            # client_socket.send(bytes("Welcome to the socket"))
            # client_socket.close()
            buffer = client_socket.recv(64)
            print(buffer)
            if 0xFF == ord('q'):
                break
        s.close()


t1 = Hello()
t2 = SocketServer()

t1.start()
t2.start()

t1.join()
t2.join()