import time
import threading

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


# with classes
class Hello(threading.Thread):
    def run(self):
        for i in range(5):
            time.sleep(0.2)
            print("Hello")


class Hi(threading.Thread):
    def run(self):
        for i in range(5):
            time.sleep(0.2)
            print("Hi")


t1 = Hello()
t2 = Hi()

t1.start()
t2.start()

t1.join()
t2.join()