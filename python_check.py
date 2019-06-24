class Temp:
    def __init__(self):
        self.arr = [1, 1, 1]

    def function(self):
        for a in self.arr:
            a += 1


temp = Temp()
temp.function()