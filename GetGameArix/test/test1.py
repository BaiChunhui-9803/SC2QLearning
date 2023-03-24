class ClassM:
    def __init__(self, a: int):
        self.a = a
        self.mm = 0
        if a == 10:
            self.mm = self.B()

    def B(self):
        return 10

    def get_mm(self):
        return self.mm

    def print_mm(self):
        print('类内打印mm属性', self.mm)


object_m1 = ClassM(5)
print('类外打印mm属性', object_m1.get_mm())
object_m1.print_mm()
object_m2 = ClassM(10)
print('类外打印mm属性', object_m2.get_mm())
object_m2.print_mm()
