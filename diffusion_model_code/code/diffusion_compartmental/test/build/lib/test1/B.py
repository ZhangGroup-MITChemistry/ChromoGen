from .A import A

class B:

    def __init__(self):

        self.a = A()

    def __call__(self):
        print(f'self.a: {self.a}',flush=True)
        print(f'self.a.a: {self.a.a}',flush=True)
        print(f'self.a(): {self.a()}',flush=True)

#b = B()
#b()

