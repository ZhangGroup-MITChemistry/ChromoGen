from .A import A

class B:

    def __init__(self):

        self.a = A()

    def __call__(self):
        print(f'A value from B class: {self.a()}',flush=True)

b = B()
b()

