
class A:

    def __init__(self):

        self.a = 1

    def __call__(self):
        print(f'A value: {self.a}',flush=True)

