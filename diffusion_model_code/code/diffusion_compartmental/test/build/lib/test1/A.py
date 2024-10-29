
class A:

    def __init__(self):

        self.a = 1

    def __call__(self):
        return f'A value: {self.a}'
        #print(f'A value: {self.a}',flush=True)

