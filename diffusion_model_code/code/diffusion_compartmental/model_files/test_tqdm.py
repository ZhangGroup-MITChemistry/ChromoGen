from tqdm.auto import tqdm 
import time

def tqdm_test():

    for _ in range(2):
        for _ in tqdm(range(20),desc='test',position=0,leave=True):#False):
            time.sleep(.05)

tqdm_test()
print('test')

