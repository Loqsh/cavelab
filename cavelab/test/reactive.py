from __future__ import print_function
from rx import Observable, Observer
import time

t1 = time.time()

def download(name):
    time.sleep(0.5)
    print('downloaded',name)
    return name

def process(name):
    time.sleep(1)
    print('process',name)
    return name

def upload(name):
    time.sleep(0.5)
    print('upload',name)
    return name

def done(name):
    t2 =time.time()
    print('done',name, t2-t1)

err = lambda e: print(e)

three_random_ints = Observable.from_([30,31,32,33,34,35])
three_random_ints\
    .map(Observable.to_async(download))\
    .map(process)\
    .map(Observable.to_async(upload))\
    .subscribe(on_next=done, on_completed=lambda: print("PROCESS 1 done!"), on_error=err)
    #.observe_on(pool_scheduler) \
print(time.time()-t1)
#three_random_ints.subscribe_on(pool_scheduler).subscribe(process)
#three_random_ints.subscribe_on(pool_scheduler).subscribe(upload)
#print(t2-t1)
