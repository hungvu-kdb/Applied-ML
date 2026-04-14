import concurrent.futures
import threading
import time
import random

def worker(n):
    sleep_time = random.uniform(1, 3)
    print(f"Thread {threading.current_thread().name} is processing {n}, sleeping for {sleep_time:.2f} seconds")
    time.sleep(sleep_time)
    print(f"Thread {threading.current_thread().name} finished {n}")

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(worker, range(5))