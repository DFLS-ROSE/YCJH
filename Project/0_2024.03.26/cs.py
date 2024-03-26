from tqdm import tqdm
import time

items = range(10)

for i in tqdm(items,desc="1",total=len(items)):
    time.sleep(0.1)
