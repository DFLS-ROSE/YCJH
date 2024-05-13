# test for decorators in Python

import time
from tqdm import tqdm

def timer(func):

    def wrapper(*args):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time:.1f} seconds")
        return result

    return wrapper

@timer
def count_primes(range_start, range_end):
    range_start = range_start if range_start > 2 else 2
    count = 0
    for num in tqdm(range(range_start, range_end + 1),desc="Counting primes"):  # 1不是质数，从2开始
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):  # 使用试除法判断质数
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    print(f"Number of primes: {count}")

@timer
def count_primes_Eratosthenes(n):
    if n < 2:
        return 0

    # 创建一个标记列表，用于标记每个数字是否为质数
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    # 使用埃拉托斯特尼筛法找出质数
    for i in tqdm(range(2, int(n**0.5) + 1),desc="Counting primes_from_Eratosthenes"):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    # 统计质数的个数
    count = sum(1 for x in is_prime if x)
    print(f"Number of primes_from_Eratosthenes: {count}")

if __name__ == '__main__':

    count_primes(1, 10**6)
    count_primes_Eratosthenes(10**6)

