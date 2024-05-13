# test for comprehensions in Python

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
        
    return True

original: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

new: list = [x for x in original]
print(f"new: {new}")

odd: list = [x for x in original if x % 2 == 1]
print(f"odd: {odd}")

even: list = [x for x in original if x % 2 == 0]
print(f"even: {even}")

prime: list = [x for x in original if is_prime(x)]
print(f"prime: {prime}")

print("-----------")

set_: set = {x for x in original}
print(f"set: {set_}")

odd_set: set = {x for x in original if x % 2 == 1}
print(f"odd_set: {odd_set}")

prime_set: set = {x for x in original if is_prime(x)}
print(f"prime_set: {prime_set}")

print("-----------")

alphabet: str = "abcdefghijklmnopqrstuvwxyz"

dict_: dict = {c: ord(c) for c in alphabet}

def display(c: chr) -> str:
    return f"{c}: {dict_[c]}"

print(display('a'))
print(display('c'))
print(display('d'))
print(display('m'))