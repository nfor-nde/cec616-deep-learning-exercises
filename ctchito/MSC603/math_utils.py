
"""math_utils — reusable mathematical helpers."""

PI = 3.141592653589793

def circle_area(r):
    """Return area of a circle with radius r."""
    return PI * r ** 2

def circle_circumference(r):
    return 2 * PI * r

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def clamp(value, lo, hi):
    """Clamp value to the range [lo, hi]."""
    return max(lo, min(value, hi))
