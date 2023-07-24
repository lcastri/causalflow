import math
import random

def linear(x):
    return x

def power(x, exp = 2):
    return x**exp

def square_root(x):
    return x**0.5

def logarithm(x, base=10):
    return math.log(x, base)

def exponential(x, base=math.e):
    return base**x

def sine(x):
    return math.sin(x)

def cosine(x):
    return math.cos(x)

def tangent(x):
    return math.tan(x)

def factorial(x):
    return math.factorial(x)

def absolute_value(x):
    return abs(x)

FUNCTIONS = [linear] 
# FUNCTIONS = [linear, power, exponential, sine, cosine, tangent, factorial, absolute_value] 