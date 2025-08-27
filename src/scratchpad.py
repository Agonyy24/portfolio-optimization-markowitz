import matplotlib.pyplot as plt
from scipy.optimize import minimize   
import numpy as np

# Funkcja celu
def f(x):
    return x[0]**2 + 3*x[0] + 4

# Punkt startowy
x0 = np.array([10.0])  # zaczynamy od x=0

# Wywołanie optymalizacji
res = minimize(f, x0, method='BFGS')

print("Czy sukces:", res.success)
print("Komunikat:", res.message)
print("Optymalne x:", res.x)
print("Minimalna wartość f(x):", res.fun)

