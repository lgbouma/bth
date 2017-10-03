import numpy as np
from scipy.integrate import trapz

x_0, x_1 = 1, 3
y_0, y_1 = 1, 2
x = np.linspace(x_0, x_1, num=int(1e3))
y = np.linspace(y_0, y_1, num=int(2e3))

# 1k x 2k array
f = x[:,None] * y[None,:]

int1 = trapz(f, y, axis=1)

g = x**2

int2 = trapz(g*int1, x)

print('numeric')
print(int2)

print('analytic')
val = 1/8 * (y_1**2 - y_0**2) * (x_1**4 - x_0**4)
print(val)
