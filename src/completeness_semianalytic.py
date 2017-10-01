from scipy.integrate import trapz
import matplotlib.pyplot as plt
import numpy as np

alpha = 3.5
f_d1c_upper = (1+0.1**alpha)**(-3)
f_d1c_lower = 2**(-3)

f_d1c = np.linspace(f_d1c_lower, f_d1c_upper, num=int(5e4))

integrand = f_d1c**5 * (np.abs(f_d1c**3 -1))**(-5/7)

norm = trapz(integrand, f_d1c)
print('normalization is: {:.8f}'.format(norm))

integrand /= norm

f,ax=plt.subplots()
ax.plot(f_d1c, integrand)

ax.set(
xlabel='$f_{d1,c}$',
ylabel='prob',
yscale='log'
)

f.savefig('prob_fd1c.pdf',
        bbox_inches='tight')
