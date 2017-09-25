import matplotlib.pyplot as plt
import numpy as np

alpha = np.linspace(1,4,num=2000)
f,ax=plt.subplots()
ax.plot(alpha, (1-alpha)/alpha)

f.savefig('alpha_power_allowed_vals.pdf',
        bbox_inches='tight')


alpha = 3.5
f_d1c_upper = (1+0.1**alpha)**(-3)
f_d1c_lower = 2**(-3)

f_d1c = np.linspace(f_d1c_lower, f_d1c_upper, num=int(1e4))

import IPython; IPython.embed()
integrand = f_d1c**5 * (f_d1c**3 -1)**(-5/7)

f,ax=plt.subplots()
ax.plot(f_d1c, integrand)

f.savefig('temp.pdf',
        bbox_inches='tight')
