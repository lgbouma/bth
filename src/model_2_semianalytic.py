'''
integrate expression for <X_Γ>
17/10/06.1
'''

import numpy as np
from scipy.integrate import trapz

q = np.arange(0.1,1+1e-4,1e-4)
α = 3.5 # M~L^α

BF = 0.44
curlyB = BF/(1-BF)

f_d = 0.475

# lower limit:
integrand = (1+q**α)**(3/2) * (1 + curlyB*(1+q**α)**(3/2)) / \
            ( 1 + curlyB*(1+q**α)**(3/2)* \
              ((1+q**α)**(-3) + f_d * q**(2/3) * (1+q**(-α))**(-3)*q**(-5)  )
            )

numerator = trapz(integrand, q)

denominator = trapz((1+q**α)**(3/2), q)

expected_X_Γ = numerator/denominator

print('lower limit <X_Γ>: ', expected_X_Γ)

# upper limit:

integrand = (1+q**α)**(3/2) * (1 + curlyB*(1+q**α)**(3/2)) \
            / \
            ( 1 + curlyB*(1+q**α)**(-3/2)
            )

numerator = trapz(integrand, q)

denominator = trapz((1+q**α)**(3/2), q)

expected_X_Γ = numerator/denominator

print('upper limit <X_Γ>: ', expected_X_Γ)
