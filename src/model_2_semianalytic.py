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

β = curlyB * (1+q**α)**(3/2)

f_d = 0.475


# lower limit:
f_d1c = (1+q**α)**(-3)
f_d2c = (1+q**(-α))**(-3) * q**(-5)

p_of_q = (1+q**α)**(3/2)


integrand = (1 + β*(1+f_d))/(1+2*β) * (1+β) / \
            (1 + β*(f_d1c + f_d*f_d2c*q**(2/3))) * \
            p_of_q

numerator = trapz(integrand, q)

denominator = trapz(p_of_q, q)

expected_X_Γ = numerator/denominator

print('lower limit <X_Γ>: ', expected_X_Γ)

# upper limit:

#integrand = (1+q**α)**(3/2) * (1 + curlyB*(1+q**α)**(3/2)) \
#            / \
#            ( 1 + curlyB*(1+q**α)**(-3/2)
#            )
#
#numerator = trapz(integrand, q)
#
#denominator = trapz((1+q**α)**(3/2), q)
#
#expected_X_Γ = numerator/denominator
#
#print('upper limit <X_Γ>: ', expected_X_Γ)
