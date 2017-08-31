'''
compute and plot the joint 2d probability distribution function of a binary
star's position and mass ratio in a magnitude-limited sample.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

r = np.linspace(0,0.5,3e3)
q = np.linspace(0,1,3e3)[::-1]

L_1_N = 1
F_lim_N = 1

d_max_of_q = ( L_1_N / (4*np.pi*F_lim_N) * (1+q**3) )**(1/2)

d_max_s = (L_1_N / (4*np.pi*F_lim_N))**(1/2)
# integral of (1+q^3)^(-9/2)dq from q=0.1 to 1 evaluates to
I_1 = 0.4645286925158471
# integral of (1+q^3)^(3/2)dq from q=0.1 to 1 evaluates to
I_2 = 1.323588493214896
# normalization constant brings integrals to 1
norm = 9/(I_1*I_2)

prob = norm * I_1 * r**2 / (3*d_max_s**3)
prob = prob[:,None] * np.ones(len(q))[None,:]

# Below line: incorrect calculation, based on math that I don't understand my
# error from, but that I work around.
## prob = np.outer(1/3 * r**2, 1/(d_max_of_q**3))

prob_pre = np.copy(prob)

print('before mask, distribn integrates to {:.3f}'.format(
    trapz(trapz(prob, r, axis=0), q[::-1])))
print('before mask, distribn integrates to {:.3f}'.format(
    trapz(trapz(prob, q[::-1], axis=1), r)))

# masks must be bool. if you try int, odd things happen.
mask_q = (q<0.1) * np.ones(len(r))[:,None].astype(bool)
prob[ mask_q ] = 0

d_max_qr = np.transpose(d_max_of_q[:,None] * np.ones(len(r))[None,:])
r_qr = r[:,None] * np.ones(len(q))[None,:]

mask_r = (r_qr > d_max_qr).astype(bool)

prob[ mask_r ] = 0

print('after mask, distribn integrates to {:.3f}'.format(
    trapz(trapz(prob, r, axis=0), q[::-1])))
print('after mask, distribn integrates to {:.3f}'.format(
    trapz(trapz(prob, q[::-1], axis=1), r)))

print('int prob(r|q=1) dr = {:.3f}'.format(trapz(prob[:,0], r)))

##################################################
##################################################
# 2d image contour plot, before masking
f, ax = plt.subplots(figsize=(4,4))

dr = (r[1] - r[0])/2
dq = (q[1] - q[0])/2
extent = [r[0]-dr, r[-1]+dr, q[-1]+dq, q[0]-dq]

im = ax.imshow(prob_pre.T,
        interpolation='bilinear',
        extent=extent,
        aspect=0.5,
        cmap='Blues')

plt.colorbar(im, fraction=0.046, pad=0.04)

ax.set_xlabel('r [arbitrary units]')
ax.set_ylabel('$q = M_2/M_1$')
ax.set_title('$r^2 / (3 d_{\mathrm{max}}^3(q))$, no mask', fontsize='small')

f.tight_layout()

f.savefig('joint_probpremask_r_q.pdf', dpi=250, bbox_inches='tight')


# 2d image contour plot, after masking
f, ax = plt.subplots(figsize=(4,4))

dr = (r[1] - r[0])/2
dq = (q[1] - q[0])/2
extent = [r[0]-dr, r[-1]+dr, q[-1]+dq, q[0]-dq]

im = ax.imshow(prob.T,
        interpolation='bilinear',
        extent=extent,
        aspect=0.5,
        cmap='Blues')

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label(r'$\mathrm{prob}(r,q) \propto r^2$ if '+\
        '$r<d_{\mathrm{max}}(q)$ and $q>0.1$', rotation=270, labelpad=12,
        fontsize='small')

ax.set_xlabel('distance from origin, $r$ [arbitrary units]')
ax.set_ylabel('$q = M_2/M_1$')

f.tight_layout()
f.savefig('joint_prob_r_q.pdf', dpi=250, bbox_inches='tight')

# Marginalized distributions.
plt.close('all')
f, ax = plt.subplots(figsize=(4,4))
q_marg = trapz(prob, r, axis=0)
print('num: int(q_marg dq) = {:.3f}'.format(trapz(q_marg, q[::-1])))
ax.plot(q, q_marg, label='numeric', lw=1, zorder=0)

_q = np.arange(0,1+1e-3,1e-3)
pdf_q_analytic = norm*I_1/9*(1+_q**3)**(3/2)
pdf_q_analytic[_q<0.1] = 0
ax.plot(_q, pdf_q_analytic, label='analytic', lw=2, zorder=-1)
print('analytic: int(q_marg dq) = {:.3f}'.format(trapz(pdf_q_analytic, _q)))

ax.legend(loc='best', fontsize='small')
ax.set(xlabel='$q=M_2/M_1$', ylabel='prob')
f.tight_layout()
f.savefig('q_marg.pdf', dpi=250, bbox_inches='tight')

plt.close('all')
f, ax = plt.subplots(figsize=(4,4))
r_marg = trapz(prob, q[::-1], axis=1)
print('int(r_marg dr) = {:.3f}'.format(trapz(r_marg, r)))
ax.plot(r, r_marg, 'k-')
ax.set(xlabel='r [arbitrary units]', ylabel='prob')
f.tight_layout()
f.savefig('r_marg.pdf', dpi=250, bbox_inches='tight')

