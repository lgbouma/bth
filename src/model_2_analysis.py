'''
run after running model_2.py

(processes those results)
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('model2_results.csv')

f,ax = plt.subplots()
ax.scatter(df['seed'],df['X_Γ'])
f.savefig('tests/seed_num_vs_X_Gamma.pdf')

plt.close('all')
f,ax = plt.subplots()
ax.scatter(df['m_lim'],df['X_Γ'])
f.savefig('tests/m_lim_vs_X_Gamma.pdf')

plt.close('all')
f,ax = plt.subplots()
ax.scatter(df['N_s'],df['X_Γ'])
f.savefig('tests/N_s_vs_X_Gamma.pdf')

plt.close('all')
f,ax = plt.subplots()
ax.scatter(df['N_d'],df['X_Γ'])
f.savefig('tests/N_d_vs_X_Gamma.pdf')

plt.close('all')
f,ax = plt.subplots()
ax.scatter(df['β'],df['X_Γ'])
f.savefig('tests/beta_vs_X_Gamma.pdf')

plt.close('all')
import seaborn as sns
f,ax = plt.subplots()
sns.distplot(df['X_Γ'], bins=15, kde=True, norm_hist=True, ax=ax)
pd.set_option('precision',3)
txt = repr(df['X_Γ'].describe())
ax.text(
        0.96, 0.96, txt,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes, fontsize='small'
        )
ax.set_ylabel('prob')
f.savefig('tests/dist_X_Gamma.pdf')


plt.close('all')
import seaborn as sns
f,ax = plt.subplots()
sns.distplot(df['X_Γ_upperlimit'], bins=15, kde=True, norm_hist=True, ax=ax)
pd.set_option('precision',3)
txt = repr(df['X_Γ_upperlimit'].describe())
ax.text(
        0.96, 0.96, txt,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes, fontsize='small'
        )
ax.set_ylabel('prob')
f.savefig('tests/dist_X_Gamma_upperlimit.pdf')


pd.set_option('precision',5)
print(df.describe())
