import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

colours = ['c', 'navy', 'dodgerblue', 'steelblue', 'royalblue', 'deepskyblue', 'mediumblue',
           'darkslategrey', 'teal', 'slateblue', 'turquoise']


fig, ax = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey='all')
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.9995, wspace=0, hspace=0)

ax[0].plot([1.334, 1.334], [1000, 5], alpha=0.2, color='k', linewidth=2.5)
ax[0].plot([1.432, 1.432], [1000, 5], alpha=0.2, color='k', linewidth=2.5)
ax[0].plot([1.706, 1.706], [1000, 5], alpha=0.2, color='k', linewidth=2.5)
 
ax[1].plot([7,  7],  [1000, 5], alpha=0.2, color='k', linewidth=2.5)
ax[1].plot([8,  8],  [1000, 5], alpha=0.2, color='k', linewidth=2.5)
ax[1].plot([11, 11], [1000, 5], alpha=0.2, color='k', linewidth=2.5)

ax[2].plot([0.4,   0.4],   [1000, 5], alpha=0.2, color='k', linewidth=2.5)
ax[2].plot([0.5,   0.5],   [1000, 5], alpha=0.2, color='k', linewidth=2.5)
ax[2].plot([0.625, 0.625], [1000, 5], alpha=0.2, color='k', linewidth=2.5)

for i, size in enumerate(np.linspace(125, 375, 11, dtype=int)):

     nat_data = glob('unbiased*'+str(size)+'*seed_steps.txt')
     data = np.genfromtxt(nat_data[0])
     #     conn_data = data[::2]
     uncn_data = data[1::2]
     ax[0].plot(uncn_data[:, 0], uncn_data[:, -1], color=colours[i], alpha=0.6, linewidth=1.5)


     nat_data = glob('unbiased*'+str(size)+'*seed_solidconn_steps.txt')
     data = np.genfromtxt(nat_data[0])
     #     conn_data = data[::2]
     uncn_data = data[1::2]
     ax[1].plot(uncn_data[:, 0], uncn_data[:, -1], color=colours[i], alpha=0.6, linewidth=1.5)


     nat_data = glob('unbiased*'+str(size)+'*seed_dotpro_steps.txt')
     data = np.genfromtxt(nat_data[0])
     #     conn_data = data[::2]
     uncn_data = data[1::2]
     ax[2].plot(uncn_data[:, 0], uncn_data[:, -1], color=colours[i], alpha=0.6, linewidth=1.5)

ax[0].text(0.9, 0.925, '(a)', size=20, transform=ax[0].transAxes)
ax[1].text(0.9, 0.925, '(b)', size=20, transform=ax[1].transAxes)
ax[2].text(0.9, 0.925, '(c)', size=20, transform=ax[2].transAxes)

ax[0].set_ylabel('$N_{q_6}$', size=20)
ax[0].set_xlabel('$\sigma_N/\sigma$', size=20)
ax[1].set_xlabel('$c_s$', size=20)
ax[2].set_xlabel('$c_q$', size=20)


ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
ax[2].tick_params(labelsize=15)


plt.savefig('stepping_all_colours.pdf', dpi=600)

plt.show()
