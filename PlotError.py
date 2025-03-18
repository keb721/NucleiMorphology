##############################################
#                                            #
#   Create Committor Error Analysis Plots    #
#                                            #
##############################################

# Import packages
# ---------------

import FuncErrors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats, optimize

rc('text', usetex=True)

# ========== #
# Parameters #
# ========== #

# Plotting properties
# -------------------

ncol     = 'blue'
scol     = 'red'
ncol_all = 'navy'
scol_all = 'crimson'
nlab     = 'Unbiased'
slab     = 'Constructed'
nmark    = 'o'
smark    = '^'

Nbins  = 10

OP     = 'Nq6'
opmean = 175   # Mean to reparameterise histogram test to using OP    
opdiff = 50    # Maximum allowed distance in OP units from mean

liq   = 32    # Either a liquid cutoff, or None (if not a solid, count as failure)
sol   = 3000  # Either a solid cutoff, or None (if not a liquid, count as success)

Ncom  = 10#00                                   # How many different realisations of committor subsampling to perform
tcom  = [10, 25, 50, 100, 150, 200, 250, 299]  # List of number of trajectories to test

scom_n  = [25, 50, 100, 250, 500, 1000, 1065, 1428]    # List of number of seeds to test (natural) - note, must be at least as long as scom_s
scom_s  = [25, 50, 100, 250, 500, 1000, 1065]          # List of number of seeds to test (synthetic)

offset = 100000 # Prevent double labelling

rplc   = False # Sample with replacement when computing error properties

savefigs           = True
sing_comm_err_name = "individual_committor_error.png"
comm_hist_err_name = "committor_histogram_error.png"


# Committor graph properties
# --------------------------
nbox  = dict(color=ncol)
nmean = dict(marker=nmark, markeredgecolor=ncol, markerfacecolor='none')
nmed  = dict(color=ncol)
nwis  = dict(color=ncol)

sbox  = dict(color=scol)
smean = dict(marker=smark, markeredgecolor=scol, markerfacecolor='none')
smed  = dict(color=scol)
swis  = dict(color=scol)


# Properties layout in cluster files - reduced
# ---------------------------------------------


idx   = {'Nq6' : 1, "QclP" : 2,  "QclR"   : 3,  "NQP" : 4, "NQR" : 5, "QclM" : 6, "NQM" : 7}
labs  = {'Nq6' : "$N_{q_6}$", "QclP" : "$Q_6^{cl}$", "QclR" : "$Q_6^{cl\ddag}$", "NQP" : "$N_{q_6}Q_6^{cl}$", 
         "NQR" : "$N_{q_6}Q_6^{cl\ddag}$", "QclM" : "$Q_6^{cl\dag}$", "NQM" : "$N_{q_6}Q_6^{cl\dag}$"}
# File names and locations
# -------------------------

nfile = "natural_distributions_hull.dat"
sfile = "synthetic_distributions_hull.dat"
bfile = "bignat_distributions_hull.dat"
dfile = "bigsynth_distributions_hull.dat"
ndats = "run_natural/final_r"
sdats = "run_synthetic/final_r"
bdats = "run_natural/big/final_r"
ddats = "run_synthetic/big/final_r"
endat = "_t0.765.txt"


# ======== #
# Get data #
# ======== #

natural   = np.genfromtxt(nfile) ; natural[:, 0]   = np.linspace(1,        len(natural[:, 0]),          len(natural[:, 0]))
synthetic = np.genfromtxt(sfile) ; synthetic[:, 0] = np.linspace(1,        len(synthetic[:, 0]),        len(synthetic[:, 0]))
# bignat    = np.genfromtxt(bfile) ; bignat[:, 0]    = np.linspace(offset+1, offset+len(bignat[:, 0]),    len(bignat[:, 0]))
# bigsynth  = np.genfromtxt(dfile) ; bigsynth[:, 0]  = np.linspace(offset+1, offset+len(bigsynth[:, 0]),  len(bigsynth[:, 0]))

# natrl = np.vstack((natural, bignat))        
# synth = np.vstack((synthetic, bigsynth))    
natrl = natural
synth = synthetic


n = natrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]
s = synth[np.where(abs(synth[:, idx[OP]] - opmean) <= opdiff)[0], :]

print("# INFO: For ", opmean-opdiff, "<=", OP, "<=", opmean+opdiff, " there are ", len(n), nlab, " files and ", len(s), slab, "files.")

# ============================ #
# Generate committor estimates #
# ============================ #

assert((liq or sol) is not None), "At least one cutoff is required"

comm = FuncErrors.Committor()

if liq is None:
    comm.pB = FuncErrors.lpB

elif sol is None:
    comm.pB = FuncErrors.spB


n_ends = FuncErrors.end_point_dict(n, ndats, endat, exp_traj=300, alt_loc=bdats, offset=offset)
s_ends = FuncErrors.end_point_dict(s, sdats, endat, exp_traj=300, alt_loc=ddats, offset=offset)
    
# Graph of error in individual committor estimates
# ------------------------------------------------
npB = FuncErrors.d_pB(comm, n_ends, tcom, Ncom, sol, liq, rplc=rplc)
spB = FuncErrors.d_pB(comm, s_ends, tcom, Ncom, sol, liq, rplc=rplc)


fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
fig.subplots_adjust(left=0.135, right=0.9975, bottom=0.135, top=0.9975)

npos = np.linspace(0.8, len(tcom)-0.2, len(tcom))
nbxd = ax.boxplot(npB, showmeans=True, whis=(0, 100), meanprops=nmean, boxprops=nbox, medianprops=nmed, widths=0.3,
          positions=npos, whiskerprops=nwis, capprops=nwis)
spos = np.linspace(1.2, len(tcom)+0.2, len(tcom))
sbxd = ax.boxplot(spB, showmeans=True, whis=(0, 100), meanprops=smean, boxprops=sbox, medianprops=smed, widths=0.3,
          positions=spos, whiskerprops=swis, capprops=swis)

ax.set_ylabel("$\Delta p_B$", size=20)
ax.set_xlabel("Trajectories", size=20)
ax.set_xticks(np.linspace(1, len(tcom), len(tcom)), labels=[str(t) for t in tcom])
ax.tick_params(labelsize=15)
ax.legend([nbxd["boxes"][0], sbxd["boxes"][0]], [nlab, slab], prop={'size':13})

if savefigs:
    plt.savefig(sing_comm_err_name, dpi=450)
plt.show()


# Graph of error in overall histogram test
# ----------------------------------------

nmu_traj, nsig_traj = FuncErrors.prop_hist_traj(comm, n_ends, tcom, Ncom, sol, liq, rplc=rplc)
smu_traj, ssig_traj = FuncErrors.prop_hist_traj(comm, s_ends, tcom, Ncom, sol, liq, rplc=rplc)

nmu_seed, nsig_seed = FuncErrors.prop_hist_seed(comm, n_ends, scom_n, Ncom, sol, liq, rplc=rplc)
smu_seed, ssig_seed = FuncErrors.prop_hist_seed(comm, s_ends, scom_s, Ncom, sol, liq, rplc=rplc)

nmu_all, nsig_all = FuncErrors.prop_hist_all(comm, n_ends, sol, liq)
smu_all, ssig_all = FuncErrors.prop_hist_all(comm, s_ends, sol, liq)


fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6), sharex='col', sharey='row')
fig.subplots_adjust(left=0.065, right=0.9975, bottom=0.07, top=0.9975, wspace=0, hspace=0)

npos = np.linspace(1, len(tcom), len(tcom))
ax[0][0].boxplot(nmu_traj, showmeans=True, whis=(0, 100), meanprops=nmean, boxprops=nbox, medianprops=nmed, widths=0.3,
                 positions=npos, whiskerprops=nwis, capprops=nwis)
nlne, = ax[0][0].plot([0.4, len(tcom)+0.6], [nmu_all, nmu_all], '--', color=ncol_all, alpha=0.5, label=nlab+" (all trajectories)")

spos = np.linspace(1, len(tcom), len(tcom))
ax[0][0].boxplot(smu_traj, showmeans=True, whis=(0, 100), meanprops=smean, boxprops=sbox, medianprops=smed, widths=0.3,
              positions=spos, whiskerprops=swis, capprops=swis)
slne, = ax[0][0].plot([0.4, len(tcom)+0.6], [smu_all, smu_all], '--', color=scol_all, alpha=0.5, label=slab+" (all trajectories)")

nbxd = ax[1][0].boxplot(nsig_traj, showmeans=True, whis=(0, 100), meanprops=nmean, boxprops=nbox, medianprops=nmed, widths=0.3,
           positions=npos, whiskerprops=nwis, capprops=nwis)
ax[1][0].plot([0.4, len(tcom)+0.6], [nsig_all, nsig_all], '--', color=ncol_all, alpha=0.5)
sbxd = ax[1][0].boxplot(ssig_traj, showmeans=True, whis=(0, 100), meanprops=smean, boxprops=sbox, medianprops=smed, widths=0.3,
           positions=spos, whiskerprops=swis, capprops=swis)
ax[1][0].plot([0.4, len(tcom)+0.6], [ssig_all, ssig_all], '--', color=scol_all, alpha=0.5)


npos = np.linspace(1, len(scom_n), len(scom_n))
ax[0][1].boxplot(nmu_seed, showmeans=True, whis=(0, 100), meanprops=nmean, boxprops=nbox, medianprops=nmed, widths=0.3,
              positions=npos, whiskerprops=nwis, capprops=nwis)
nlne, = ax[0][1].plot([0.4, len(scom_n)+0.6], [nmu_all, nmu_all], '--', color=ncol_all, alpha=0.5, label=nlab+" (all trajectories)")

spos = np.linspace(1, len(scom_s), len(scom_s))
ax[0][1].boxplot(smu_seed, showmeans=True, whis=(0, 100), meanprops=smean, boxprops=sbox, medianprops=smed, widths=0.3,
              positions=spos, whiskerprops=swis, capprops=swis)
slne, = ax[0][1].plot([0.4, len(scom_s)+0.6], [smu_all, smu_all], '--', color=scol_all, alpha=0.5, label=slab+" (all trajectories)")

npos[:3] = npos[:3]-0.1 # Optional - offset for legibility
spos[:3] = spos[:3]+0.1

nbxd = ax[1][1].boxplot(nsig_seed, showmeans=True, whis=(0, 100), meanprops=nmean, boxprops=nbox, medianprops=nmed, widths=0.3,
                        positions=npos, whiskerprops=nwis, capprops=nwis)
ax[1][1].plot([0.4, len(scom_n)+0.6], [nsig_all, nsig_all], '--', color=ncol_all, alpha=0.5)

sbxd = ax[1][1].boxplot(ssig_seed, showmeans=True, whis=(0, 100), meanprops=smean, boxprops=sbox, medianprops=smed, widths=0.3,
                        positions=spos, whiskerprops=swis, capprops=swis)
ax[1][1].plot([0.4, len(scom_s)+0.6], [ssig_all, ssig_all], '--', color=scol_all, alpha=0.5)

# Consider adding OP to labels

ax[0][0].set_ylabel("$\mu_h$", size=20)
ax[1][0].set_xlabel("Trajectories", size=20)
ax[1][0].set_ylabel("$\sigma_h$", size=20)
ax[1][1].set_xlabel("Seeds", size=20)
ax[1][0].set_xticks(np.linspace(1, len(tcom), len(tcom)), labels=[str(t) for t in tcom])
ax[1][1].set_xticks(np.linspace(1, len(scom_n), len(scom_n)), labels=[str(s) for s in scom_n])
ax[0][0].tick_params(labelsize=15)
ax[1][0].tick_params(labelsize=15)
ax[1][0].tick_params(labelsize=15)
ax[1][1].tick_params(labelsize=15)
ax[0][1].legend([nbxd["boxes"][0], sbxd["boxes"][0], nlne, slne], [nlab, slab, nlab+" (all trajectories)", slab+" (all trajectories)"], prop={'size':13})

if savefigs:
    plt.savefig(comm_hist_err_name, dpi=450)
plt.show()
