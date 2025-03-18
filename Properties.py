###############################################
#                                             #
#    Plotting Cluster/Committor Properties    #
#                                             #
###############################################

# Import packages
# ---------------

import FuncHist, FuncErrors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm, colors, colorbar
from scipy import stats

rc('text', usetex=True)

# ========== #
# Parameters #
# ========== #

# Plotting properties
# -------------------

ncol  = 'blue'
scol  = 'red'
ncol2 = 'teal'
scol2 = 'maroon'
nlab  = 'Unbiased'
slab  = 'Constructed'
nmark = 'o'
smark = '^'

Nbins  = 10

OP     = 'Nq6'
 # opmean = 200   # Mean to reparameterise histogram test to using OP    
# opdiff = 50    # Maximum allowed distance in OP units from mean

opmean = 300
opdiff = 100  

liq   = 21    # Either a liquid cutoff, or None (if not a solid, count as failure)
sol   = 5000  # Either a solid cutoff, or None (if not a liquid, count as success)


alpha = True  # Use alpha-shape values of volume and surface area  


# Resampling properties
# ---------------------

samples = 50
err     = 'minmax'
tests   = 50
pBbins  = 20

# Properties layout in cluster files
# ----------------------------------


idx   = {'Nq6' : 1, "QclP" : 2,  "QclR"   : 3,  "NQP" : 4, "NQR" : 5, "QclM" : 6, "NQM" : 7, "SA" : 8, "V" : 9, "SAN" : 10, "VN" : 11, "rho" : 12,
         "Ecl" : 13, "EcN" : 14, "Ecl-cl" : 15, "Ecl-clP" : 16, "sol" : 17, "Nq6-P" : 18, "Nq6-2" : 19, "Nq6-2P" : 20, "pB" : 21 }
labs  = {'Nq6' : "$N_{q_6}$", "QclP" : "$Q_6^{cl}$", "QclR" : "$Q_6^{cl\ddag}$", "NQP" : "$N_{q_6}Q_6^{cl}$", "rho" : r'$\rho_N^{*, \,cl}$',
         "NQR" : "$N_{q_6}Q_6^{cl\ddag}$", "QclM" : "$Q_6^{cl\dag}$", "NQM" : "$N_{q_6}Q_6^{cl\dag}$", "SA" : "$A^{*, \,cl}$", "V" : "$V^{*, \,cl}$",
         "Ecl" : "$U^{*, \, cl}$", "EcN" : "$U^{*, \, cl}/N_{q_6}$", "Ecl-Ecl" : "$U^{*, \, cl-cl}$", "Ecl-clP" : "$U^{*, \, cl}_{\mathrm{intra}}/U^{*, \, cl}$",
         "SAN" : "$A^{*, \,cl}/N_{q_6}$", "VN" : "$V^{*, \,cl}/N_{q_6}$", "sol" : "$N_{\mathrm{solid}}$", "Nq6-P" : "$N_{q_6}/N_{\mathrm{solid}}$",
         "Nq6-2" : "$N_{q_6, \, 2}$", "Nq6-2P" : "$N_{q_6, \, 2}/N_{q_6}$", "pB" : "$p_B$"}

# File names and locations
# -------------------------

nfile = "natural_distributions_hull.dat"
sfile = "synthetic_distributions_hull.dat"
bfile = "bignat_distributions_full.dat"
dfile = "bigsynth_distributions_hull.dat"
ndats = "run_natural/final_r"
sdats = "run_synthetic/final_r"
bdats = "run_natural/bigger/final_r"
ddats = "run_synthetic/bigger/final_r"
endat = "_t0.8.txt"
nenf  = "natural_cluster_decomp_energies.dat"
senf  = "synthetic_cluster_decomp_energies.dat"
benf  = "bignat_cluster_decomp_energies.dat"
denf  = "bigsynth_cluster_decomp_energies.dat"
nalph = "natural_alpha.dat"
salph = "synthetic_alpha.dat"
balph = "bignat_alpha.dat"
dalph = "bigsynth_alpha.dat"
nglob = "natural_distributions_global.dat"
sglob = "synthetic_distributions_global.dat"
bglob = None
dglob = "bigsynth_distributions_global.dat"



lowT_nfile = "../lowT/natural_distributions_hull.dat"
lowT_sfile = "../lowT/synthetic_distributions_hull.dat"
lowT_bfile = "../lowT/bignat_distributions_hull.dat"
lowT_dfile = "../lowT/bigsynth_distributions_hull.dat"
lowT_ndats = "../lowT/run_natural/final_r"
lowT_sdats = "../lowT/run_synthetic/final_r"
lowT_bdats = "../lowT/run_natural/big/final_r"
lowT_ddats = "../lowT/run_synthetic/big/final_r"
lowT_endat = "_t0.765.txt"
lowT_nenf  = "../lowT/natural_cluster_decomp_energies.dat"
lowT_senf  = "../lowT/synthetic_cluster_decomp_energies.dat"
lowT_benf  = "../lowT/bignat_cluster_decomp_energies.dat"
lowT_denf  = "../lowT/bigsynth_cluster_decomp_energies.dat"
lowT_nalph = "../lowT/natural_alpha.dat"
lowT_salph = "../lowT/synthetic_alpha.dat"
lowT_balph = "../lowT/bignat_alpha.dat"
lowT_dalph = "../lowT/bigsynth_alpha.dat"
lowT_nglob = "../lowT/natural_distributions_global.dat"
lowT_sglob = "../lowT/synthetic_distributions_global.dat"
lowT_bglob = "../lowT/bignat_distributions_global.dat"
lowT_dglob = "../lowT/bigsynth_distributions_global.dat"

# ======== #
# Get data #
# ======== #

natural,   natural_ends   = FuncHist.get_seed_props(nfile, nglob, nalph, nenf, idx, alpha, ndats, endat, liq, sol, remove=True)
synthetic, synthetic_ends = FuncHist.get_seed_props(sfile, sglob, salph, senf, idx, alpha, sdats, endat, liq, sol, remove=True)
bignat,    bignat_ends    = FuncHist.get_seed_props(bfile, bglob, balph, benf, idx, alpha, bdats, endat, liq, sol, remove=True)
bigsynth,  bigsynth_ends  = FuncHist.get_seed_props(dfile, dglob, dalph, denf, idx, alpha, ddats, endat, liq, sol, remove=True)

natrl = np.vstack((natural, bignat))        ; nat_ends = np.vstack((natural_ends,   bignat_ends))
synth = np.vstack((synthetic, bigsynth))    ; syn_ends = np.vstack((synthetic_ends, bigsynth_ends))

# natrl = natural
# synth = synthetic

n = natrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]
s = synth[np.where(abs(synth[:, idx[OP]] - opmean) <= opdiff)[0], :]

print("# INFO: For ", opmean-opdiff, "<=", OP, "<=", opmean+opdiff, " there are ", len(n), nlab, " files and ", len(s), slab, "files for highT.")

print(np.max(n[:, 1]), np.max(s[:, 1]), np.min(n[:, 1]), np.min(s[:, 1]))


endat = "_t0.765.txt"

lowT_natural,   natural_ends   = FuncHist.get_seed_props(lowT_nfile, lowT_nglob, lowT_nalph, lowT_nenf, idx, alpha, lowT_ndats, endat, 32, sol, remove=True)
lowT_synthetic, synthetic_ends = FuncHist.get_seed_props(lowT_sfile, lowT_sglob, lowT_salph, lowT_senf, idx, alpha, lowT_sdats, endat, 32, sol, remove=True)
lowT_bignat,    bignat_ends    = FuncHist.get_seed_props(lowT_bfile, lowT_bglob, lowT_balph, lowT_benf, idx, alpha, lowT_bdats, endat, 32, sol, remove=True)
lowT_bigsynth,  bigsynth_ends  = FuncHist.get_seed_props(lowT_dfile, lowT_dglob, lowT_dalph, lowT_denf, idx, alpha, lowT_ddats, endat, 32, sol, remove=True)

lowT_natrl = np.vstack((lowT_natural, lowT_bignat))
lowT_synth = np.vstack((lowT_synthetic, lowT_bigsynth)) 

lowTn = lowT_natrl[np.where(abs(lowT_natrl[:, idx[OP]] - 250) <= 125)[0], :]
lowTs = lowT_synth[np.where(abs(lowT_synth[:, idx[OP]] - 250) <= 125)[0], :]


print("# INFO: For ", 250-125, "<=", OP, "<=", 250+125, " there are ", len(lowTn), nlab, " files and ", len(lowTs), slab, "files for lowT.")

print(np.max(lowTn[:, 1]), np.max(lowTs[:, 1]), np.min(lowTn[:, 1]), np.min(lowTs[:, 1]))

FuncHist.make_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, 'VN', op2='SAN', op3='Ecl-clP', savename='Test\
_cryst_shape.png')                                                                                                    

FuncHist.crystallinity_scatter(n, s, ncol, scol, nmark, smark, nlab, slab, idx, labs, savename="Crystallinity_scatt\
er_small_lowT.png")                                                                                                   

FuncHist.committor_scatter(n, s, ncol, scol, nmark, smark, nlab, slab, idx, labs, 'Nq6', op2=None, data3=[bigsynth,\
 'purple', 's', 'big'], data4=None, savename=None)                                                                    

FuncHist.committor_stairs(n, s, ncol, scol, nlab, slab, Nbins, "", idx, n2=[], s2=[], lab2=None, KS=True, savename=\
None)                                                    
