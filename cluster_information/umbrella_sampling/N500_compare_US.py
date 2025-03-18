import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import Counter
from scipy import stats
import FuncUSgraph

rc('text', usetex=True)

natrl_data_fl   = "natural_US_distribution_seed4_N500_r0.4.txt"
natrl_alpha_fl  = "natural_N500_r0.4_alpha.dat"
natrl_energy_fl = "natural_US_N500_r0.4_cluster_energy.txt"

synth_data_fl   = "synth_US_distribution_seed216_N500_r0.4.txt"
synth_alpha_fl  = "synth_N500_r0.4_alpha.dat"
synth_energy_fl = "synth_US_N500_r0.4_cluster_energy.txt"

nseed_data_fl   = "natural_seed.dat"
nseed_alpha_fl  = "natural_seed_alpha.dat"
nseed_energy_fl = "natural_seed_energy.dat"

sseed_data_fl   = "synthetic_seed.dat"
sseed_alpha_fl  = "synthetic_seed_alpha.dat"
sseed_energy_fl = "synthetic_seed_energy.dat"

ncol = 'blue'
scol = 'red'

nc1 = 'slateblue'
nc2 = 'cornflowerblue'
sc1 = 'tomato'
sc2 = 'palevioletred'
nsc = 'teal'
ssc = 'maroon'
# suc = 'lightcoral'

Nbins = 10

idx   = {'Nq6' : 1, "QclP" : 2,  "QclR"   : 3,  "NQP" : 4, "NQR" : 5, "QclM" : 6, "NQM" : 7, "SA" : 8, "V" : 9, "SAN" : 10, "VN" : 11, "rho" : 12,
         "Ecl" : 13, "EcN" : 14, "Ecl-cl" : 15, "Ecl-clP" : 16, "tEn": 17, "pEc" : 18, "Ecl-clPT" : 19, "Ecl-clN" : 20}
labs  = {'Nq6' : "$N_{q_6}$", "QclP" : "$Q_6^{cl}$", "QclR" : "$Q_6^{cl\ddag}$", "NQP" : "$N_{q_6}Q_6^{cl}$", "rho" : r'$\rho_N^{*, \,cl}$',
         "NQR" : "$N_{q_6}Q_6^{cl\ddag}$", "QclM" : "$Q_6^{cl\dag}$", "NQM" : "$N_{q_6}Q_6^{cl\dag}$", "SA" : "$A^{*, \,cl}$", "V" : "$V^{*, \,cl}$",
         "Ecl" : "$U^{*, \, cl}$", "EcN" : "$U^{*, \, cl}/N_{q_6}$", "Ecl-Ecl" : "$U^{*, \, cl-cl}$", "Ecl-clP" : "$U^{*, \, cl-cl}/U^{*, \, cl}$",
         "SAN" : "$A^{*, \,cl}/N_{q_6}$", "VN" : "$V^{*, \,cl}/N_{q_6}$", "tEn" : "$U^*$", "pEc" : "$U^{cl,\,*}/U^*$",
         "Ecl-clPT" : "$U^{cl-cl, \,*}/U^{*}$", "Ecl-clN" : "$E^{*,\,cl-cl}/E^{*,\,cl}$"}

natural = FuncUSgraph.get_data(natrl_data_fl, natrl_energy_fl, natrl_alpha_fl, idx, alpha=True, US=True)
synth   = FuncUSgraph.get_data(synth_data_fl, synth_energy_fl, synth_alpha_fl, idx, alpha=True, US=True)
nsd     = FuncUSgraph.get_data(nseed_data_fl, nseed_energy_fl, nseed_alpha_fl, idx, alpha=True, US=False)
ssd     = FuncUSgraph.get_data(sseed_data_fl, sseed_energy_fl, sseed_alpha_fl, idx, alpha=True, US=False)


nseed = nsd[np.where(nsd[:, 1] < 300)[0], :]
sseed = ssd[np.where(ssd[:, 1] < 300)[0], :]

print(len(nseed), len(sseed))

nlab = 'Initially unbiased'
slab = 'Initially constructed'
nslb = 'Unbiased seeds'
sslb = 'Constructed seeds'
ops  = ['Nq6', 'QclP', 'VN', 'SAN', 'EcN', 'Ecl-clP']

nlist = [natural[10000:, :], ncol, nlab]
slist = [synth[10000::100, :],   scol, slab]
sslst = [sseed,              sc1,  sslb]


FuncUSgraph.equil_test(natural, synth, nc2, ncol, sc1, scol, nlab, slab, Nbins, idx, labs, ops, savename='N500_r0_4_equil.png')

FuncUSgraph.reweighted_plot(natural[10000:, :], ncol, nlab, nseed, nc2, nslb, Nbins, idx, labs, ops, slist, sslst, 'N500_r0_4_reweight_to_nat_end.png', KS=False)
FuncUSgraph.reweighted_plot(synth[10000:, :], scol, slab, nseed, nc2, nslb, Nbins, idx, labs, ops, nlist, sslst, 'N500_r0_4_reweight_to_synth_end.png', KS=False)


FuncUSgraph.reweighted_plot(natural[10000::100, :], ncol, nlab, nseed, nc2, nslb, Nbins, idx, labs, ops, slist, sslst, 'N500_r0_4_reweight_to_nat_end_subsample.png', KS=True)


