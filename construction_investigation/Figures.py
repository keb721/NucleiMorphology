###############################################
#                                             #
#    Plotting Cluster/Committor Properties    #
#                                             #
###############################################

# Import packages
# ---------------

import FuncHist
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

opmean = 250
opdiff = 125

liq   = 32    # Either a liquid cutoff, or None (if not a solid, count as failure)
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
         "Ecl" : "$U^{*, \, cl}$", "EcN" : "$U^{*, \, cl}/N_{q_6}$", "Ecl-Ecl" : "$U^{*, \, cl}_{\mathrm{intra}}$", "Ecl-clP" : "$U^{*, \, cl}_{\mathrm{intra}}\,/U^{*, \, cl}$",
         "SAN" : "$A^{*, \,cl}/N_{q_6}$", "VN" : "$V^{*, \,cl}/N_{q_6}$", "sol" : "$N_{\mathrm{solid}}$", "Nq6-P" : "$N_{q_6}/N_{\mathrm{solid}}$",
         "Nq6-2" : "$N_{q_6, \, 2}$", "Nq6-2P" : "$N_{q_6, \, 2}/N_{q_6}$", "pB" : "$p_B$"}

# File names and locations
# -------------------------

base   = "_distributions_hull.dat"
energy = "_cluster_decomp_energies.dat"
alphaf = "_alpha.dat"
globl  = "_distributions_global.dat"


# ======== #
# Get data #
# ======== #

unbiased   = FuncHist.get_seed_props("unbiased"+base, "unbiased"+globl, "unbiased"+alphaf, "unbiased"+energy, idx, alpha, "unbiased_committor.dat")
bigunb    = FuncHist.get_seed_props("bigunb"+base, "bigunb"+globl, "bigunb"+alphaf, "bigunb"+energy, idx, alpha, "bigunb_committor.dat")

print(len(unbiased))
print(len(bigunb))


remnat = np.genfromtxt('lowT_unbiased_nosubset.dat', dtype=int)
remnat = remnat - 1
print(remnat, len(remnat))
unbiased = np.delete(unbiased, remnat, axis=0)

rembnt = np.genfromtxt('lowT_bigunb_nosubset.dat', dtype=int)
rembnt = rembnt - 1
print(rembnt, len(rembnt))

bigunb = np.delete(bigunb, rembnt, axis=0)

print(len(unbiased))
print(len(bigunb))


natrl = np.vstack((unbiased, bigunb))     

n = natrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

print(len(natrl), len(n))

# Neighbours

neighs  = {}
solids  = {}
dotpros = {}

neighs['432'] = n
solids['8']   = n
dotpros['5']  = n

for ndist in ['334', '706']:
    nunbiased   = FuncHist.get_seed_props("unbiased_distributions_cutoff_1_{}.txt".format(ndist), None,  "proximal_unbiased_1{}_alpha.dat".format(ndist), "proximal_unbiased_1{}_cluster_decomp_energies.dat".format(ndist), idx, alpha, "unbiased_committor.dat")

    nbigunb   = FuncHist.get_seed_props("bigunb_distributions_cutoff_1_{}.txt".format(ndist), None,  "proximal_bigunb_1{}_alpha.dat".format(ndist), "proximal_bigunb_1{}_cluster_decomp_energies.dat".format(ndist), idx, alpha, "bigunb_committor.dat")

    nunbiased = np.delete(nunbiased, remnat, axis=0)
    nbigunb  = np.delete(nbigunb, rembnt, axis=0)
    
    nnatrl = np.vstack((nunbiased, nbigunb))     
    
    nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

    neighs[ndist] = nn

for solid in ['7', '11']:
    nunbiased   = FuncHist.get_seed_props("unbiased_distributions_solid_{}.txt".format(solid), None,  "proximal_unbiased_proximal_alpha_solid{}.dat".format(solid), "proximal_unbiased_solid{}_cluster_decomp_energies.dat".format(solid), idx, alpha, "unbiased_committor.dat")

    nbigunb   = FuncHist.get_seed_props("bigunb_distributions_solid_{}.txt".format(solid), None,  "proximal_bigunb_proximal_alpha_solid{}.dat".format(solid), "proximal_bigunb_solid{}_cluster_decomp_energies.dat".format(solid), idx, alpha, "bigunb_committor.dat")

    nunbiased = np.delete(nunbiased, remnat, axis=0)
    nbigunb  = np.delete(nbigunb, rembnt, axis=0)
    
    nnatrl = np.vstack((nunbiased, nbigunb))     
    
    nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

    solids[solid] = nn

    
for dotpro in ['4', '625']:
    nunbiased   = FuncHist.get_seed_props("unbiased_distributions_dotpro_0{}.txt".format(dotpro), None,  "proximal_unbiased_proximal_alpha_dotpro0{}.dat".format(dotpro), "proximal_unbiased_dotpro0{}_cluster_decomp_energies.dat".format(dotpro), idx, alpha, "unbiased_committor.dat")

    nbigunb   = FuncHist.get_seed_props("bigunb_distributions_dotpro_0{}.txt".format(dotpro), None,  "proximal_bigunb_proximal_alpha_dotpro0{}.dat".format(dotpro), "proximal_bigunb_dotpro0{}_cluster_decomp_energies.dat".format(dotpro), idx, alpha, "bigunb_committor.dat")
    
    nunbiased = np.delete(nunbiased, remnat, axis=0)
    nbigunb  = np.delete(nbigunb, rembnt, axis=0)
    
    nnatrl = np.vstack((nunbiased, nbigunb))     
    
    nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

    dotpros[dotpro] = nn


data    = {'neighs': neighs, 'solids': solids, 'dotpro': dotpros}
cutoffs = {'neighs': ['334', '432', '706'],
           'solids': ['7', '8', '11'],
           'dotpro': ['4', '5', '625']}
colours = {'neighs': {'334': 'limegreen', '432': 'mediumseagreen',  '706': 'darkgreen'},
           'solids': {'7'  : 'gold',      '8'  : 'orange',          '11' : 'chocolate'},
           'dotpro': {'4'  : 'violet',    '5'  : 'mediumvioletred', '625': 'purple'}}
label   = {'neighs': r'$\sigma_N = 1.{}~\sigma$', 'solids': r'$c_s = {}$', 'dotpro': r'$c_q = 0.{}$'}


fig, ax = plt.subplots(3, 6, figsize=(12.8, 8), sharey='row', sharex='col')
ops = ['Nq6', 'QclP', 'VN', 'SAN', 'EcN', 'Ecl-clP']
fig.subplots_adjust(left=0.075, right=0.9975, bottom=0.1, top=0.9975, wspace=0, hspace=0)


for j, cutoff in enumerate(['neighs', 'solids', 'dotpro']):
    for i, op in enumerate(ops):
        mns = [] ; mxs = []
        for val in cutoffs[cutoff]:
            mxs.append(np.max(data[cutoff][val][:, idx[op]]))
            mns.append(np.min(data[cutoff][val][:, idx[op]]))
         
        mnop = np.min(mns)
        mxop = np.max(mxs)

        bns = np.linspace(mnop, mxop, Nbins+1)

        for k, val in enumerate(cutoffs[cutoff]):
            ni, bns = np.histogram(data[cutoff][val][:, idx[op]], bns)
    
            ax[j][i].stairs(ni, edges=bns, color=colours[cutoff][val], alpha=0.8, label=label[cutoff].format(val), linewidth=2.5)        

            ax[j][i].set_xlabel(labs[op], size=20)
            ax[j][i].tick_params(labelsize=15)
            
        ax[j][0].set_ylabel("Count", size=20)
        # ax[j][-1].legend(loc='upper left', prop={'size':13}) #, bbox_to_anchor=(1, 0.5))

ax[1][-1].legend(loc='upper left', prop={'size':13}) #, bbox_to_anchor=(1, 0.5))
ax[2][-1].legend(loc='upper left', prop={'size':13}) #, bbox_to_anchor=(1, 0.5))
ax[0][-1].legend(loc='upper left', prop={'size':13}, bbox_to_anchor=(0.025, 0.875))

ax[0][0].text(0.825, 0.9, '(a)', size=20, transform=ax[0][0].transAxes)
ax[0][1].text(0.825, 0.9, '(b)', size=20, transform=ax[0][1].transAxes)
ax[0][2].text(0.825, 0.9, '(c)', size=20, transform=ax[0][2].transAxes)
ax[0][3].text(0.825, 0.9, '(d)', size=20, transform=ax[0][3].transAxes)
ax[0][4].text(0.825, 0.9, '(e)', size=20, transform=ax[0][4].transAxes)
ax[0][5].text(0.825, 0.9, '(f)', size=20, transform=ax[0][5].transAxes)
                     
ax[1][0].text(0.825, 0.9, '(g)', size=20, transform=ax[1][0].transAxes)
ax[1][1].text(0.825, 0.9, '(h)', size=20, transform=ax[1][1].transAxes)
ax[1][2].text(0.85, 0.9, '(i)', size=20, transform=ax[1][2].transAxes)
ax[1][3].text(0.85, 0.9, '(j)', size=20, transform=ax[1][3].transAxes)
ax[1][4].text(0.825, 0.9, '(k)', size=20, transform=ax[1][4].transAxes)
ax[1][5].text(0.85, 0.9, '(l)', size=20, transform=ax[1][5].transAxes)

ax[2][0].set_ylim([0, 6500])

ax[2][0].text(0.775, 0.9, '(m)', size=20, transform=ax[2][0].transAxes)
ax[2][1].text(0.8, 0.9, '(n)', size=20, transform=ax[2][1].transAxes)
ax[2][2].text(0.8, 0.9, '(o)', size=20, transform=ax[2][2].transAxes)
ax[2][3].text(0.8, 0.9, '(p)', size=20, transform=ax[2][3].transAxes)
ax[2][4].text(0.8, 0.9, '(q)', size=20, transform=ax[2][4].transAxes)
ax[2][5].text(0.85, 0.9, '(r)', size=20, transform=ax[2][5].transAxes)

        
plt.savefig('Fig6.pdf', dpi=600)

plt.show()


# Here determine the limits of OPs to consider    

props = {'Nq6': {'neighs': {'334': [235, 115, 231], '706': [225, 115, 231]},
                 'solids': {'7'  : [425, 125, 251], '11' : [100, 75, 151]},
                 'dotpro': {'4'  : [750, 250, 501], '625': [125, 100, 201]}},
         'NQP': {'neighs': {'334': [60, 39, 79],    '706': [28, 22, 45]},
                 'solids': {'7'  : [60, 39, 79],    '11' : [40, 30, 61]},
                 'dotpro': {'4'  : [60, 35, 71],    '625': [45, 35, 71]}}}


defprops = {'Nq6': [250, 125, 251], 'NQP': [60, 39, 79]}

for OP in ['Nq6', 'NQP']:

    daat = data['solids']['8']
    
    print(len(np.where(abs(daat[:, idx[OP]] - defprops[OP][0]) <= defprops[OP][1])[0]))
    
    for i, cutoff in enumerate(['neighs', 'solids', 'dotpro']):
        for val in props[OP][cutoff].keys():
            print(OP,  cutoff, val)
            daaat = data[cutoff][val]
            print(len(np.where(abs(daaat[:, idx[OP]] - props[OP][cutoff][val][0]) <= props[OP][cutoff][val][1])[0]))


             




samples = 50
err     = 'minmax'
tests   = 50
pBbins  = 20


col2 = {'neighs': {'334': 'lime',   '706': 'forestgreen'},
        'solids': {'7'  : 'y',      '11' : 'sienna'},
        'dotpro': {'4'  : 'orchid', '625': 'm'}}


for OP in ['Nq6', 'NQP']:
    
    comms = {'neighs': {'334': None, '432': None, '706': None},
             'solids': {'7'  : None, '8'  : None, '11' : None},
             'dotpro': {'4'  : None, '5'  : None, '625': None}}

    edges = {'neighs': {'334': None, '432': None, '706': None},
             'solids': {'7'  : None, '8'  : None, '11' : None},
             'dotpro': {'4'  : None, '5'  : None, '625': None}}

    
    comm_dict, def_edges = FuncHist.plot_flatten(n, OP, defprops[OP][0], defprops[OP][1], defprops[OP][2], pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
    
    comms['neighs']['432'] = comm_dict
    comms['solids']['8']   = comm_dict
    comms['dotpro']['5']   = comm_dict
    
    edges['neighs']['432'] = def_edges
    edges['solids']['8']   = def_edges
    edges['dotpro']['5']   = def_edges


    for i, cutoff in enumerate(['neighs', 'solids', 'dotpro']):
    # for cutoff in ['dotpro']:
        for val in props[OP][cutoff].keys():
            # test, ax = plt.subplots(1,1)
            # ax.hist(data[cutoff][val][:, idx[OP]], bins=np.linspace(props[OP][cutoff][val][0]-props[OP][cutoff][val][1], props[OP][cutoff][val][0]+props[OP][cutoff][val][1], props[OP][cutoff][val][2]))
            # test.show()
                    
            test_comm, OP_edges = FuncHist.plot_flatten(data[cutoff][val], OP, props[OP][cutoff][val][0], props[OP][cutoff][val][1], props[OP][cutoff][val][2], pBbins, colours[cutoff][val], col2[cutoff][val], label[cutoff].format(val), idx, labs, samples, tests, err)

            comms[cutoff][val] = test_comm
            edges[cutoff][val] = OP_edges
            
    
    fig, ax = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey='row')
    fig.subplots_adjust(left=0.07, right=0.9975, bottom=0.14, top=0.9975, wspace=0, hspace=0)

    figs, axs = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey='row')
    figs.subplots_adjust(left=0.07, right=0.9975, bottom=0.14, top=0.9975, wspace=0, hspace=0)

    centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(comm_dict, def_edges, OP, pBbins, samples=samples)

    print('default', OP, width)
       
    # for i, t in enumerate([['neighs', '432'], ['solids', '8'], ['dotpro', '5']]):
    #     cutoff = t[0] ; val = t[1]
    #     centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(comms[cutoff][val], edges[cutoff][val], OP, pBbins, samples=samples)
    #     ax[i].plot(centres, mean, color=colours[cutoff][val], alpha=0.8, label=label[cutoff].format(val), linewidth=2.5)        
    #     ax[i].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color=colours[cutoff][val], alpha=0.2)

    #     axs[i].plot(centres, means, color=colours[cutoff][val], alpha=0.8, label=label[cutoff].format(val), linewidth=2.5)        
    #     axs[i].fill_between(centres, std_err[0,:], std_err[1,:],  color=colours[cutoff][val], alpha=0.2)

    
    for i, cutoff in enumerate(['neighs', 'solids', 'dotpro']):
        min_centre = 500
        max_centre = 0
    # for cutoff in ['dotpro']:     
        for val in comms[cutoff].keys():
            
            centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(comms[cutoff][val], edges[cutoff][val], OP, pBbins, samples=samples)
        
            print(cutoff, val, OP, width)

            min_centre = min_centre if min_centre < np.min(centres) else np.min(centres)
            max_centre = max_centre if max_centre > np.max(centres) else np.max(centres)
            
            ax[i].plot(centres, mean, color=colours[cutoff][val], alpha=0.8, label=label[cutoff].format(val), linewidth=2.5)        
            ax[i].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color=colours[cutoff][val], alpha=0.2)

            axs[i].plot(centres, means, color=colours[cutoff][val], alpha=0.8, label=label[cutoff].format(val), linewidth=2.5)        
            axs[i].fill_between(centres, std_err[0,:], std_err[1,:],  color=colours[cutoff][val], alpha=0.2)

        ax[i].plot([min_centre, max_centre], [0.5, 0.5], color='k', linestyle='dashed', linewidth=2.5, alpha=0.6)
        axs[i].plot([min_centre, max_centre], [0.5, 0.5], color='k', linestyle='dashed', linewidth=2.5, alpha=0.6)

    for i in range(3):
        
        ax[i].tick_params(labelsize=15)
        ax[i].legend(prop={'size':13}, loc='lower right')
        ax[i].set_xlabel(labs[OP]+" window centre", size=20)

        axs[i].tick_params(labelsize=15)
        axs[i].legend(prop={'size':13}, loc='lower right')
        axs[i].set_xlabel(labs[OP]+" window centre", size=20)


    ax[0].text(0.025, 0.95, '(a)', size=20, transform=ax[0].transAxes)
    ax[1].text(0.025, 0.95, '(b)', size=20, transform=ax[1].transAxes)
    ax[2].text(0.025, 0.95, '(c)', size=20, transform=ax[2].transAxes)

        
        
    ax[0].set_ylabel("$\mu_h$", size=20)
    axs[0].set_ylabel("$\mu_h$", size=20)

    fig.savefig("minmax_committor_cutoffs_{}.pdf".format(OP), dpi=600)
    figs.savefig("std_committor_cutoffs_{}.pdf".format(OP), dpi=600)
    
    plt.show()

    


# File names and locations
# -------------------------

base   = "_distributions_hull.dat"
energy = "_cluster_decomp_energies.dat"
alphaf = "_alpha.dat"
globl  = "_distributions_global.dat"

cutoffs = {'solid' : ['7', '11'], 'dotpro' : ['04', '0625']}

data = {'unbiased_proximal'  : {'neighs' : {}, 'solid' : {}, 'dotpro' : {}},
        'unbiased_oriented'    : {'neighs' : {}, 'solid' : {}, 'dotpro' : {}},
        'constructed_proximal': {'neighs' : {}, 'solid' : {}, 'dotpro' : {}},
        'constructed_oriented'  : {'neighs' : {}, 'solid' : {}, 'dotpro' : {}}}

big   = {'unbiased': 'bigunb', 'constructed': 'bigconstr'}

# ======== #
# Get data #
# ======== #

for tp in data.keys():
    seedtp = tp.split('_')[0]
    clstp  = tp.split('_')[1]
    
    glbl   = None if clstp == 'oriented' else '{}/{}'.format(tp, seedtp)+globl

    prefix = tp+'/'+seedtp
    
    bgtp   = big[seedtp]+'_'+clstp
    bgglbl = None if clstp == 'oriented' else '{}/{}'.format(bgtp, big[seedtp])+globl

    bgpref = bgtp+'/'+big[seedtp]


    dat   = FuncHist.get_seed_props(prefix+base, glbl, prefix+alphaf, prefix+energy, idx, alpha, seedtp+"_committor.dat")
    bgdat = FuncHist.get_seed_props(bgpref+base, bgglbl, bgpref+alphaf, bgpref+energy, idx, alpha, big[seedtp]+"_committor.dat")

    remnat = np.genfromtxt('lowT_{}_nosubset.dat'.format(seedtp), dtype=int)
    remnat = remnat - 1

    dt = np.delete(dat, remnat, axis=0)

    rembnt = np.genfromtxt('lowT_{}_nosubset.dat'.format(big[seedtp]), dtype=int)
    rembnt = rembnt - 1

    bigdt = np.delete(bgdat, rembnt, axis=0)

    natrl = np.vstack((dt, bigdt))     

    n = natrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

    data[tp]['neighs']['432'] = n
    data[tp]['solid']['8']   = n
    data[tp]['dotpro']['05']  = n

    
    for ndist in ['334', '706']:
        ddat   = FuncHist.get_seed_props(prefix+"_distributions_cutoff_1_{}.txt".format(ndist), None,  tp+'/'+clstp+'_'+seedtp+"_1{}_alpha.dat".format(ndist), tp+'/'+clstp+'_'+seedtp+"_1{}_cluster_decomp_energies.dat".format(ndist), idx, alpha, seedtp+"_committor.dat")

        dbigdat = FuncHist.get_seed_props(bgpref+"_distributions_cutoff_1_{}.txt".format(ndist), None,  bgtp+'/'+clstp+'_'+big[seedtp]+"_1{}_alpha.dat".format(ndist), bgtp+'/'+clstp+'_'+big[seedtp]+"_1{}_cluster_decomp_energies.dat".format(ndist), idx, alpha, big[seedtp]+"_committor.dat")

        ddata     = np.delete(ddat, remnat, axis=0)
        dbigdata  = np.delete(dbigdat, rembnt, axis=0)
    
        nnatrl = np.vstack((ddata, dbigdata))     
    
        nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

        data[tp]['neighs'][ndist] = nn

    for cutoff in ['solid', 'dotpro']:
        for val in cutoffs[cutoff]:
            ddat   = FuncHist.get_seed_props(prefix+"_distributions_{}_{}.txt".format(cutoff, val), None,  tp+'/'+clstp+'_'+seedtp+'_'+clstp+"_alpha_{}{}.dat".format(cutoff, val), tp+'/'+clstp+'_'+seedtp+"_cluster_decomp_energies_{}{}.dat".format(cutoff, val), idx, alpha, seedtp+"_committor.dat")

            dbigdat   = FuncHist.get_seed_props(bgpref+"_distributions_{}_{}.txt".format(cutoff, val), None,  bgtp+'/'+clstp+'_'+big[seedtp]+'_'+clstp+"_alpha_{}{}.dat".format(cutoff, val), bgtp+'/'+clstp+'_'+big[seedtp]+"_cluster_decomp_energies_{}{}.dat".format(cutoff, val), idx, alpha, big[seedtp]+"_committor.dat")

            ddata = np.delete(ddat, remnat, axis=0)
            dbigdata  = np.delete(dbigdat, rembnt, axis=0)
    
            nnatrl = np.vstack((ddata, dbigdata))     
        
            nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]
            
            data[tp][cutoff][val] = nn

        
cutoffs = {'neighs': ['334', '432', '706'],
           'solid' : ['7', '8', '11'],
           'dotpro': ['04', '05', '0625']}
#colours = {'neighs': {'334': 'limegreen', '432': 'mediumseagreen',  '706': 'darkgreen'},
#           'solids': {'7'  : 'gold',      '8'  : 'orange',          '11' : 'chocolate'},
#           'dotpro': {'4'  : 'violet',    '5'  : 'mediumvioletred', '625': 'purple'}}
colours = {'unbiased_oriented'  : 'blue', 'constructed_oriented'  : 'red',
           'unbiased_proximal': 'teal', 'constructed_proximal': 'maroon'} 

label   = {'neighs': r'$\sigma_N = 1.{}~\sigma$', 'solid': r'$c_s = {}$', 'dotpro': r'$c_q = 0.{}$',
           'unbiased_oriented': r'Unbiased, $^\mathrm{ort}$',   'constructed_oriented': r'Constructed, $^\mathrm{ort}$',
           'unbiased_proximal': r'Unbiased, $^\mathrm{prx}$', 'constructed_proximal': r'Constructed, $^\mathrm{prx}$'}


    

#####################
#                   #
#     SI Figures    #
#                   #
#####################


# File names and locations
# -------------------------

base   = "_distributions_hull.dat"
energy = "_cluster_decomp_energies.dat"
alphaf = "_alpha.dat"
globl  = "_distributions_global.dat"

cutoffs = {'solid' : ['7', '11'], 'dotpro' : ['04', '0625']}

data = {'unbiased_proximal'  : {'neighs' : {}, 'solid' : {}, 'dotpro' : {}},
        'unbiased_oriented'    : {'neighs' : {}, 'solid' : {}, 'dotpro' : {}},
        'constructed_proximal': {'neighs' : {}, 'solid' : {}, 'dotpro' : {}},
        'constructed_oriented'  : {'neighs' : {}, 'solid' : {}, 'dotpro' : {}}}

big   = {'unbiased': 'bigunb', 'constructed': 'bigconstr'}

# ======== #
# Get data #
# ======== #

for tp in data.keys():
    seedtp = tp.split('_')[0]
    clstp  = tp.split('_')[1]
    
    glbl   = None if clstp == 'oriented' else '{}/{}'.format(tp, seedtp)+globl

    prefix = tp+'/'+seedtp
    
    bgtp   = big[seedtp]+'_'+clstp
    bgglbl = None if clstp == 'oriented' else '{}/{}'.format(bgtp, big[seedtp])+globl

    bgpref = bgtp+'/'+big[seedtp]


    dat   = FuncHist.get_seed_props(prefix+base, glbl, prefix+alphaf, prefix+energy, idx, alpha, seedtp+"_committor.dat")
    bgdat = FuncHist.get_seed_props(bgpref+base, bgglbl, bgpref+alphaf, bgpref+energy, idx, alpha, big[seedtp]+"_committor.dat")

    remnat = np.genfromtxt('lowT_{}_nosubset.dat'.format(seedtp), dtype=int)
    remnat = remnat - 1

    dt = np.delete(dat, remnat, axis=0)

    rembnt = np.genfromtxt('lowT_{}_nosubset.dat'.format(big[seedtp]), dtype=int)
    rembnt = rembnt - 1

    bigdt = np.delete(bgdat, rembnt, axis=0)

    natrl = np.vstack((dt, bigdt))     

    n = natrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

    data[tp]['neighs']['432'] = n
    data[tp]['solid']['8']   = n
    data[tp]['dotpro']['05']  = n

    
    for ndist in ['334', '706']:
        ddat   = FuncHist.get_seed_props(prefix+"_distributions_cutoff_1_{}.txt".format(ndist), None,  tp+'/'+clstp+'_'+seedtp+"_1{}_alpha.dat".format(ndist), tp+'/'+clstp+'_'+seedtp+"_1{}_cluster_decomp_energies.dat".format(ndist), idx, alpha, seedtp+"_committor.dat")

        dbigdat = FuncHist.get_seed_props(bgpref+"_distributions_cutoff_1_{}.txt".format(ndist), None,  bgtp+'/'+clstp+'_'+big[seedtp]+"_1{}_alpha.dat".format(ndist), bgtp+'/'+clstp+'_'+big[seedtp]+"_1{}_cluster_decomp_energies.dat".format(ndist), idx, alpha, big[seedtp]+"_committor.dat")

        ddata     = np.delete(ddat, remnat, axis=0)
        dbigdata  = np.delete(dbigdat, rembnt, axis=0)
    
        nnatrl = np.vstack((ddata, dbigdata))     
    
        nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]

        data[tp]['neighs'][ndist] = nn

    for cutoff in ['solid', 'dotpro']:
        for val in cutoffs[cutoff]:
            ddat   = FuncHist.get_seed_props(prefix+"_distributions_{}_{}.txt".format(cutoff, val), None,  tp+'/'+clstp+'_'+seedtp+'_'+clstp+"_alpha_{}{}.dat".format(cutoff, val), tp+'/'+clstp+'_'+seedtp+"_cluster_decomp_energies_{}{}.dat".format(cutoff, val), idx, alpha, seedtp+"_committor.dat")

            dbigdat   = FuncHist.get_seed_props(bgpref+"_distributions_{}_{}.txt".format(cutoff, val), None,  bgtp+'/'+clstp+'_'+big[seedtp]+'_'+clstp+"_alpha_{}{}.dat".format(cutoff, val), bgtp+'/'+clstp+'_'+big[seedtp]+"_cluster_decomp_energies_{}{}.dat".format(cutoff, val), idx, alpha, big[seedtp]+"_committor.dat")

            ddata = np.delete(ddat, remnat, axis=0)
            dbigdata  = np.delete(dbigdat, rembnt, axis=0)
    
            nnatrl = np.vstack((ddata, dbigdata))     
        
            nn = nnatrl[np.where(abs(natrl[:, idx[OP]] - opmean) <= opdiff)[0], :]
            
            data[tp][cutoff][val] = nn

        
cutoffs = {'neighs': ['334', '432', '706'],
           'solid' : ['7', '8', '11'],
           'dotpro': ['04', '05', '0625']}
#colours = {'neighs': {'334': 'limegreen', '432': 'mediumseagreen',  '706': 'darkgreen'},
#           'solids': {'7'  : 'gold',      '8'  : 'orange',          '11' : 'chocolate'},
#           'dotpro': {'4'  : 'violet',    '5'  : 'mediumvioletred', '625': 'purple'}}
colours = {'unbiased_oriented'  : 'blue', 'constructed_oriented'  : 'red',
           'unbiased_proximal': 'teal', 'constructed_proximal': 'maroon'} 

label   = {'neighs': r'$\sigma_N = 1.{}~\sigma$', 'solid': r'$c_s = {}$', 'dotpro': r'$c_q = 0.{}$',
           'unbiased_oriented': r'Unbiased, $^\mathrm{ort}$',   'constructed_oriented': r'Constructed, $^\mathrm{ort}$',
           'unbiased_proximal': r'Unbiased, $^\mathrm{prx}$', 'constructed_proximal': r'Constructed, $^\mathrm{prx}$'}


#for cutoff in cutoffs.keys():
for cutoff in ['solid', 'dotpro']:
    fig, ax = plt.subplots(3, 6, figsize=(12.8, 9.6), sharey='row', sharex='col')
    ops = ['Nq6', 'QclP', 'VN', 'SAN', 'EcN', 'Ecl-clP']
    fig.subplots_adjust(left=0.075, right=0.9975, bottom=0.1, top=0.9975, wspace=0, hspace=0)

    for j, val in enumerate(cutoffs[cutoff]):
        for i, op in enumerate(ops):
            mns = [] ; mxs = []
            for tp in data.keys(): 
                mxs.append(np.max(data[tp][cutoff][val][:, idx[op]]))
                mns.append(np.min(data[tp][cutoff][val][:, idx[op]]))
         
            mnop = np.min(mns)
            mxop = np.max(mxs)
            
            bns = np.linspace(mnop, mxop, Nbins+1)
                        
            for tp in data.keys():
                d = data[tp][cutoff][val]

#                print(len(d), tp, cutoff, val)
                ni, bns = np.histogram(d[:, idx[op]], bns)
                ax[j][i].stairs(ni, edges=bns, color=colours[tp], alpha=0.8, label=label[tp], linewidth=2.5)        

                ax[j][i].set_xlabel(labs[op], size=20)
                ax[j][i].tick_params(labelsize=15)

                
        if cutoff=='dotpro':
            ax[j][0].set_ylabel("Count, "+label[cutoff].format(val[1:]), size=20)
        else:
            ax[j][0].set_ylabel("Count, "+label[cutoff].format(val), size=20)
            
    if cutoff=='neighs' or cutoff=='dotpro':
        ax[1][0].set_ylim([0, 4250])
    ax[1][0].legend(loc='center', prop={'size':13}, bbox_to_anchor=(0.5, 0.6))
    # else:
    #     ax[-1][-1].legend(loc='upper left', prop={'size':13}) #, bbox_to_anchor=(1, 0.5))

    ax[0][0].text(0.825, 0.9, '(a)', size=20, transform=ax[0][0].transAxes)
    ax[0][1].text(0.825, 0.9, '(b)', size=20, transform=ax[0][1].transAxes)
    ax[0][2].text(0.825, 0.9, '(c)', size=20, transform=ax[0][2].transAxes)
    ax[0][3].text(0.825, 0.9, '(d)', size=20, transform=ax[0][3].transAxes)
    if cutoff=='neighs':
        ax[0][4].text(0.825, 0.9, '(e)', size=20, transform=ax[0][4].transAxes)
    else:
        ax[0][4].text(0.025, 0.9, '(e)', size=20, transform=ax[0][4].transAxes)
    ax[0][5].text(0.825, 0.9, '(f)', size=20, transform=ax[0][5].transAxes)

    ax[1][0].text(0.825, 0.9, '(g)', size=20, transform=ax[1][0].transAxes)
    ax[1][1].text(0.825, 0.9, '(h)', size=20, transform=ax[1][1].transAxes)
    ax[1][2].text(0.85, 0.9, '(i)', size=20, transform=ax[1][2].transAxes)
    ax[1][3].text(0.85, 0.9, '(j)', size=20, transform=ax[1][3].transAxes)
    ax[1][4].text(0.825, 0.9, '(k)', size=20, transform=ax[1][4].transAxes)
    ax[1][5].text(0.85, 0.9, '(l)', size=20, transform=ax[1][5].transAxes)

    #ax[2][0].set_ylim([0, 6500])

    ax[2][0].text(0.775, 0.9, '(m)', size=20, transform=ax[2][0].transAxes)
    ax[2][1].text(0.8, 0.9, '(n)', size=20, transform=ax[2][1].transAxes)
    ax[2][2].text(0.8, 0.9, '(o)', size=20, transform=ax[2][2].transAxes)
    ax[2][3].text(0.8, 0.9, '(p)', size=20, transform=ax[2][3].transAxes)
    if cutoff=='neighs':
        ax[2][4].text(0.825, 0.9, '(q)', size=20, transform=ax[2][4].transAxes)
    else:
        ax[2][4].text(0.025, 0.9, '(q)', size=20, transform=ax[2][4].transAxes)
    ax[2][5].text(0.85, 0.9, '(r)', size=20, transform=ax[2][5].transAxes)

        
    plt.savefig('SI_connections_hist_{}.pdf'.format(cutoff), dpi=600)

    plt.show()



#########################
#                       #
#    FIGURES 2 AND 3    #
#                       #
#########################



prox_labs  = {'Nq6' : "$N_{q_6}^\mathrm{prx}$", "QclP" : "$Q_6^{cl, \, \mathrm{prx}}$", "QclR" : "$Q_6^{cl\ddag, \, \mathrm{prx}}$",
              "NQP" : "$N_{q_6}^\mathrm{prx}Q_6^{cl, \,\mathrm{prx}}$", "rho" : r'$\rho_N^{*, \,cl, \,\mathrm{prx}}$',
              "NQR" : "$N_{q_6}^\mathrm{prx}Q_6^{cl\ddag, \, \mathrm{prx}}$", "QclM" : "$Q_6^{cl\dag, \, \mathrm{prx}}$",
              "NQM" : "$N_{q_6}^\mathrm{prx}Q_6^{cl\dag, \, \mathrm{prx}}$", "SA" : "$A^{*, \,cl, \, \mathrm{prx}}$", "V" : "$V^{*, \,cl, \, \mathrm{prx}}$",
              "Ecl" : "$U^{*, \, cl, \, \mathrm{prx}}$", "EcN" : "$U^{*, \, cl, \, \mathrm{prx}}/N_{q_6}^\mathrm{prx}$", "Ecl-Ecl" : "$U^{*, \, cl-cl, \, \mathrm{prx}}$",
              "Ecl-clP" : "$U^{*, \, cl, \, \mathrm{prx}}_{\mathrm{intra}}/U^{*, \, cl, \, \mathrm{prx}}$",
              "SAN" : "$A^{*, \,cl, \,\mathrm{prx}}/N_{q_6}^\mathrm{prx}$", "VN" : "$V^{*, \,cl, \, \mathrm{prx}}/N_{q_6}^\mathrm{prx}$",
              "sol" : "$N_{\mathrm{solid}}^\mathrm{prx}$", "Nq6-P" : "$N_{q_6}^\mathrm{prx}/N_{\mathrm{solid}}^\mathrm{prx}$",
              "Nq6-2" : "$N_{q_6, \, 2, \, \mathrm{prx}}$", "Nq6-2P" : "$N_{q_6, \, 2, \, \mathrm{prx}}/N_{q_6}^\mathrm{prx}$", "pB" : "$p_B$"}


conn_labs  = {'Nq6' : "$N_{q_6}^\mathrm{ort}$", "QclP" : "$Q_6^{cl, \, \mathrm{ort}}$", "QclR" : "$Q_6^{cl\ddag, \, \mathrm{ort}}$",
              "NQP" : "$N_{q_6}^\mathrm{ort}Q_6^{cl, \,\mathrm{ort}}$", "rho" : r'$\rho_N^{*, \,cl, \,\mathrm{ort}}$',
              "NQR" : "$N_{q_6}^\mathrm{ort}Q_6^{cl\ddag, \, \mathrm{ort}}$", "QclM" : "$Q_6^{cl\dag, \, \mathrm{ort}}$",
              "NQM" : "$N_{q_6}^\mathrm{ort}Q_6^{cl\dag, \, \mathrm{ort}}$", "SA" : "$A^{*, \,cl, \, \mathrm{ort}}$", "V" : "$V^{*, \,cl, \, \mathrm{ort}}$",
              "Ecl" : "$U^{*, \, cl, \, \mathrm{ort}}$", "EcN" : "$U^{*, \, cl, \, \mathrm{ort}}/N_{q_6}^\mathrm{ort}$", "Ecl-Ecl" : "$U^{*, \, cl-cl, \, \mathrm{ort}}$",
              "Ecl-clP" : "$U^{*, \, cl, \, \mathrm{ort}}_{\mathrm{intra}}/U^{*, \, cl, \, \mathrm{ort}}$",
              "SAN" : "$A^{*, \,cl, \,\mathrm{ort}}/N_{q_6}^\mathrm{ort}$", "VN" : "$V^{*, \,cl, \, \mathrm{ort}}/N_{q_6}^\mathrm{ort}$",
              "sol" : "$N_{\mathrm{solid}}^\mathrm{ort}$", "Nq6-P" : "$N_{q_6}^\mathrm{ort}/N_{\mathrm{solid}}^\mathrm{ort}$",
              "Nq6-2" : "$N_{q_6, \, 2, \, \mathrm{ort}}$", "Nq6-2P" : "$N_{q_6, \, 2, \, \mathrm{ort}}/N_{q_6}^\mathrm{ort}$", "pB" : "$p_B$"}


# File names and locations
# -------------------------

lowT_nfile = "unbiased_distributions_hull.dat"
lowT_sfile = "constructed_distributions_hull.dat"
lowT_bfile = "bigunb_distributions_hull.dat"
lowT_dfile = "bigconstr_distributions_hull.dat"
lowT_nenf  = "unbiased_cluster_decomp_energies.dat"
lowT_senf  = "constructed_cluster_decomp_energies.dat"
lowT_benf  = "bigunb_cluster_decomp_energies.dat"
lowT_denf  = "bigconstr_cluster_decomp_energies.dat"
lowT_nalph = "unbiased_alpha.dat"
lowT_salph = "constructed_alpha.dat"
lowT_balph = "bigunb_alpha.dat"
lowT_dalph = "bigconstr_alpha.dat"
lowT_nglob = "unbiased_distributions_global.dat"
lowT_sglob = "constructed_distributions_global.dat"
lowT_bglob = "bigunb_distributions_global.dat"
lowT_dglob = "bigconstr_distributions_global.dat"

oriented_lowT_nfile = "oriented_unbiased_distributions_conn.dat"
oriented_lowT_sfile = "oriented_constructed_distributions_conn.dat"
oriented_lowT_bfile = "oriented_bigunb_distributions_conn.dat"
oriented_lowT_dfile = "oriented_bigconstr_distributions_conn.dat"
oriented_lowT_nenf  = "oriented_unbiased_cluster_decomp_energies_conn.dat"
oriented_lowT_senf  = "oriented_synth_cluster_decomp_energies_conn.dat"
oriented_lowT_benf  = "oriented_bigunb_cluster_decomp_energies_conn.dat"
oriented_lowT_denf  = "oriented_bigconstr_cluster_decomp_energies_conn.dat"
oriented_lowT_nalph = "oriented_unbiased_oriented_alpha.dat"
oriented_lowT_salph = "oriented_constructed_oriented_alpha.dat"
oriented_lowT_balph = "oriented_bigunb_oriented_alpha.dat"
oriented_lowT_dalph = "oriented_bigconstr_alpha_conn.dat"
oriented_lowT_nglob = None
oriented_lowT_sglob = None
oriented_lowT_bglob = None
oriented_lowT_dglob = None

lowT_nsubset = 'lowT_unbiased_nosubset.dat'
lowT_ssubset = 'lowT_constructed_nosubset.dat'
lowT_bsubset = 'lowT_bigunb_nosubset.dat'
lowT_dsubset = 'lowT_bigconstr_nosubset.dat'

# ======== #
# Get data #
# ======== #


lowT_unbiased   = FuncHist.get_seed_props(lowT_nfile, lowT_nglob, lowT_nalph, lowT_nenf, idx, alpha, 'unbiased_committor.dat')
lowT_constructed = FuncHist.get_seed_props(lowT_sfile, lowT_sglob, lowT_salph, lowT_senf, idx, alpha, 'constructed_committor.dat')
lowT_bigunb    = FuncHist.get_seed_props(lowT_bfile, lowT_bglob, lowT_balph, lowT_benf, idx, alpha, 'bigunb_committor.dat')
lowT_bigconstr  = FuncHist.get_seed_props(lowT_dfile, lowT_dglob, lowT_dalph, lowT_denf, idx, alpha, 'bigconstr_committor.dat')

conn_lowT_unbiased   = FuncHist.get_seed_props(oriented_lowT_nfile, oriented_lowT_nglob, oriented_lowT_nalph, oriented_lowT_nenf, idx, alpha, 'unbiased_committor.dat')
conn_lowT_constructed = FuncHist.get_seed_props(oriented_lowT_sfile, oriented_lowT_sglob, oriented_lowT_salph, oriented_lowT_senf, idx, alpha, 'constructed_committor.dat')
conn_lowT_bigunb    = FuncHist.get_seed_props(oriented_lowT_bfile, oriented_lowT_bglob, oriented_lowT_balph, oriented_lowT_benf, idx, alpha, 'bigunb_committor.dat')
conn_lowT_bigconstr  = FuncHist.get_seed_props(oriented_lowT_dfile, oriented_lowT_dglob, oriented_lowT_dalph, oriented_lowT_denf, idx, alpha, 'bigconstr_committor.dat')


remove_lowT_unbiased   = np.genfromtxt(lowT_nsubset)
if len(remove_lowT_unbiased) != 0:
    lowT_unbiased      = FuncHist.remove_seeds(lowT_unbiased,      remove_lowT_unbiased)
    conn_lowT_unbiased = FuncHist.remove_seeds(conn_lowT_unbiased, remove_lowT_unbiased)
remove_lowT_bigunb    = np.genfromtxt(lowT_bsubset)
if len(remove_lowT_bigunb) != 0:
    lowT_bigunb      = FuncHist.remove_seeds(lowT_bigunb,      remove_lowT_bigunb)
    conn_lowT_bigunb = FuncHist.remove_seeds(conn_lowT_bigunb, remove_lowT_bigunb)
remove_lowT_constructed = np.genfromtxt(lowT_ssubset)
if len(remove_lowT_constructed) != 0:
    lowT_constructed      = FuncHist.remove_seeds(lowT_constructed,      remove_lowT_constructed)
    conn_lowT_constructed = FuncHist.remove_seeds(conn_lowT_constructed, remove_lowT_constructed)
remove_lowT_bigconstr  = np.genfromtxt(lowT_dsubset)
if len(remove_lowT_bigconstr) != 0:
    lowT_bigconstr      = FuncHist.remove_seeds(lowT_bigconstr,      remove_lowT_bigconstr)
    conn_lowT_bigconstr = FuncHist.remove_seeds(conn_lowT_bigconstr, remove_lowT_bigconstr)
    
lowT_natrl = np.vstack((lowT_unbiased,   lowT_bigunb))
lowT_synth = np.vstack((lowT_constructed, lowT_bigconstr)) 

lowTn = lowT_natrl[np.where(abs(lowT_natrl[:, idx[OP]] - lowpmean) <= lowpdiff)[0], :]
lowTs = lowT_synth[np.where(abs(lowT_synth[:, idx[OP]] - lowpmean) <= lowpdiff)[0], :]

conn_lowT_natrl = np.vstack((conn_lowT_unbiased,   conn_lowT_bigunb))
conn_lowT_synth = np.vstack((conn_lowT_constructed, conn_lowT_bigconstr)) 


clowTn = conn_lowT_natrl[np.where(abs(lowT_natrl[:, idx[OP]] - lowpmean) <= lowpdiff)[0], :]
clowTs = conn_lowT_synth[np.where(abs(lowT_synth[:, idx[OP]] - lowpmean) <= lowpdiff)[0], :]


print("# INFO: For ", lowpmean-lowpdiff, "<=", OP, "<=", lowpmean+lowpdiff, " there are ", len(lowTn), nlab, " files and ", len(lowTs), slab, "files for lowT.")

test_lcn = clowTn[np.where(abs(clowTn[:, idx[OP]] - lowpmean) <= lowpdiff)[0], :]
test_lcs = clowTs[np.where(abs(clowTs[:, idx[OP]] - lowpmean) <= lowpdiff)[0], :]

print("# INFO: For ", lowpmean-lowpdiff, "<=", OP, "<=", lowpmean+lowpdiff, " there are ", len(test_lcn), nlab, " files and ", len(test_lcs), slab, "files for oriented lowT.")


NQP_lowTn = lowTn[np.where(abs(lowTn[:, idx['NQP']] - 60) <= 39)[0], :]
NQP_lowTs = lowTs[np.where(abs(lowTs[:, idx['NQP']] - 60) <= 39)[0], :]


print("# INFO: For ", 60-39, "<= NQP <=", 60 + 39, " there are ", len(NQP_lowTn), nlab, " files and ", len(NQP_lowTs), slab, "files for proximal lowT.")

NQP_lcn = clowTn[np.where(abs(clowTn[:, idx['NQP']] - 60) <= 39)[0], :]
NQP_lcs = clowTs[np.where(abs(clowTs[:, idx['NQP']] - 60) <= 39)[0], :]

print("# INFO: For ", 60-39, "<= NQP <=", 60+39, " there are ", len(NQP_lcn), nlab, " files and ", len(NQP_lcs), slab, "files for oriented lowT.")


OP = 'Nq6'
nx = lowTn ; ny = clowTn ; sx = lowTs ; sy = clowTs

edges = FuncHist.get_bins(nx[:, idx[OP]], sx[:, idx[OP]], 1)    

ncomm_dict, n_edges = FuncHist.plot_flatten(lowTn, OP, 250, 125, 251, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
nccmm_dict, nc_edgs = FuncHist.plot_flatten(clowTn, OP, 250, 125, 251, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
scomm_dict, s_edges = FuncHist.plot_flatten(lowTs, OP, 250, 125, 251, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
sccmm_dict, sc_edgs = FuncHist.plot_flatten(clowTs, OP, 250, 125, 251, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)


fig, ax = plt.subplots(1, 3, figsize=(12.8, 4.8))
figs, axs = plt.subplots(1, 1, figsize=(6.4, 4.8))
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.99, wspace=0.22, hspace=0)

ax[0].scatter(nx[:, idx[OP]], ny[:, idx[OP]], color=ncol, marker=nmark, alpha=0.25, label=nlab)
ax[0].scatter(sx[:, idx[OP]], sy[:, idx[OP]], color=scol, marker=smark, alpha=0.25, label=slab)
        
ax[0].plot(edges, edges, 'k--', linewidth=2.5, alpha=0.2)

ax[0].set_xlabel(conn_labs[OP], size=20)
ax[0].set_ylabel(prox_labs[OP], size=20)
ax[0].tick_params(labelsize=15)
ax[0].legend(prop={'size':13}, loc='lower right')

ax[0].tick_params(labelsize=15)
ax[0].text(0.025, 0.925, '(a)', size=20, transform=ax[0].transAxes)


DOP_n = ny[:, idx[OP]] - nx[:, idx[OP]]
DOP_s = sy[:, idx[OP]] - sx[:, idx[OP]]

DOP_bns = FuncHist.get_bins(DOP_n, DOP_s, 250)

Dni, bns = np.histogram(DOP_n, DOP_bns)
Dsi, bns = np.histogram(DOP_s, DOP_bns)

ax[1].set_yscale('log') 
        
ax[1].stairs(Dni, edges=DOP_bns, color=ncol, alpha=0.6, label=nlab, linewidth=2.5)
ax[1].stairs(Dsi, edges=DOP_bns, color=scol, alpha=0.6, label=slab, linewidth=2.5)
ax[1].set_xlabel('$\Delta$'+labs[OP], size=20)
ax[1].set_ylabel('Count', size=20)
        
ax[1].tick_params(labelsize=15)
ax[1].legend(prop={'size':13}, loc='center left')
        
ax[1].text(0.025, 0.925, '(b)', size=20, transform=ax[1].transAxes)
yedges = ax[1].get_ylim()

ax[1].plot([0, 0], yedges, 'k--', linewidth=2.5, alpha=0.2)
ax[1].set_yscale('log')

centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(nccmm_dict, nc_edgs, OP, pBbins, samples=samples)
ax[2].plot(centres, mean, 'b', alpha=0.8, label='Unbiased, '+conn_labs[OP], linewidth=2.5)
axs.plot(centres, mean, 'b', alpha=0.8, label='Unbiased, '+conn_labs[OP], linewidth=2.5)
ax[2].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='b', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='b', alpha=0.2)


centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(sccmm_dict, sc_edgs, OP, pBbins, samples=samples)
ax[2].plot(centres, mean, 'r', alpha=0.8, label='Constructed, '+conn_labs[OP], linewidth=2.5)
axs.plot(centres,   mean, 'r', alpha=0.8, label='Constructed, '+conn_labs[OP], linewidth=2.5)
ax[2].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='r', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='r', alpha=0.2)


centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(ncomm_dict, n_edges, OP, pBbins, samples=samples)
ax[2].plot(centres, mean, 'teal', alpha=0.8, label='Unbiased, '+prox_labs[OP], linewidth=2.5)
axs.plot(centres,   mean, 'teal', alpha=0.8, label='Unbiased, '+prox_labs[OP], linewidth=2.5)
ax[2].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='teal', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='teal', alpha=0.2)


centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(scomm_dict, s_edges, OP, pBbins, samples=samples)
ax[2].plot(centres, mean, 'maroon', alpha=0.8, label='Constructed, '+prox_labs[OP], linewidth=2.5)
axs.plot(centres,   mean, 'maroon', alpha=0.8, label='Constructed, '+prox_labs[OP], linewidth=2.5)
ax[2].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='maroon', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='maroon', alpha=0.2)

        
ax[2].tick_params(labelsize=15)
ax[2].legend(prop={'size':13}, loc='lower right')
ax[2].set_xlabel(labs[OP]+" window centre, width $=5$", size=20)

axs.tick_params(labelsize=15)
axs.legend(prop={'size':13}, loc='lower right')
axs.set_xlabel(labs[OP]+" window centre, width=5", size=20)


ax[2].text(0.025, 0.925, '(c)', size=20, transform=ax[2].transAxes)

ax[2].plot([125, 375], [0.5, 0.5], 'k--', linewidth=2.5, alpha=0.2)
        
        
ax[2].set_ylabel("$\mu_h$", size=20)
axs.set_ylabel("$\mu_h$", size=20)

fig.savefig("Nq6_ort_v_prox.pdf", dpi=600)
figs.savefig("std_committor_cutoffs_Nq6.pdf", dpi=600)
    
plt.show()





OP = 'NQP'
nx = lowTn ; ny = clowTn ; sx = lowTs ; sy = clowTs

edges = FuncHist.get_bins(nx[:, idx[OP]], sx[:, idx[OP]], 1)    

ncomm_dict, n_edges = FuncHist.plot_flatten(lowTn, OP, 60, 39, 79, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
nccmm_dict, nc_edgs = FuncHist.plot_flatten(clowTn, OP, 60, 39, 79, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
scomm_dict, s_edges = FuncHist.plot_flatten(lowTs, OP, 60, 39, 79, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)
sccmm_dict, sc_edgs = FuncHist.plot_flatten(clowTs, OP, 60, 39, 79, pBbins, 'b', 'teal', nlab, idx, labs, samples, tests, err)


fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6))
figs, axs = plt.subplots(1, 1, figsize=(6.4, 4.8))
fig.subplots_adjust(left=0.075, right=0.9975, bottom=0.08, top=0.9975, wspace=0.14, hspace=0.22)

ax[0][0].scatter(nx[:, idx[OP]], ny[:, idx[OP]], color=ncol, marker=nmark, alpha=0.25, label=nlab)
ax[0][0].scatter(sx[:, idx[OP]], sy[:, idx[OP]], color=scol, marker=smark, alpha=0.25, label=slab)

ax[0][0].plot(edges, edges, 'k--', linewidth=2.5, alpha=0.2)

ax[0][0].set_xlabel(conn_labs[OP], size=20)
ax[0][0].set_ylabel(prox_labs[OP], size=20)
ax[0][0].tick_params(labelsize=15)
ax[0][0].legend(prop={'size':13}, loc='lower right')

ax[0][0].tick_params(labelsize=15)
ax[0][0].text(0.025, 0.925, '(a)', size=20, transform=ax[0][0].transAxes)


DOP_n = ny[:, idx[OP]] - nx[:, idx[OP]]
DOP_s = sy[:, idx[OP]] - sx[:, idx[OP]]

DOP_bns = FuncHist.get_bins(DOP_n, DOP_s, 250)

Dni, bns = np.histogram(DOP_n, DOP_bns)
Dsi, bns = np.histogram(DOP_s, DOP_bns)

ax[0][1].set_yscale('log') 
      
ax[0][1].stairs(Dni, edges=DOP_bns, color=ncol, alpha=0.6, label=nlab, linewidth=2.5)
ax[0][1].stairs(Dsi, edges=DOP_bns, color=scol, alpha=0.6, label=slab, linewidth=2.5)
ax[0][1].set_xlabel('$\Delta$'+labs[OP], size=20)
ax[0][1].set_ylabel('Count', size=20)

ax[0][1].tick_params(labelsize=15)
ax[0][1].legend(prop={'size':13}, loc='center left')

ax[0][1].text(0.025, 0.925, '(b)', size=20, transform=ax[0][1].transAxes)
yedges = ax[0][1].get_ylim()

ax[0][1].plot([0, 0], yedges, 'k--', linewidth=2.5, alpha=0.2)
ax[0][1].set_yscale('log')

D_Nn = ny[:, idx['Nq6']] - nx[:, idx['Nq6']]
D_Ns = sy[:, idx['Nq6']] - sx[:, idx['Nq6']]
D_Qn = ny[:, idx['QclP']] - nx[:, idx['QclP']]
D_Qs = sy[:, idx['QclP']] - sx[:, idx['QclP']]


ax[1][0].scatter(D_Qn, D_Nn, color=ncol, marker=nmark, alpha=0.25, label=nlab)
ax[1][0].scatter(D_Qs, D_Ns, color=scol, marker=smark, alpha=0.25, label=slab)

DQ_bns = FuncHist.get_bins(D_Qn, D_Qs, 2)
DN_bns = FuncHist.get_bins(D_Nn, D_Ns, 2)

ax[1][0].plot(DQ_bns, [0, 0, 0], 'k--', linewidth=2.5, alpha=0.2)
ax[1][0].plot([0, 0, 0], DN_bns, 'k--', linewidth=2.5, alpha=0.2)

ax[1][0].set_xlabel('$\Delta Q_6^{cl}$', size=20)
ax[1][0].set_ylabel('$\Delta N_{q_6}$', size=20)
ax[1][0].tick_params(labelsize=15)
ax[1][0].legend(prop={'size':13}, loc='lower right')

ax[1][0].tick_params(labelsize=15)
ax[1][0].text(0.025, 0.925, '(c)', size=20, transform=ax[1][0].transAxes)



centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(nccmm_dict, nc_edgs, OP, pBbins, samples=samples)
ax[1][1].plot(centres, mean, 'b', alpha=0.8, label='Unbiased, '+conn_labs[OP], linewidth=2.5)
axs.plot(centres, mean, 'b', alpha=0.8, label='Unbiased, '+conn_labs[OP], linewidth=2.5)
ax[1][1].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='b', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='b', alpha=0.2)


centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(sccmm_dict, sc_edgs, OP, pBbins, samples=samples)
ax[1][1].plot(centres, mean, 'r', alpha=0.8, label='Constructed, '+conn_labs[OP], linewidth=2.5)
axs.plot(centres,   mean, 'r', alpha=0.8, label='Constructed, '+conn_labs[OP], linewidth=2.5)
ax[1][1].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='r', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='r', alpha=0.2)


centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(ncomm_dict, n_edges, OP, pBbins, samples=samples)
ax[1][1].plot(centres, mean, 'teal', alpha=0.8, label='Unbiased, '+prox_labs[OP], linewidth=2.5)
axs.plot(centres,   mean, 'teal', alpha=0.8, label='Unbiased, '+prox_labs[OP], linewidth=2.5)
ax[1][1].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='teal', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='teal', alpha=0.2)


centres, mean, means, mnmx_err, std_err, width = FuncHist.get_mean_centre(scomm_dict, s_edges, OP, pBbins, samples=samples)
ax[1][1].plot(centres, mean, 'maroon', alpha=0.8, label='Constructed, '+prox_labs[OP], linewidth=2.5)
axs.plot(centres,   mean, 'maroon', alpha=0.8, label='Constructed, '+prox_labs[OP], linewidth=2.5)
ax[1][1].fill_between(centres, mnmx_err[0,:], mnmx_err[1,:],  color='maroon', alpha=0.2)
axs.fill_between(centres, std_err[0,:], std_err[1,:],  color='maroon', alpha=0.2)

        
ax[1][1].tick_params(labelsize=15)
ax[1][1].legend(prop={'size':13}, loc='lower right')
ax[1][1].set_xlabel(labs[OP]+" window centre, width $= {0:.2f}$".format(width), size=20)

axs.tick_params(labelsize=15)
axs.legend(prop={'size':13}, loc='lower right')
axs.set_xlabel(labs[OP]+" window centre, width=5", size=20)


ax[1][1].text(0.025, 0.925, '(d)', size=20, transform=ax[1][1].transAxes)

ax[1][1].plot([21, 99], [0.5, 0.5], 'k--', linewidth=2.5, alpha=0.2)
        
        
ax[1][1].set_ylabel("$\mu_h$", size=20)
axs.set_ylabel("$\mu_h$", size=20)

fig.savefig("NQP_ort_v_prox.pdf", dpi=600)
figs.savefig("std_committor_cutoffs_NQP.pdf", dpi=600)
    
plt.show()

    


