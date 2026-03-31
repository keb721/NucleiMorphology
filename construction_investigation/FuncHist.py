####################################################
#                                                  #
#    Functions for Cluster/Committor Properties    #
#                                                  #
####################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm, colors, colorbar
from scipy import stats

rc('text', usetex=True)
# This code has the following modules
#
# Committor class and functions
# seed_props, hist_ends = get_seed_props(cluster_file, global_file, alpha_file, energy_file, idx, alpha, comm_file_start, comm_file_end, liq, sol,
#                                        exp_traj=300, offset=0, remove=True)
# end points plot       = plot_ends(n_ends, s_ends, OPmean, OPdiff, ncol, scol, nmark, smark, nlab, slab, sol, liq, savename=None)
# bin_edges             = get_bins(natrl, synth, Nbins)
# bin_edges             = get_mult_bins(natrl, synth, d3, d4, Nbins)
# stairs plot           = make_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, op1, op2=None, op3=None, KS=False, savename=None)
# critical stairs plot  = crit_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, crit1, critop, op1, op2, op3, op4, op5, crit2=None, KS=False, savename=None)
# crystallinity scatter = crystallinity_scatter(n, s, ncol, scol, nmark, smark, nlab, slab, idx, labs, savename=None)
# committor scatter     = committor_scatter(n, s, ncol, scol, nmark, smark, nlab, slab, idx, labs, op1, op2=None, data3=None, data4=None, savename=None)
# committor stairs plot = committor_stairs(n1, s1, ncol, scol, nlab, slab, Nbins, lab1, idx, n2=[], s2=[], lab2=None, KS=False, savename=None)
# property scatter plot = distribution_scatter(n, s, Nbins, ncol, scol, nmark, smark, nlab, slab, idx, labs, opdis, op1, op2, op3, op4, op5, savename=None, data3=None, data4=None)
# heatmaps              = heatmap(n, s, x_op, y_op, z_op, idx, x_bins, y_bins, labs, nlab, slab, z_lims=None, differential=True, yticks=None, xticks=None,
#                                 savename = None, shift = None, annotate = False)


    
# Get data about cluster properties, global properites, shape properties energy, and committor and return a single array and histogram test end points
# ----------------------------------------------------------------------------------------------------------------------------------------------------
   
def get_seed_props(cluster_file, global_file, alpha_file, energy_file, idx, alpha, comm_file):
    
    # Get data
    print("# INFO: generating data from {}".format(cluster_file))
    print("# INFO: somewhat assuming shape of data file - check if properties are unexpected")
    
    data = np.genfromtxt(cluster_file) ; data[:, 0] = np.linspace(1, np.shape(data)[0], np.shape(data)[0]) # File, Nq6, QclP, QclR, NQP, NQR, QclM, NQM, SA, V
    if global_file is None:
        glb = data[:, idx['V']:] ; data = data[:, :idx['V']+1]
       

    # Cluster volume/surface area properties
    # ---------------------------------------

    if alpha:
        print("# INFO: taking volumes and surface areas computed by alpha-shape.")
        alpha = np.genfromtxt(alpha_file)
        data[:,   idx['V']] = alpha[:, 1] ; data[:,   idx['SA']] = alpha[:, 2]
        
    else:
        print("# INFO: taking volumes and surface areas computed by quickhull.")   

    dr = data[:, 8]/data[:, 1]     ; data   = np.hstack((data, dr.reshape(len(dr), 1))) # Add SA/atom
    dr = data[:, 9]/data[:, 1]     ; data   = np.hstack((data, dr.reshape(len(dr), 1))) # Add V/atom
    dr = data[:, 1]/data[:, 9]     ; data   = np.hstack((data, dr.reshape(len(dr), 1))) # Add number density


    # Cluster energy properties
    # -------------------------

    enf = np.genfromtxt(energy_file)
    en  = enf[:, 1]             ; data = np.hstack((data, en.reshape(len(en), 1)))   # Add cluster energy
    en  = enf[:, 1]/data[:, 1]  ; data = np.hstack((data, en.reshape(len(en), 1)))   # Add cluster energy per atom
    enc = enf[:, 2]             ; data = np.hstack((data, enc.reshape(len(enc), 1))) # Add cluster-cluster energy
    enp = enf[:, 2]/enf[:, 1]   ; data = np.hstack((data, enp.reshape(len(enp), 1))) # Add percentage cluster-cluster


    # Global OP properties
    # --------------------

    print(global_file)
    
    glb = glb if global_file is None else np.genfromtxt(global_file) 

    ns  = glb[:, 1]              ; data = np.hstack((data, ns.reshape(len(ns), 1)))   # Add total number of solid atoms
    nsp = data[:, 1]/glb[:, 1]   ; data = np.hstack((data, nsp.reshape(len(nsp), 1))) # Add percentage of solid atoms in largest cluster
    n2  = glb[:, 2]              ; data = np.hstack((data, n2.reshape(len(n2), 1)))   # Add size of second-largest cluster
    n2p = glb[:, 2]/data[:, 1]   ; data = np.hstack((data, n2p.reshape(len(n2p), 1))) # Add percentage of atoms in second-largest cluster compared to largest cluster

    # Committor properties
    # ---------------------

    comm = np.genfromtxt(comm_file)        
    
    data = np.hstack((data, comm[:, 1].reshape((len(data), 1))))
    
    return data

def remove_seeds(data, IDs):
    for index in IDs:
        loc  = np.where(data[:, 0] == index)[0][0]
        data = np.vstack((data[:loc, :], data[loc+1:, :]))
    return data 

                     
# Mean and variance (assume Gaussian) based on centre of sampled region
# ---------------------------------------------------------------------

def change_centre(committor_dict, edges, bins, width, uni='resample', samples=None, rplace=None, error='std', tests=1):

    # print("# INFO: considering samples of a width of ", width, " OP histogram bins")
    
    samples = 50   if samples is None else samples
    rplace  = True if rplace  is None else rplace

    message = 'reweighting' if uni=='weight' else 'taking '+str(samples)
    if uni=='resample':
        message += ' samples with replacement' if rplace else 'samples without replacement'
    
    centres = []
    means   = []
    sigmas  = []

    window_width = edges[width]-edges[0]

    # print("# INFO:", message, "to get uniform OP distribution")
    
    for j in range(len(edges)-width):
        centres.append(0.5*(edges[j]+edges[j+width]))
        if uni=='weight':
            mu, sig = weighted_hist(committor_dict, edges[j:j+width+1], bins, props=True)
        elif uni=='resample':
            if tests==1:
                mu, sig = resampled_hist(committor_dict, edges[j:j+width+1], bins, samples=samples, rplace=rplace, props=True)
            else:
                mu = np.zeros(tests) ; sig = np.zeros(tests)
                for t in range(tests):
                    mu[t], sig[t] = resampled_hist(committor_dict, edges[j:j+width+1], bins, samples=samples, rplace=rplace, props=True)
                    
        means.append(mu)
        sigmas.append(sig)

    if tests==1:
        return centres, means, sigmas, window_width

    else:
        # print("# INFO: presenting histogram properties as the average over", tests, "realisations")
        means = np.array(means).transpose()  ; sigmas = np.array(sigmas).transpose()
        mm    = np.mean(means, axis=0)       ; sm     = np.mean(sigmas, axis=0)
        me    = np.zeros((2,  len(centres))) ; se     = np.zeros((2,  len(centres))) 
        
        if  error=='minmax':
            # print("# INFO: giving minimum and maximum values of histogram properties")
            
            me[0, :] = np.min(means,  axis=0)
            me[1, :] = np.max(means,  axis=0)
            se[0, :] = np.min(sigmas, axis=0)
            se[1, :] = np.max(sigmas, axis=0)
             
        elif error=="std":

            me[0, :] = mm - np.std(means,  axis=0)/np.sqrt(tests)
            me[1, :] = mm + np.std(means,  axis=0)/np.sqrt(tests)
            se[0, :] = sm - np.std(sigmas, axis=0)/np.sqrt(tests)
            se[1, :] = sm + np.std(sigmas, axis=0)/np.sqrt(tests)

        else:
            print("ERROR: unknown error type. Try 'std' or 'minmax'")
            return None

        return centres, mm, me, sm, se, window_width


    

# Plot only change in mean with centre with line at 0.5 as guide to eye
# --------------------------------------------------------------------

def get_mean_centre(comm, OP_edges, OP, pBbins, samples=None, rwt='resample', test_width=5):

    tests = 50
    centres, mean, mnmx_err, sig, sg_err, window_width = change_centre(comm, OP_edges, pBbins, test_width, rwt, samples, error='minmax', tests=tests)
    centres, mean1, std_err, sig, sg_err, window_width = change_centre(comm, OP_edges, pBbins, test_width, rwt, samples, error='std', tests=tests)
    

    return centres, mean, mean1, mnmx_err, std_err, window_width
            



# Plot OP distribution, and effects of resampling
# -----------------------------------------------

def plot_flatten(s, OP, OP_mean, OP_maxdiff, OPbins, pBbins, scol, scol2, slab, idx, labs, samples, tests=10, err='minmax', n=[], ncol=None, ncol2=None, nlab=None, savename=None):

    OP_edges = np.linspace(OP_mean-OP_maxdiff, OP_mean+OP_maxdiff, OPbins+1)
    pB_edges = np.linspace(0, 1, pBbins+1)
    half_bin = 0.5/(pBbins)
    pB_mids  = np.linspace(half_bin, 1-half_bin, pBbins)
    
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6))
    fig.subplots_adjust(top=0.995, right=0.995, bottom=0.065, left=0.075, hspace=0.175, wspace=0.175) 
    
    sop, OP_edges = np.histogram(s[:, idx[OP]], OP_edges)
    ax[0][0].stairs(sop, edges=OP_edges, color=scol, alpha=0.8, label=slab, linewidth=2.5)

    ax[0][0].set_ylabel("Count", size=20)
    ax[0][0].set_xlabel(labs[OP], size=20)
    ax[0][0].tick_params(labelsize=15)

    scomm, smax, smin, smn = get_committor_dict(s, OP, OP_edges, idx)
    
    Nmax = smax ; Nmin = smin ; Nmn = smn
            
    # Full histograms
    # ---------------

    spb = []
    for i in OP_edges[:-1]:
        spb.extend(scomm[i])


    swt           = [1.0/len(spb)]*len(spb)
    sbp, pB_edges = np.histogram(spb, pB_edges, weights=swt)

    ax[0][1].stairs(sbp, edges=pB_edges, color=scol, alpha=0.4, label=slab+", unresampled", linewidth=2.5)
    ax[1][0].stairs(sbp, edges=pB_edges, color=scol, alpha=0.4, label=slab+", unresampled", linewidth=2.5)
    ax[1][1].stairs(sbp, edges=pB_edges, color=scol, alpha=0.4, label=slab+", unresampled", linewidth=2.5)

    if len(n) != 0:
        nop, OP_edges = np.histogram(n[:, idx[OP]], OP_edges)
        ax[0][0].stairs(nop, edges=OP_edges, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
        
        ncomm, nmax, nmin, nmn = get_committor_dict(n, OP, OP_edges, idx)
        Nmax = max(nmax, smax) ; Nmin = min(nmin, smin) ; Nmn = round(0.5*(nmn+smn))
        npb = []
        for i in OP_edges[:-1]:
            npb.extend(ncomm[i])
            
        nwt           = [1.0/len(npb)]*len(npb)
        nbp, pB_edges = np.histogram(npb, pB_edges, weights=nwt)
        ax[0][1].stairs(nbp, edges=pB_edges, color=ncol, alpha=0.4, label=nlab+", unresampled", linewidth=2.5)
        ax[1][0].stairs(nbp, edges=pB_edges, color=ncol, alpha=0.4, label=nlab+", unresampled", linewidth=2.5)
        ax[1][1].stairs(nbp, edges=pB_edges, color=ncol, alpha=0.4, label=nlab+", unresampled", linewidth=2.5)
        print("# INFO: minimum, maximum and mean number of samples in a bin:", nmin, nmax)
        
    print("# INFO: minimum, maximum and mean number of samples in a bin:", Nmin, Nmax, Nmn)
    print("# INFO: minimum, maximum and mean number of samples in a bin:", smin, smax)
    
    

    
    
    # Weighted histograms
    # --------------------

    sbp, bns = weighted_hist(scomm, OP_edges, pBbins)
    ax[0][1].stairs(sbp, edges=pB_edges, color=scol2, alpha=0.4, label=slab+", weights", linewidth=2.5)
    ax[0][1].set_ylabel("$P(p_B)$", size=20)
    ax[0][1].set_xlabel("$p_B$", size=20)
    ax[0][1].tick_params(labelsize=15)

    if len(n) !=0:
        nbp, bns = weighted_hist(ncomm, OP_edges, pBbins)
        ax[0][1].stairs(nbp, edges=pB_edges, color=ncol2, alpha=0.4, label=nlab+", weights", linewidth=2.5)

    ax[0][1].legend(prop={'size':13}, loc='upper left')

        
    # Resampled histograms
    # --------------------

    rt = samples
    
    if samples > Nmin:
        rt = Nmin
        print("# WARNING: attempting to take more samples than possible. Taking ", Nmin, " samples without replacement and ", samples, " samples with replacement.")
    else:
        print("# INFO: taking ", samples, " samples.")
        
    spb, pB_edges, se = resampled_hist(scomm, OP_edges, pBbins, rt, rplace=False, tests=tests, error=err)
    ax[1][0].stairs(spb, edges=pB_edges, color=scol2, alpha=0.4, label=slab+", resampled", linewidth=2.5)
    ax[1][0].errorbar(pB_mids, spb, linestyle='none', yerr=se, capsize=0.1, ecolor=scol2, alpha=0.4)
    ax[1][0].set_ylabel("$P(p_B)$", size=20)
    ax[1][0].set_xlabel("$p_B$", size=20)
    ax[1][0].tick_params(labelsize=15)

    spb, pB_edges, se = resampled_hist(scomm, OP_edges, pBbins, samples, rplace=True, tests=tests, error=err)
    ax[1][1].stairs(spb, edges=pB_edges, color=scol2, alpha=0.4, label=slab+", resampled with replacement", linewidth=2.5)
    ax[1][1].errorbar(pB_mids, spb, linestyle='none', yerr=se, capsize=0.1, ecolor=scol2, alpha=0.4)
    ax[1][1].set_ylabel("$P(p_B)$", size=20)
    ax[1][1].set_xlabel("$p_B$", size=20)
    ax[1][1].tick_params(labelsize=15)


    if len(n) != 0 :
        npb, pB_edges, ne = resampled_hist(ncomm, OP_edges, pBbins, rt, rplace=False, tests=tests, error=err)
        ax[1][0].stairs(npb, edges=pB_edges, color=ncol2, alpha=0.4, label=nlab+", resampled", linewidth=2.5)
        ax[1][0].errorbar(pB_mids, npb, linestyle='none', yerr=ne, capsize=0.1, ecolor=ncol2, alpha=0.4)
        
        npb, pB_edges, ne = resampled_hist(ncomm, OP_edges, pBbins, samples, rplace=True, tests=tests, error=err)
        ax[1][1].stairs(npb, edges=pB_edges, color=ncol2, alpha=0.4, label=nlab+", resampled with replacement", linewidth=2.5)
        ax[1][1].errorbar(pB_mids, npb, linestyle='none', yerr=ne, capsize=0.1, ecolor=ncol2, alpha=0.4)

    ax[1][0].legend(prop={'size':13}, loc='upper left')
    ax[1][1].legend(prop={'size':13}, loc='upper left')
    ax[0][0].legend(prop={'size':13}, loc='upper right')


    ax[0][0].text(0.635, 0.925, '(a)', size=20, transform=ax[0][0].transAxes)
    ax[0][1].text(0.5, 0.925, '(b)', size=20, transform=ax[0][1].transAxes)
    ax[1][0].text(0.5, 0.925, '(c)', size=20, transform=ax[1][0].transAxes)
    ax[1][1].text(0.715, 0.925, '(d)', size=20, transform=ax[1][1].transAxes)
    
    
    if savename != None:
        plt.savefig(savename, dpi=450)
                
    plt.show()

    if len(n) != 0 :
        return scomm, ncomm, OP_edges

    return scomm, OP_edges


def get_committor_dict(data, OP, OPedges, idx):  

    Nmin = 36
    Nmax = 0
    Nmn  = 0

    pB = {}
    
    for i in range(len(OPedges[:-2])):
        dst = np.where((data[:, idx[OP]] >= OPedges[i])&(data[:, idx[OP]] < OPedges[i+1]))[0]
        c = []
        for j in dst:
            c.append(data[j, idx['pB']])

        Nmin = len(c) if len(c) < Nmin else Nmin
        Nmax = len(c) if len(c) > Nmax else Nmax
        Nmn += len(c)
        pB[OPedges[i]] = c

    dst = np.where((data[:, idx[OP]] >= OPedges[-2])&(data[:, idx[OP]] <= OPedges[-1]))[0]
    c = []
    for j in dst:
        c.append(data[j, idx['pB']])
    Nmin = len(c) if len(c) < Nmin else Nmin
    Nmax = len(c) if len(c) > Nmax else Nmax
    Nmn += len(c)
    pB[OPedges[-2]] = c

    return pB, Nmax, Nmin, Nmn/(len(OPedges)-1)

# Compute weighted histogram
# --------------------------

def weighted_hist(committor_dict, OP_edges, pBbins, props=False):
    OPbns = len(OP_edges)-1
    full  = 1.0/OPbns

    pB_edges = np.linspace(0, 1, pBbins+1)
    
    pb = []
    wt = []
    
    for i in OP_edges[:-1]:
        pb.extend(committor_dict[i])
        wt.extend([full/len(committor_dict[i])]*len(committor_dict[i]))

    if props:
        mean = np.average(pb, weights=wt)
        std  = np.sqrt(np.cov(pb, aweights=wt, ddof=0))
        return mean, std
        
    bp, bins = np.histogram(pb, pB_edges, weights=wt)

    return bp, bins

# Compute resampled histogram
# ----------------------------

def resampled_hist(committor_dict, OP_edges, pBbins, samples, rplace=False, tests=1, error='minmax', props=False):
    pB_edges = np.linspace(0, 1, pBbins+1)

    h = np.zeros((tests, pBbins))
        
    for j in range(tests):
        pb = []
        for i in OP_edges[:-1]:
            d = committor_dict[i]
            try:
                p = np.random.choice(d, size=samples, replace=rplace)
                pb.extend(p)
            except ValueError:
                print("ERROR: trying to take too many samples")
                return None
            
        wt = [1.0/len(pb)]*len(pb)
        h[j, :], bins = np.histogram(pb, pB_edges, weights=wt)
        if props:
            mean = np.mean(pb)
            std  = np.std(pb)
            return mean, std

    pb = np.mean(h, axis=0)
    
    if tests==1:
        return pb, pB_edges
        
    if  error=='minmax':
        print("# INFO: giving minimum and maximum values for ", tests, "samples")
        e       = np.zeros((2,  pBbins))
        e[0, :] = pb - np.min(h, axis=0)
        e[1, :] = np.max(h, axis=0) - pb
        
    elif error=="std":
        e = np.std(h, axis=0)/np.sqrt(tests)

    else:
        print("ERROR: unknown error type. Try 'std' or 'minmax'")
        return None

    return pb, pB_edges, e


def get_bins(natrl, synth, Nbins):
    mn = np.min(natrl) if np.min(natrl) < np.min(synth) else np.min(synth)
    mx = np.max(natrl) if np.max(natrl) > np.max(synth) else np.max(synth)
    bn = np.linspace(mn, mx, Nbins+1)
    return bn
