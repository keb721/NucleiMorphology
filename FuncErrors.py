###############################################
#                                              #
#    Functions for Committor Error Analysis    #
#                                              #
################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm, colors, colorbar
from scipy import stats, optimize

rc('text', usetex=True)

# This code has the following modules
#
# Committor class and functions
# gaussian   = Gaussian(x, mu, sigma)
#
# --------------------------------------------------
# Functions to probe effect of finite amount of data
# --------------------------------------------------
# end points = end_point_dict(data, data_loc, end_name, exp_traj=300, alt_loc=None, offset=100000)
# d_pB       = d_pB(comm, end_point_dict, tcom, Ncom, sol, liq, rplc=False)
# mu, sigma  = prop_hist_traj(comm, end_point_dict, tcom, Ncom, sol, liq, rplc=False)
# mu, sigma  = prop_hist_seed(comm, end_point_dict, scom, Ncom, sol, liq, rplc=False)
# mu, sigma  = prop_hist_all(comm, end_point_dict, sol, liq)
#
# -------------------------------------------------------------------------------------------------
# Functions to probe effect of lack of flatness in OP space - NOTE: assumes data array contains pB
# ------------------------------------------------------------------------------------------------
# commitor_dict                                      = get_committor_dict(data, OP, OPedges, idx)
# rewighted_pB, bins/mean, std                       = weighted_hist(pb, OPedges, pB_edges, props=False)
# resampled_pb, pB_edges, e/mean, std                = resampled_hist(committor_dict, OP_edges, pB_edges, samples, rplace=False, tests=1, error='minmax', props=False)
# commitor_dict_s, [committor_dict_n], OP_edges      = plot_flatten(s, OP, OP_mean, OP_maxdiff, OPbins, pBbins, scol, scol2, slab, idx, labs, samples, tests, err='minmax',
#                                                                   n=[], ncol=None, ncol2=None, nlab=None, savename=None)
# widths, mean, mean_err, sig, sig_err                = change_width(comm_dict, OP_edges, pBbins, uni='resample', samples=None, rplace=None, error='std', tests=1)
# centres, mean, mean_err, sig, sig_err, window_width = change_centre(comm_dict, OP_edges, pBbins, width, uni='resample', samples=None, rplace=None, error='std', tests=1)
# histprops graph                                     = plot_histprops_all(scomm, scol, slab, OP_mean, OP_edges, OP, pBbins, labs, samples=None, err_type='minmax', ncomm=[],
#                                                       ncol=None, nlab=None, d3comm=[], d3col=None, d3lab=None, d4comm=[], d4col=None, d4lab=None, savename=None, tests=50,
#                                                       rwt='resample', test_width=5)
# histprops mean changing centre                      = plot_histprops_mean_centre(scomm, scol, slab, OP_edges, OP, pBbins, labs, samples=None, err_type='minmax', ncomm=[],
#                                                       ncol=None, nlab=None, d3comm=[], d3col=None, d3lab=None, d4comm=[], d4col=None, d4lab=None, savename=None, tests=50,
#                                                       rwt='resample', test_width=5)


# ============================= #
# COMMITTOR CLASS AND FUNCTIONS #
# ============================= #

# Define committor
# -----------------

class Committor:
    def pB(self, data, sol, liq, trajs):
        # Let this be the default case for everything defined
        B = len(np.where(data >= sol)[0])
        A = len(np.where(data <= liq)[0]) 
        return B / (B + A) 

def lpB(data, sol, liq, trajs):
    B = len(np.where(data >= sol)[0])
    return B / trajs

def spB(data, sol, liq, trajs):
    A = len(np.where(data <= liq)[0])
    return 1 - A / trajs

# Guassian function
# ------------------

def Gaussian(x, mu, sigma):
    p = (x-mu)/sigma
    g = np.exp(p*p*0.5*-1.0)
    g = 1.0*g/(sigma*np.sqrt(2*np.pi))
    return g

# =============================== #
# EFFECT OF FINITE AMOUNT OF DATA #
# =============================== #


# Get end point dictionaries
# --------------------------

def end_point_dict(data, data_loc, end_name, exp_traj=300, alt_loc=None, offset=100000):
    
    end_dict = {}
    
    for i in range(np.shape(data)[0]):
        ID = str(int(data[i, 0]))
        if data[i, 0] > offset:
            N  = str(int(data[i, 0]-offset))
            fl = alt_loc+N+end_name
        else:
            N     = str(int(data[i, 0]))
            fl = data_loc+N+end_name
        try:
            d     = np.genfromtxt(fl)
            trajs = len(d)
            if trajs!=exp_traj:
                print("# WARNING: File ", fl," has incorrect number of trajectories", trajs, ". No dictionary entry has been made.")
            else:
                end_dict[ID] = d

        except FileNotFoundError:
            print("# WARNING:  File ", fl," not found. No dictionary entry has been made.")

    return end_dict



# Error in individual committor estimates
# ---------------------------------------
    
def d_pB(comm, end_point_dict, tcom, Ncom, sol, liq, rplc=False):

    IDs = list(end_point_dict.keys())

    dpB = np.zeros((Ncom*len(IDs), len(tcom))) # List of differences in committor 

    for i, ID in enumerate(IDs):
        d     = end_point_dict[ID]
        trajs = len(d)
        pB_all = comm.pB(d, sol, liq, trajs)
        for j, t in enumerate(tcom):
            for trials in range(Ncom):
                sd         = np.random.choice(d, size=t, replace=rplc)
                ii         = i*Ncom + trials
                pB_new     = comm.pB(sd, sol, liq, t)
                dpB[ii, j] = pB_new - pB_all

    return dpB


# Overall histogram properties due to different number of trajectories
# ---------------------------------------------------------------------

def prop_hist_traj(comm, end_point_dict, tcom, Ncom, sol, liq, rplc=False):
    mu    = np.zeros((Ncom, len(tcom))) # List of differences in histogram mean
    sigma = np.zeros((Ncom, len(tcom))) # List of differences in histogram variance 

    IDs = list(end_point_dict.keys())
    
    for tt, t in enumerate(tcom):
        for trial in range(Ncom):
            hist = []
            for i, ID in enumerate(IDs):
                d      = end_point_dict[ID]
                sd     = np.random.choice(d, size=t, replace=rplc)
                pB_new = comm.pB(sd, sol, liq, t)
                hist.append(pB_new)
            # Have all of the relevant seed data
            mu[trial, tt]    = np.mean(hist)
            sigma[trial, tt] = np.std(hist)

    return mu, sigma


# Overall histogram properties due to different number of seeds
# --------------------------------------------------------------

def prop_hist_seed(comm, end_point_dict, scom, Ncom, sol, liq, rplc=False):
    mu    = np.zeros((Ncom, len(scom))) # List of differences in histogram mean
    sigma = np.zeros((Ncom, len(scom))) # List of differences in histogram variance 

    pBs = []
    
    IDs = list(end_point_dict.keys())
    for i, ID in enumerate(IDs):
        d      = end_point_dict[ID]
        pB_new = comm.pB(d, sol, liq, len(d))
        pBs.append(pB_new)

    for ss, seeds in enumerate(scom):
        for trials in range(Ncom):
            hist =  np.random.choice(pBs, size=seeds, replace=rplc)

            mu[trials, ss]    = np.mean(hist)
            sigma[trials, ss] = np.std(hist)

    return mu, sigma


# Total histogram properties from all seeds and trajectories
# -----------------------------------------------------------

def prop_hist_all(comm, end_point_dict, sol, liq):

    pBs = []
    
    IDs = list(end_point_dict.keys())
    for i, ID in enumerate(IDs):
        d      = end_point_dict[ID]
        pB_new = comm.pB(d, sol, liq, len(d))
        pBs.append(pB_new)

    mu    = np.mean(pBs)
    sigma = np.std(pBs)

    return mu, sigma



# ================================ #
# EFFECT OF LACK OF FLATNESS IN RC #
# ================================ #

# Note these rely on data with included pB, unlike above

# Create dictionary of committor indexed by OP value
# --------------------------------------------------

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
        
    print("# INFO: minimum, maximum and mean number of samples in a bin:", Nmin, Nmax, Nmn)

    
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



# Mean and variance (assume Gaussian) based on width of sampled region
# --------------------------------------------------------------------

def change_width(committor_dict, edges, bins, uni='resample', samples=None, rplace=None, error='std', tests=1):
    
    samples = 50    if samples is None else samples
    rplace  = True  if rplace  is None else rplace

    message = 'reweighting' if uni=='weight' else 'taking '+str(samples)
    if uni=='resample':
        message += ' samples with replacement' if rplace else 'samples without replacement'

    widths = []
    means  = []
    sigmas = []
    if len(edges)%2==1:
        print("# INFO: even number of OP bins, therefore central bin will be wider than expected")
        low  = int(0.5*len(edges)-1)
        high = low + 2
    else:
        high = int(0.5*len(edges))
        low  = high - 1
        
    print("# INFO:", message, "to get uniform OP distribution")
    
    for j in range(low+1):
        widths.append(0.5*(edges[high+j]-edges[low-j]))
        if uni=='weight':
            mu, sig = weighted_hist(committor_dict, edges[low-j:high+j+1], bins, props=True)
        elif uni=='resample':
            if tests==1:
                mu, sig = resampled_hist(committor_dict, edges[low-j:high+j+1], bins, samples=samples, rplace=rplace, props=True)
            else:
                mu = np.zeros(tests) ; sig = np.zeros(tests)
                for t in range(tests):
                    mu[t], sig[t] = resampled_hist(committor_dict, edges[low-j:high+j+1], bins, samples=samples, rplace=rplace, props=True)
                    
        means.append(mu)
        sigmas.append(sig)

    if tests==1:
        return widths, means, sigmas

    else:
        print("# INFO: presenting histogram properties as the average over", tests, "realisations")
        means = np.array(means).transpose() ; sigmas = np.array(sigmas).transpose()
        mm    = np.mean(means, axis=0)      ; sm     = np.mean(sigmas, axis=0)
        me    = np.zeros((2,  len(widths))) ; se    = np.zeros((2,  len(widths)))
        if  error=='minmax':
            print("# INFO: giving minimum and maximum values of histogram properties")
        
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

        return widths, mm, me, sm, se


# Mean and variance (assume Gaussian) based on centre of sampled region
# ---------------------------------------------------------------------

def change_centre(committor_dict, edges, bins, width, uni='resample', samples=None, rplace=None, error='std', tests=1):

    print("# INFO: considering samples of a width of ", width, " OP histogram bins")
    
    samples = 50   if samples is None else samples
    rplace  = True if rplace  is None else rplace

    message = 'reweighting' if uni=='weight' else 'taking '+str(samples)
    if uni=='resample':
        message += ' samples with replacement' if rplace else 'samples without replacement'
    
    centres = []
    means   = []
    sigmas  = []

    window_width = edges[width]-edges[0]

    print("# INFO:", message, "to get uniform OP distribution")
    
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
        print("# INFO: presenting histogram properties as the average over", tests, "realisations")
        means = np.array(means).transpose()  ; sigmas = np.array(sigmas).transpose()
        mm    = np.mean(means, axis=0)       ; sm     = np.mean(sigmas, axis=0)
        me    = np.zeros((2,  len(centres))) ; se     = np.zeros((2,  len(centres))) 
        
        if  error=='minmax':
            print("# INFO: giving minimum and maximum values of histogram properties")
            
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


    
# Plot histogram mean and std as window width and center change
# -------------------------------------------------------------

def plot_histprops_all(scomm, scol, slab, OP_mean, OP_edges, OP, pBbins, labs, samples=None, err_type='minmax', ncomm=[], ncol=None, nlab=None,
                       d3comm=[], d3col=None, d3lab=None, d4comm=[], d4col=None, d4lab=None, savename=None, tests=50, rwt='resample', test_width=5):


    fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.065, right=0.9975, bottom=0.07, top=0.9975, wspace=0, hspace=0)

    if tests != 1:
        widths, smean, smn_err, ssig, ssg_err = change_width(scomm, OP_edges, pBbins, rwt, samples, error=err_type, tests=tests)
        ax[0][0].plot(widths, smean,  color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[0][0].fill_between(widths, smn_err[0,:], smn_err[1,:],  color=scol, alpha=0.2)
        ax[1][0].plot(widths, ssig, color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[1][0].fill_between(widths, ssg_err[0,:], ssg_err[1,:],  color=scol, alpha=0.2)
    
        centres, smean, smn_err, ssig, ssg_err, window_width = change_centre(scomm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
        ax[0][1].plot(centres, smean,  color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[0][1].fill_between(centres, smn_err[0,:], smn_err[1,:],  color=scol, alpha=0.2)
        ax[1][1].plot(centres, ssig, color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[1][1].fill_between(centres, ssg_err[0,:], ssg_err[1,:],  color=scol, alpha=0.2)

    else:
        widths, smean, ssig, = change_width(scomm, OP_edges, pBbins, rwt, samples, error=err_type, tests=tests)
        ax[0][0].plot(widths, smean,  color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[1][0].plot(widths, ssig, color=scol, alpha=0.8, label=slab, linewidth=2.5)
    
        centres, smean, ssig,  window_width = change_centre(scomm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
        ax[0][1].plot(centres, smean,  color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[1][1].plot(centres, ssig, color=scol, alpha=0.8, label=slab, linewidth=2.5)
        
        
    extra_comm = {'n' : ncomm, 'd3' : d3comm, 'd4' : d4comm}
    extra_col  = {'n' : ncol,  'd3' : d3col,  'd4' : d4col}
    extra_lab  = {'n' : nlab,  'd3' : d3lab,  'd4' : d4lab}
    
    
    for data in ['n', 'd3', 'd4']:
        comm = extra_comm[data]
        if len(comm) != 0:
            col = extra_col[data] ; lab = extra_lab[data]
            if tests != 1:
                widths, mean, mn_err, sig, sg_err = change_width(comm, OP_edges, pBbins, rwt, samples, error=err_type, tests=tests)
                ax[0][0].plot(widths, mean,  color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax[0][0].fill_between(widths, mn_err[0,:], mn_err[1,:],  color=col, alpha=0.2)
                ax[1][0].plot(widths, sig, color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax[1][0].fill_between(widths, sg_err[0,:], sg_err[1,:],  color=col, alpha=0.2)
    
                centres, mean, mn_err, sig, sg_err, window_width = change_centre(comm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
                ax[0][1].plot(centres, mean,  color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax[0][1].fill_between(centres, mn_err[0,:], mn_err[1,:],  color=col, alpha=0.2)
                ax[1][1].plot(centres, sig, color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax[1][1].fill_between(centres, sg_err[0,:], sg_err[1,:],  color=col, alpha=0.2)
                print(centres, mean, smean)
                            
            else:
                widths, mean, sig, = change_width(comm, OP_edges, pBbins, rwt, samples, error=err_type, tests=tests)
                ax[0][0].plot(widths, mean,  color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax[1][0].plot(widths, sig, color=col, alpha=0.8, label=lab, linewidth=2.5)
    
                centres, mean, sig, window_width = change_centre(comm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
                ax[0][1].plot(centres, mean,  color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax[1][1].plot(centres, sig, color=col, alpha=0.8, label=lab, linewidth=2.5)


    
    ax[0][0].set_ylabel("$\mu_h$", size=20)
    ax[1][0].set_ylabel("$\sigma_h$", size=20)
    if OP == 'Nq6':
        ax[1][0].set_xlabel(labs[OP]+" window width, centre $= {:3d}$".format(OP_mean), size=20)
        ax[1][1].set_xlabel(labs[OP]+" window centre, width $= {:3d}$".format(round(window_width)), size=20)
    else:
        ax[1][0].set_xlabel(labs[OP]+" window width, centre $= {0:.2f}$".format(OP_mean), size=20)
        ax[1][1].set_xlabel(labs[OP]+" window centre, width $= {0:.2f}$".format(window_width), size=20)
        
    ax[0][0].tick_params(labelsize=15)
    ax[0][1].tick_params(labelsize=15)
    ax[1][0].tick_params(labelsize=15)
    ax[1][1].tick_params(labelsize=15)
    ax[1][1].legend(prop={'size':13})

    ax[0][0].text(0.025, 0.925, '(a)', size=20, transform=ax[0][0].transAxes)
    ax[0][1].text(0.025, 0.925, '(b)', size=20, transform=ax[0][1].transAxes)
    ax[1][0].text(0.025, 0.925, '(c)', size=20, transform=ax[1][0].transAxes)
    ax[1][1].text(0.29, 0.925, '(d)', size=20, transform=ax[1][1].transAxes)
    
    if savename != None:
        plt.savefig(savename, dpi=450)

    plt.show()

    return None
    

# Plot only change in mean with centre with line at 0.5 as guide to eye
# --------------------------------------------------------------------

def plot_histprops_mean_centre(scomm, scol, slab, OP_edges, OP, pBbins, labs, samples=None, err_type='minmax', ncomm=[], ncol=None, nlab=None,
                               d3comm=[], d3col=None, d3lab=None, d4comm=[], d4col=None, d4lab=None, savename=None, tests=50, rwt='resample', test_width=5):

    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.13, right=0.9975, bottom=0.14, top=0.9975, wspace=0, hspace=0)

    if tests != 1:
        centres, smean, smn_err, ssig, ssg_err, window_width = change_centre(scomm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
        ax.plot(centres, [0.5]*len(centres), color='k', linestyle='dashed', linewidth=2.5, alpha=0.6)
        ax.plot(centres, smean,  color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax.fill_between(centres, smn_err[0,:], smn_err[1,:],  color=scol, alpha=0.2)

    else:
        centres, smean, ssig, window_width = change_centre(scomm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
        ax.plot(centres, [0.5]*len(centres), color='k', linestyle='dashed', linewidth=2.5, alpha=0.6)
        ax.plot(centres, smean,  color=scol, alpha=0.8, label=slab, linewidth=2.5)
    
    extra_comm = {'n' : ncomm, 'd3' : d3comm, 'd4' : d4comm}
    extra_col  = {'n' : ncol,  'd3' : d3col,  'd4' : d4col}
    extra_lab  = {'n' : nlab,  'd3' : d3lab,  'd4' : d4lab}
    
    
    for data in ['n', 'd3', 'd4']:
        comm = extra_comm[data]
        if len(comm) != 0:
            col = extra_col[data] ; lab = extra_lab[data]
            if tests != 1:
                centres, mean, mn_err, sig, sg_err, window_width = change_centre(comm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
                ax.plot(centres, mean,  color=col, alpha=0.8, label=lab, linewidth=2.5)
                ax.fill_between(centres, mn_err[0,:], mn_err[1,:],  color=col, alpha=0.2)
            else:
                centres, mean, sig, window_width = change_centre(comm, OP_edges, pBbins, test_width, rwt, samples, error=err_type, tests=tests)
                ax.plot(centres, mean,  color=col, alpha=0.8, label=lab, linewidth=2.5)
                
    ax.set_ylabel("$\mu_h$", size=20)
    if OP == 'Nq6':
        ax.set_xlabel(labs[OP]+" window centre, width $= {:3d}$".format(round(window_width)), size=20)
    else:
        ax.set_xlabel(labs[OP]+" window centre, width $= {0:.2f}$".format(window_width), size=20)
        
    ax.tick_params(labelsize=15)
    ax.legend(prop={'size':13})

    if savename != None:
        plt.savefig(savename, dpi=450)

    plt.show()

    return None
    
