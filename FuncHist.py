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
# full property stairs  = mult_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, opdis, op1, op2, op3, op4, op5, savename=None, data3=None, data4=None)
# Nq6 comparision hist  = Nq6_comp(data, histcol, bins, upper, idx, labs, op1, op2, op3, op4, op5, savename=None)
# heatmaps              = heatmap(n, s, x_op, y_op, z_op, idx, x_bins, y_bins, labs, nlab, slab, z_lims=None, differential=True, yticks=None, xticks=None, savename = None, shift = None, annotate = False)
# mult heatmaps         = mult_heatmap(n, s, d3, d4, x_op, y_ops, idx, x_bins, y_bins, labs, nlab, slab, savename = None, shift = None, annotate = False)

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

    
# Get data about cluster properties, global properites, shape properties energy, and committor and return a single array and histogram test end points
# ----------------------------------------------------------------------------------------------------------------------------------------------------
   
def get_seed_props(cluster_file, global_file, alpha_file, energy_file, idx, alpha, comm_file_start, comm_file_end, liq, sol, exp_traj=300, offset=0, remove=False):
    
    # Get data
    print("# INFO: generating data from {}".format(cluster_file))
    print("# INFO: somewhat assuming shape of data file - check if properties are unexpected")
    
    data = np.genfromtxt(cluster_file) ; data[:, 0] = np.linspace(offset+1, offset+np.shape(data)[0], np.shape(data)[0]) # File, Nq6, QclP, QclR, NQP, NQR, QclM, NQM, SA, V
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

    glb = glb if global_file is None else np.genfromtxt(global_file) 

    ns  = glb[:, 1]              ; data = np.hstack((data, ns.reshape(len(ns), 1)))   # Add total number of solid atoms
    nsp = data[:, 1]/glb[:, 1]   ; data = np.hstack((data, nsp.reshape(len(nsp), 1))) # Add percentage of solid atoms in largest cluster
    n2  = glb[:, 2]              ; data = np.hstack((data, n2.reshape(len(n2), 1)))   # Add size of second-largest cluster
    n2p = glb[:, 2]/data[:, 1]   ; data = np.hstack((data, n2p.reshape(len(n2p), 1))) # Add percentage of atoms in second-largest cluster compared to largest cluster

    # Committor properties
    # ---------------------

    assert((liq or sol) is not None), "At least one cutoff is required"

    comm = Committor()

    if liq is None:
        comm.pB = lpB

    elif sol is None:
        comm.pB = spB

    finals    = np.array([[0, 0]])
    data_comm = np.zeros((len(data), 1))
    rmv       = []
    
    for i, line in enumerate(data):
        N  = str(int(line[0]))
        fl = comm_file_start+N+comm_file_end
        try:
            ends   = np.genfromtxt(fl)
            trajs  = len(ends)
            x      = np.vstack(([line[1]]*trajs, ends)).transpose() 
            finals = np.vstack((x, finals))
            if trajs != exp_traj:
                if remove:
                    rmv.append(i)
                    print("# INFO: File {} has incorrect number of trajectories, {}. Removed.".format(fl, trajs))
                else:
                    print("# INFO: File {} has incorrect number of trajectories {}".format(fl, trajs))
            data_comm[i] = comm.pB(ends, sol, liq, trajs)
        except FileNotFoundError:
            print('NOT FOUND')
            if remove:
                rmv.append(i)
                print("# WARNING:  File {} not found. Entry deleted.".format(fl))
            else:
                print("# WARNING:  File {} not found. Committor set to 0 - this may cause errors when plotting committor functions.".format(fl))

    finals = finals[:-1, :]
    
    data = np.hstack((data, data_comm))
    data = np.delete(data, rmv, axis=0)
    
    return data, finals


# Plot scatter plot of starting OP against end point
# --------------------------------------------------

def plot_ends(n_ends, s_ends, OPmean, OPdiff, ncol, scol, nmark, smark, nlab, slab, sol, liq, savename=None):

    n_end = n_ends[np.where((n_ends[:,0]-OPmean)<=OPdiff)[0], :]
    s_end = s_ends[np.where((s_ends[:,0]-OPmean)<=OPdiff)[0], :]
    
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey='all')
    fig.subplots_adjust(left=0.08, right=0.9975, bottom=0.16, top=0.9975, wspace=0, hspace=0)

    ax[0].scatter(n_end[:, 0], n_end[:, 1], color=ncol, marker=nmark, alpha=0.25, label=nlab)
    ax[1].scatter(s_end[:, 0], s_end[:, 1], color=scol, marker=smark, alpha=0.25, label=slab) 


    if sol is not None:
        ax[0].plot([OPmean-OPdiff, OPmean+OPdiff], [sol, sol], 'k--', linewidth=2.5)
        ax[1].plot([OPmean-OPdiff, OPmean+OPdiff], [sol, sol], 'k--', linewidth=2.5)
    if liq is not None:
        ax[0].plot([OPmean-OPdiff, OPmean+OPdiff], [liq, liq], 'k--', linewidth=2.5)
        ax[1].plot([OPmean-OPdiff, OPmean+OPdiff], [liq, liq], 'k--', linewidth=2.5)

 
    ax[0].set_ylabel("$N_{q_6}$ (end)", size=20)
    ax[0].set_xlabel("$N_{q_6}$ (start)", size=20)
    ax[1].set_xlabel("$N_{q_6}$ (start)", size=20)
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15)
    ax[0].text(0.925, 0.5, '(a)', size=20, transform=ax[0].transAxes)
    ax[1].text(0.925, 0.5, '(b)', size=20, transform=ax[1].transAxes)
    ax[0].legend(prop={'size':13}, loc='center left')
    ax[1].legend(prop={'size':13}, loc='center left')
    
    if savename != None:
        plt.savefig(savename, dpi=450)
        
    plt.show()
    return None

# Get bin edges for two different properties
# ------------------------------------------

def get_bins(natrl, synth, Nbins):
    mn = np.min(natrl) if np.min(natrl) < np.min(synth) else np.min(synth)
    mx = np.max(natrl) if np.max(natrl) > np.max(synth) else np.max(synth)
    bn = np.linspace(mn, mx, Nbins+1)
    return bn


# Get bin edges for four different properties
# --------------------------------------------

def get_mult_bins(natrl, synth, d3, d4, Nbins):
    mns = [] ; mxs = []
    for data in [natrl, synth, d3, d4]:
        mns.append(np.min(data)) ; mxs.append(np.max(data))
    mn = np.min(mns) ; mx = np.max(mxs)
    bn = np.linspace(mn, mx, Nbins+1)
    return bn


# Plot stair histograms of defined properties
# -------------------------------------------

def make_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, op1, op2=None, op3=None, KS=False, savename=None):
    ops = [op1]
    bns = get_bins(n[:, idx[op1]], s[:, idx[op1]], Nbins)

    ni, bns = np.histogram(n[:, idx[op1]], bns)
    si, bns = np.histogram(s[:, idx[op1]], bns)
    
    if (op2==None and op3==None):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        fig.subplots_adjust(left=0.13, right=0.9975, bottom=0.16, top=0.9975, wspace=0, hspace=0)

        ax.stairs(ni, edges=bns, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
        ax.stairs(si, edges=bns, color=scol, alpha=0.8, label=slab, linewidth=2.5)

        if KS:
            print("# INFO: Kolmogorov-Smirnov statistic for {}  is {}".format(op1, stats.ks_2samp(n[:, idx[op1]], s[:, idx[op1]]).pvalue))
        
        ax.set_ylabel("Count", size=20)
        ax.set_xlabel(labs[op1], size=20)
        ax.tick_params(labelsize=15)
        ax.legend(loc='upper left', prop={'size':13})
        
    else:
        
        if (op2==None and op3!=None) or (op2!=None and op3==None):
            fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey='all')
            fig.subplots_adjust(left=0.0625, right=0.9975, bottom=0.16, top=0.9975, wspace=0, hspace=0)
            opn = op2 if op3 is None else op3
            ops.append(opn)

        else:
            fig, ax = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey='all')
            fig.subplots_adjust(left=0.065, right=0.9975, bottom=0.16, top=0.9975, wspace=0, hspace=0)
            ops.append(op2) ; ops.append(op3)

        ax[0].stairs(ni, edges=bns, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
        ax[0].stairs(si, edges=bns, color=scol, alpha=0.8, label=slab, linewidth=2.5)
        ax[0].set_ylabel("Count", size=20)
        ax[0].set_xlabel(labs[op1], size=20)
        ax[0].tick_params(labelsize=15)
        
        if KS:
            print("# INFO: Kolmogorov-Smirnov statistic for {} is {}".format(op1, stats.ks_2samp(n[:, idx[op1]], s[:, idx[op1]]).pvalue))

        for i in range(1, len(ops)):
            bns = get_bins(n[:, idx[ops[i]]], s[:, idx[ops[i]]], Nbins)
            
            ni, bns = np.histogram(n[:, idx[ops[i]]], bns)
            si, bns = np.histogram(s[:, idx[ops[i]]], bns)
        
            ax[i].stairs(ni, edges=bns, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
            ax[i].stairs(si, edges=bns, color=scol, alpha=0.8, label=slab, linewidth=2.5)        
            ax[i].set_xlabel(labs[ops[i]], size=20)
            ax[i].tick_params(labelsize=15)

            if KS:
                print("# INFO: Kolmogorov-Smirnov statistic for {} is {}".format(ops[i], stats.ks_2samp(n[:, idx[ops[i]]], s[:, idx[ops[i]]]).pvalue))
            
        if len(ops)==2:
            ax[0].text(0.925, 0.925, '(a)', size=20, transform=ax[0].transAxes)
            ax[1].text(0.65, 0.925, '(b)', size=20, transform=ax[1].transAxes)
            ax[1].legend(prop={'size':13})

        else:
            ax[0].text(0.9, 0.925, '(a)', size=20, transform=ax[0].transAxes)
            ax[1].text(0.9, 0.925, '(b)', size=20, transform=ax[1].transAxes)
            ax[2].text(0.5, 0.925, '(c)', size=20, transform=ax[2].transAxes)
            ax[2].legend(prop={'size':13})

            
    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None


# Plot stair histograms of defined properties at 2 critical points
# -----------------------------------------------------------------

def crit_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, crit1, critop, op1, op2, op3, op4, op5, crit2=None, KS=False, savename=None):

    nc1 = n[np.where(n[:, idx[critop]]==crit1)[0], :]
    sc1 = s[np.where(s[:, idx[critop]]==crit1)[0], :]

    print('# INFO: for critical point {0}={1}, there are {2} {3} files and {4} {5} files.'.format(critop, crit1, len(nc1), nlab, len(sc1), slab))

    ops = [op1, op2, op3, op4, op5]
    
    if crit2==None:
        fig, ax = plt.subplots(1, 5, figsize=(12.8, 4.8))
        fig.subplots_adjust(left=0.13, right=0.9975, bottom=0.16, top=0.9975, wspace=0, hspace=0)
        print("1 crit - subplots and labels adjust")

        for i, op in enumerate(ops):
            bns = get_bins(nc1[:, idx[op]], sc1[:, idx[op]], Nbins)

            ni, bns = np.histogram(nc1[:, idx[op]], bns)
            si, bns = np.histogram(sc1[:, idx[op]], bns)
    
            ax[i].stairs(ni, edges=bns, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
            ax[i].stairs(si, edges=bns, color=scol, alpha=0.8, label=slab, linewidth=2.5)

            if KS:
                print("# INFO: Kolmogorov-Smirnov statistic for {} is {}".format(op, stats.ks_2samp(nc1[:, idx[op]], sc1[:, idx[op]]).pvalue))
        

            ax[i].set_xlabel(labs[op], size=20)
            ax[i].tick_params(labelsize=15)

        ax[0].set_ylabel("Count", size=20)
        ax[-1].legend(loc='upper left', prop={'size':13})

        ax[0].text(0.925, 0.925, '(a)', size=20, transform=ax[0].transAxes)
        ax[1].text(0.925, 0.925, '(b)', size=20, transform=ax[1].transAxes)
        ax[2].text(0.925, 0.925, '(c)', size=20, transform=ax[1].transAxes)
        ax[3].text(0.925, 0.925, '(d)', size=20, transform=ax[1].transAxes)
        ax[4].text(0.55, 0.925, '(e)', size=20, transform=ax[1].transAxes)

        
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12.8, 4.8), sharey='row', sharex='col')
        fig.subplots_adjust(left=0.055, right=0.9975, bottom=0.15, top=0.9975, wspace=0, hspace=0)
        
        nc2 = n[np.where(n[:, idx[critop]]==crit2)[0], :]
        sc2 = s[np.where(s[:, idx[critop]]==crit2)[0], :]

        
        print('# INFO: for critical point {0}={1}, there are {2} {3} files and {4} {5} files.'.format(critop, crit2, len(nc2), nlab, len(sc2), slab))

        for i, op in enumerate(ops):
            bns = get_mult_bins(nc1[:, idx[op]], sc1[:, idx[op]], nc2[:, idx[op]], sc2[:, idx[op]], Nbins)

            ni, bns = np.histogram(nc1[:, idx[op]], bns)
            si, bns = np.histogram(sc1[:, idx[op]], bns)
    
            ax[0][i].stairs(ni, edges=bns, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
            ax[0][i].stairs(si, edges=bns, color=scol, alpha=0.8, label=slab, linewidth=2.5)

            ni, bns = np.histogram(nc2[:, idx[op]], bns)
            si, bns = np.histogram(sc2[:, idx[op]], bns)
            
            ax[1][i].stairs(ni, edges=bns, color=ncol, alpha=0.8, label=nlab, linewidth=2.5)
            ax[1][i].stairs(si, edges=bns, color=scol, alpha=0.8, label=slab, linewidth=2.5)

            if KS:
                print("# INFO: Kolmogorov-Smirnov statistic for {} at {} is {}".format(op, crit1, stats.ks_2samp(nc1[:, idx[op]], sc1[:, idx[op]]).pvalue))
                print("# INFO: Kolmogorov-Smirnov statistic for {} at {} is {}".format(op, crit2, stats.ks_2samp(nc2[:, idx[op]], sc2[:, idx[op]]).pvalue))
        

            ax[0][i].set_xlabel(labs[op], size=20)
            ax[0][i].tick_params(labelsize=15)
            ax[1][i].set_xlabel(labs[op], size=20)
            ax[1][i].tick_params(labelsize=15)
            
        ax[0][0].set_ylabel("Count", size=20)
        ax[1][0].set_ylabel("Count", size=20)
        ax[1][-1].legend(loc='upper left', prop={'size':13})
        

        ax[0][0].text(0.05, 0.85, '(a)', size=20, transform=ax[0][0].transAxes)
        ax[0][1].text(0.85, 0.85, '(b)', size=20, transform=ax[0][1].transAxes)
        ax[0][2].text(0.85, 0.85, '(c)', size=20, transform=ax[0][2].transAxes)
        ax[0][3].text(0.85, 0.85, '(d)', size=20, transform=ax[0][3].transAxes)
        ax[0][4].text(0.85, 0.85, '(e)', size=20, transform=ax[0][4].transAxes)
                              
        ax[1][0].text(0.85, 0.85, '(f)', size=20, transform=ax[1][0].transAxes)
        ax[1][1].text(0.85, 0.85, '(g)', size=20, transform=ax[1][1].transAxes)
        ax[1][2].text(0.85, 0.85, '(h)', size=20, transform=ax[1][2].transAxes)
        ax[1][3].text(0.85, 0.85, '(i)', size=20, transform=ax[1][3].transAxes)
        ax[1][4].text(0.85, 0.85, '(j)', size=20, transform=ax[1][4].transAxes)

        
    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None



# Crystallinity Scatter Plots
# ---------------------------

def crystallinity_scatter(n, s, ncol, scol, nmark, smark, nlab, slab, idx, labs, savename=None):

    fig, ax = plt.subplots(1, 3, figsize=(12.8, 4.8))
    fig.subplots_adjust(left=0.07, right=0.9975, bottom=0.15, top=0.9975, wspace=0.3, hspace=0)

    ax[0].scatter(n[:, idx['QclP']], n[:, idx['QclM']], color=ncol, marker=nmark, alpha=0.25, label=nlab)
    ax[0].scatter(s[:, idx['QclP']], s[:, idx['QclM']], color=scol, marker=smark, alpha=0.25, label=slab)
    ax[1].scatter(n[:, idx['QclP']], n[:, idx['QclR']], color=ncol, marker=nmark, alpha=0.25, label=nlab)
    ax[1].scatter(s[:, idx['QclP']], s[:, idx['QclR']], color=scol, marker=smark, alpha=0.25, label=slab)
    ax[2].scatter(n[:, idx['QclM']], n[:, idx['QclR']], color=ncol, marker=nmark, alpha=0.25, label=nlab)
    ax[2].scatter(s[:, idx['QclM']], s[:, idx['QclR']], color=scol, marker=smark, alpha=0.25, label=slab)

    ax[0].set_ylabel(labs['QclM'], size=20)
    ax[0].set_xlabel(labs['QclP'], size=20)
    ax[1].set_ylabel(labs['QclR'], size=20)
    ax[1].set_xlabel(labs['QclP'], size=20)
    ax[2].set_ylabel(labs['QclR'], size=20)
    ax[2].set_xlabel(labs['QclM'], size=20)
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15)
    ax[2].tick_params(labelsize=15)
    ax[0].text(0.9, 0.5, '(a)', size=20, transform=ax[0].transAxes)
    ax[1].text(0.9, 0.5, '(b)', size=20, transform=ax[1].transAxes)
    ax[2].text(0.9, 0.5, '(c)', size=20, transform=ax[2].transAxes)
    ax[2].legend(prop={'size':13})

    if savename != None:
        plt.savefig(savename, dpi=450)
        
    plt.show()
    return None


# Scatter plots of committor vs OP 
# -------------------------------- 

def committor_scatter(n, s, ncol, scol, nmark, smark, nlab, slab, idx, labs, op1, op2=None, data3=None, data4=None, savename=None):

    # data3 and data4 are there to allow size discrimination for op != Nq6
    # if present should be a tuple/list of [data, col, marker, lab]
    
    if op2==None:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        fig.subplots_adjust(left=0.125, right=0.9975, bottom=0.135, top=0.9975, wspace=0, hspace=0)
        
        ax.scatter(n[:, idx[op1]], n[:, idx['pB']], color=ncol, marker=nmark, alpha=0.25, label=nlab)
        ax.scatter(s[:, idx[op1]], s[:, idx['pB']], color=scol, marker=smark, alpha=0.25, label=slab)

        if data3 != None:
            dat = data3[0] ; col = data3[1] ;  mark = data3[2] ; lab = data3[3]
            ax.scatter(dat[:, idx[op1]], dat[:, idx['pB']], color=col, marker=mark, alpha=0.25, label=lab)
        if data4 != None:
            dat = data4[0] ; col = data4[1] ;  mark = data4[2] ; lab = data4[3]
            ax.scatter(dat[:, idx[op1]], dat[:, idx['pB']], color=col, marker=mark, alpha=0.25, label=lab)
        
        ax.set_ylabel(labs['pB'], size=20)
        ax.set_xlabel(labs[op1], size=20)
        ax.tick_params(labelsize=15)
        ax.legend(prop={'size':13})
                   
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey='all')
        fig.subplots_adjust(left=0.0675, right=0.9975, bottom=0.135, top=0.975, wspace=0, hspace=0)

        ax[0].scatter(n[:, idx[op1]], n[:, idx['pB']], color=ncol, marker=nmark, alpha=0.25, label=nlab)
        ax[0].scatter(s[:, idx[op1]], s[:, idx['pB']], color=scol, marker=smark, alpha=0.25, label=slab)

        ax[1].scatter(n[:, idx[op2]], n[:, idx['pB']], color=ncol, marker=nmark, alpha=0.25, label=nlab)
        ax[1].scatter(s[:, idx[op2]], s[:, idx['pB']], color=scol, marker=smark, alpha=0.25, label=slab)

        if data3 != None:
            dat = data3[0] ; col = data3[1] ;  mark = data3[2] ; lab = data3[3]
            ax[0].scatter(dat[:, idx[op1]], dat[:, idx['pB']], color=col, marker=mark, alpha=0.25, label=lab)
            ax[1].scatter(dat[:, idx[op2]], dat[:, idx['pB']], color=col, marker=mark, alpha=0.25, label=lab)
        if data4 != None:
            dat = data4[0] ; col = data4[1] ;  mark = data4[2] ; lab = data4[3]
            ax[0].scatter(dat[:, idx[op1]], dat[:, idx['pB']], color=col, marker=mark, alpha=0.25, label=lab)
            ax[1].scatter(dat[:, idx[op2]], dat[:, idx['pB']], color=col, marker=mark, alpha=0.25, label=lab)

        
        ax[0].set_ylabel(labs['pB'], size=20)
        ax[0].set_xlabel(labs[op1], size=20)
        ax[1].set_xlabel(labs[op2], size=20)
        ax[0].tick_params(labelsize=15)
        ax[1].tick_params(labelsize=15)
        ax[1].legend(prop={'size':13}, loc='lower right')
        ax[0].text(0.025, 0.925, '(a)', size=20, transform=ax[0].transAxes)                                                                                                  
        ax[1].text(0.025, 0.925, '(b)', size=20, transform=ax[1].transAxes)  
            
    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None


# Make full committor histogram
# -----------------------------

def committor_stairs(n1, s1, ncol, scol, nlab, slab, Nbins, lab1, idx, n2=[], s2=[], lab2=None, KS=False, savename=None, lab=False):

    bns = np.linspace(0, 1, Nbins+1)

    nwt = [1.0/len(n1)]*len(n1)
    swt = [1.0/len(s1)]*len(s1)

    nh, bns = np.histogram(n1[:, idx['pB']], bns, weights=nwt)
    sh, bns = np.histogram(s1[:, idx['pB']], bns, weights=swt)

    if len(n2)==0 and len(s2)==0:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        fig.subplots_adjust(left=0.13, right=0.9975, bottom=0.16, top=0.9975, wspace=0, hspace=0)

        ax.stairs(nh, edges=bns, color=ncol, alpha=0.8, label=nlab+lab1, linewidth=2.5)
        ax.stairs(sh, edges=bns, color=scol, alpha=0.8, label=slab+lab1, linewidth=2.5)

        ax.set_ylabel("$P(p_B)$", size=20)
        ax.set_xlabel("$p_B$", size=20)
        ax.tick_params(labelsize=15)
        ax.legend(prop={'size':13})

        if lab:
            ax.text(0.025, 0.925, '(c)', size=20, transform=ax.transAxes)
        
        if KS:
            print("# INFO: Kolmogorov-Smirnov statistic for {} committors is {}".format(lab1, stats.ks_2samp(n1[:, idx['pB']], s1[:, idx['pB']]).pvalue))
                    

    else:
        fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey='all')
        fig.subplots_adjust(left=0.0675, right=0.9975, bottom=0.135, top=0.975, wspace=0, hspace=0)

        ax[0].stairs(nh, edges=bns, color=ncol, alpha=0.8, label=nlab+lab1, linewidth=2.5)
        ax[0].stairs(sh, edges=bns, color=scol, alpha=0.8, label=slab++lab1, linewidth=2.5)
        
        nwt = [1.0/len(n2)]*len(n2)
        swt = [1.0/len(s2)]*len(s2)

        nh, bns = np.histogram(n2[:, idx['pB']], bns, weights=nwt)
        sh, bns = np.histogram(s2[:, idx['pB']], bns, weights=swt)
        
        ax[1].stairs(nh, edges=bns, color=ncol, alpha=0.8, label=nlab+lab2, linewidth=2.5)
        ax[1].stairs(sh, edges=bns, color=scol, alpha=0.8, label=slab+lab2, linewidth=2.5)

        ax[0].set_ylabel("$P(p_B)$", size=20)
        ax[0].set_xlabel("$p_B$", size=20)
        ax[1].set_xlabel("$p_B$", size=20)
        ax[0].tick_params(labelsize=15)
        ax[1].tick_params(labelsize=15)
        ax[1].legend(prop={'size':13})
        ax[0].text(0.925, 0.925, '(a)', size=20, transform=ax[0].transAxes)                                                                                                  
        ax[1].text(0.025, 0.925, '(b)', size=20, transform=ax[1].transAxes)

        if KS:
            print("# INFO: Kolmogorov-Smirnov statistic for {} committors is {}".format(lab1, stats.ks_2samp(n1[:, idx['pB']], s1[:, idx['pB']]).pvalue))
            print("# INFO: Kolmogorov-Smirnov statistic for {} committors is {}".format(lab2, stats.ks_2samp(n2[:, idx['pB']], s2[:, idx['pB']]).pvalue))
        
            
    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None



# Distribution of properties as f(Nq6)
# -------------------------------------

def distribution_scatter(n, s, Nbins, ncol, scol, nmark, smark, nlab, slab, idx, labs, opdis, op1, op2, op3, op4, op5, savename=None, data3=None, data4=None):

    fig, ax = plt.subplots(2, 5, figsize=(12.8, 4.8), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.0675, right=0.795, bottom=0.16, top=0.9975, wspace=0, hspace=0)
    
    ops = [op1, op2, op3, op4, op5]

    distributions = []
    
    if data3==None and data4==None:
        for op in ops: 
            distributions.append(get_bins(n[:, idx[op]], s[:, idx[op]], Nbins))
        plot_props = zip([n, s], [ncol, scol], [nmark, smark], [nlab, slab])
    else:
        if data3!=None and data4==None:
            dat = data3[0] ; dat2 = dat                
            plot_props = zip([n, s, dat], [ncol, scol, data3[1]], [nmark, smark, data3[2]], [nlab, slab, data3[3]])
        elif data3==None and data4!=None:
            dat = data4[0] ; dat2 = dat
            plot_props = zip([n, s, dat], [ncol, scol, data4[1]], [nmark, smark, data4[2]], [nlab, slab, data4[3]])
        else:
            dat = data3[0] ; dat2 = data4[0]
            plot_props = zip([n, s, dat, dat2], [ncol, scol, data3[1], data4[1]], [nmark, smark, data3[2], data4[2]], [nlab, slab, data3[3], data4[3]])
        for op in ops:
            distributions.append(get_mult_bins(n[:, idx[op]], s[:, idx[op]], dat[:, idx[op]], dat2[:, idx[op]], Nbins))
    

    # For each OP, want to plot a scatter plot as a function of opdis, and then a normal histogram

    for p in plot_props:
        for i, op in enumerate(ops):
            ax[0][i].scatter(p[0][:, idx[op]], p[0][:, idx[opdis]], color=p[1], marker=p[2], alpha=0.25, label=p[3])
            dh, bns = np.histogram(p[0][:, idx[op]], distributions[i])
            ax[1][i].stairs(dh, edges=bns, color=p[1], alpha=0.8, label=p[3], linewidth=2.5)
            

    for i, op in enumerate(ops):
        ax[1][i].set_xlabel(labs[op], size=20)
        ax[0][i].tick_params(labelsize=15)
        ax[1][i].tick_params(labelsize=15)

    ax[0][0].set_ylabel(labs[opdis], size=20)
    ax[1][0].set_ylabel("Count", size=20)
    ax[0][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':13})
    ax[1][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':13})
    
    ax[0][0].text(0.05, 0.85, '(a)', size=20, transform=ax[0][0].transAxes)
    ax[0][1].text(0.825, 0.85, '(b)', size=20, transform=ax[0][1].transAxes)
    ax[0][2].text(0.825, 0.85, '(c)', size=20, transform=ax[0][2].transAxes)
    ax[0][3].text(0.825, 0.85, '(d)', size=20, transform=ax[0][3].transAxes)
    ax[0][4].text(0.05, 0.85, '(e)', size=20, transform=ax[0][4].transAxes)
    ax[1][0].text(0.85, 0.85, '(f)', size=20, transform=ax[1][0].transAxes)
    ax[1][1].text(0.825, 0.85, '(g)', size=20, transform=ax[1][1].transAxes)
    ax[1][2].text(0.825, 0.85, '(h)', size=20, transform=ax[1][2].transAxes)
    ax[1][3].text(0.85, 0.85, '(i)', size=20,  transform=ax[1][3].transAxes)
    ax[1][4].text(0.85, 0.85, '(j)', size=20,  transform=ax[1][4].transAxes)

    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None


def mult_stairs(n, s, Nbins, ncol, scol, nlab, slab, idx, labs, op1, op2, op3, op4, op5, savename=None, data3=None, data4=None):

    fig, ax = plt.subplots(1, 5, figsize=(19.2, 4.8), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.16, top=0.9975, wspace=0, hspace=0)
    
    ops = [op1, op2, op3, op4, op5]

    distributions = []
    
    if data3==None and data4==None:
        for op in ops: 
            distributions.append(get_bins(n[:, idx[op]], s[:, idx[op]], Nbins))
        plot_props = zip([n, s], [ncol, scol], [nmark, smark], [nlab, slab])
    else:
        if data3!=None and data4==None:
            dat = data3[0] ; dat2 = dat                
            plot_props = zip([n, s, dat], [ncol, scol, data3[1]], [nmark, smark, data3[2]], [nlab, slab, data3[3]])
        elif data3==None and data4!=None:
            dat = data4[0] ; dat2 = dat
            plot_props = zip([n, s, dat], [ncol, scol, data4[1]], [nmark, smark, data4[2]], [nlab, slab, data4[3]])
        else:
            dat = data3[0] ; dat2 = data4[0]
            plot_props = zip([n, s, dat, dat2], [ncol, scol, data3[1], data4[1]], [nmark, smark, data3[2], data4[2]], [nlab, slab, data3[3], data4[3]])
        for op in ops:
            distributions.append(get_mult_bins(n[:, idx[op]], s[:, idx[op]], dat[:, idx[op]], dat2[:, idx[op]], Nbins))
    

    # For each OP, want to plot a scatter plot as a function of opdis, and then a normal histogram

    for p in plot_props:
        for i, op in enumerate(ops):
            dh, bns = np.histogram(p[0][:, idx[op]], distributions[i])
            ax[i].stairs(dh, edges=bns, color=p[1], alpha=0.8, label=p[3], linewidth=2.5)
            ax[i].set_xlabel(labs[op], size=20)
            ax[i].tick_params(labelsize=15)

    ax[0].set_ylabel("Count", size=20)
    
    ax[-1].legend(prop={'size':13}, loc='upper left')

    ax[0].text(0.9, 0.925, '(a)', size=20, transform=ax[0].transAxes)
    ax[1].text(0.9, 0.925, '(b)', size=20, transform=ax[1].transAxes)
    ax[2].text(0.9, 0.925, '(c)', size=20, transform=ax[2].transAxes)
    ax[3].text(0.9, 0.925, '(d)', size=20, transform=ax[3].transAxes)
    ax[4].text(0.9, 0.925, '(e)', size=20, transform=ax[4].transAxes)

    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None


def Nq6_comp(data, histcol, bins, upper, idx, labs, op1, op2, op3, op4, op5, savename=None):

    fig, ax = plt.subplots(1, 5, figsize=(19.2, 4.8), sharex='col', sharey='row')
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.16, top=0.99, wspace=0, hspace=0)
    
    ops = [op1, op2, op3, op4, op5]
    
    range1 = data[np.where(data[:, 1] < upper[0])[0], :]
    range2 = data[np.where((data[:, 1] >= upper[0])&(data[:, 1] < upper[1]))[0], :]
    range3 = data[np.where((data[:, 1] >= upper[1])&(data[:, 1] <= upper[2]))[0], :]
        
    for i, op in enumerate(ops):
        ax[i].hist(data[:, idx[op]], bins[i], weights=[1.0/len(data)]*len(data), color=histcol, alpha=0.2, label=r'$'+str(int(np.min(data[:, 1])))+r' \leq N_{q_6} \leq'+str(int(np.max(data[:, 1])))+r'$')
        ax[i].set_xlabel(labs[op], size=20)
        ax[i].tick_params(labelsize=15)
        
        dh, bns = np.histogram(range1[:, idx[op]], bins[i], weights=[1/len(range1)]*len(range1)) 
        ax[i].stairs(dh, edges=bns, color='lightgreen', alpha=0.8, label=r'$'+str(int(np.min(data[:, 1])))+r' \leq N_{q_6} <'+str(upper[0])+r'$', linewidth=2.5)
        dh, bns = np.histogram(range2[:, idx[op]], bins[i], weights=[1/len(range2)]*len(range2)) 
        ax[i].stairs(dh, edges=bns, color='limegreen', alpha=0.8, label=r'$'+str(upper[0])+r' \leq N_{q_6} <'+str(upper[1])+r'$', linewidth=2.5)  
        dh, bns = np.histogram(range3[:, idx[op]], bins[i], weights=[1/len(range3)]*len(range3)) 
        ax[i].stairs(dh, edges=bns, color='darkgreen', alpha=0.8, label=r'$'+str(upper[1])+r' \leq N_{q_6} \leq '+str(upper[2])+r'$', linewidth=2.5)
        
    ax[0].set_ylabel("Weight", size=20)
    
    ax[-1].legend(prop={'size':13}, loc='upper left')

    ax[0].text(0.875, 0.925, '(a)', size=20, transform=ax[0].transAxes)
    ax[1].text(0.875, 0.925, '(b)', size=20, transform=ax[1].transAxes)
    ax[2].text(0.875, 0.925, '(c)', size=20, transform=ax[2].transAxes)
    ax[3].text(0.875, 0.925, '(d)', size=20, transform=ax[3].transAxes)
    ax[4].text(0.875, 0.925, '(e)', size=20, transform=ax[4].transAxes)

    if savename!=None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None



# =============== #
# Define heatmaps #
# =============== #

def heatmap(n, s, x_op, y_op, z_op, idx, x_bins, y_bins, labs, nlab, slab, z_lims = None, differential = True, yticks = None, xticks = None,
            savename = None, shift = None, annotate = False):


    ''' Creates a heatmap of unbiased and constructed OPs.

        Parameters
        ----------
        n, s                 : the two distributions 
        x_op, y_op, z_op     : the x/y/z data IDs (z_op = None gives count)
        idx                  : mapping of IDs to data locations
        x_bins, y_bins       : the number of bins in the x/y direction
        labs                 : mapping of IDs to labels
        z_lims (opt)         : the limits of the colourbar
        differential (opt)   : produce a plot of the difference between natural/synthetic heatmap (True/False)
        xticks, yticks (opt) : location of the ticks on the x/y axes
        f (opt)              : prefix to use when saving file 
        shift (opt)          : shift LHS of graph to prevent truncation of axis label
        annotate (opt)       : annotate heatmap with number of seeds (True/False). '''
    

    shift = shift if shift is not None else 0.075
    count = False
    
    natrl_grid = np.zeros((y_bins, x_bins)) ; natrl_count = np.zeros((y_bins, x_bins))
    synth_grid = np.zeros((y_bins, x_bins)) ; synth_count = np.zeros((y_bins, x_bins))

    nx_data = n[:, idx[x_op]] ; ny_data = n[:, idx[y_op]] 
    sx_data = s[:, idx[x_op]] ; sy_data = s[:, idx[y_op]]

    if z_op != None:
        nz_data = n[:, idx[z_op]]  ; sz_data = s[:, idx[z_op]]
    
    binx = get_bins(nx_data, sx_data, x_bins)
    biny = get_bins(ny_data, sy_data, y_bins)

    for i in range(len(nx_data)):
        nx = np.where(nx_data[i] >= binx)[0][-1]
        ny = np.where(ny_data[i] >= biny)[0][-1]
        nx = x_bins - 1 if nx == x_bins else nx
        ny = y_bins - 1 if ny == y_bins else ny
        if z_op != None:
            natrl_grid[ny, nx]  += nz_data[i]
        natrl_count[ny, nx] += 1 

    for i in range(len(sx_data)):
        nx = np.where(sx_data[i] >= binx)[0][-1]
        ny = np.where(sy_data[i] >= biny)[0][-1]
        nx = x_bins - 1 if nx == x_bins else nx
        ny = y_bins - 1 if ny == y_bins else ny
        if z_op != None:
            synth_grid[ny, nx]  += sz_data[i]
        synth_count[ny, nx] += 1 

    if z_op == None:
        av_natrl = natrl_count  ; nz_data = natrl_count
        av_synth = synth_count  ; sz_data = synth_count
        count    = True
        z_label  = "Count"
        
    else:
        av_natrl = np.ma.masked_array(natrl_grid, mask=natrl_count==0)
        av_natrl = av_natrl/natrl_count
        av_synth = np.ma.masked_array(synth_grid, mask=synth_count==0)
        av_synth = av_synth/synth_count
        z_label  = r'$\langle '+labs[z_op][1:-1]+r' \rangle$'
        
    if z_lims is None:
        mnz = np.min(nz_data) if np.min(nz_data) < np.min(sz_data) else np.min(sz_data)
        mxz = np.max(nz_data) if np.max(nz_data) > np.max(sz_data) else	np.max(sz_data)
    else:
        mnz = z_lims[0] ; mxz = z_lims[1]

        
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey='all')
    fig.subplots_adjust(left=shift, right=1.05, bottom=0.15, top=0.925, wspace=0, hspace=0)
    ax[0].pcolormesh(av_natrl, cmap=cm.autumn, vmin=mnz, vmax=mxz)
    ax[1].pcolormesh(av_synth, cmap=cm.autumn, vmin=mnz, vmax=mxz)

    norm = colors.Normalize(vmin=mnz, vmax=mxz)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.autumn), orientation='vertical', ax=ax, pad=0.01)
    cb.set_label(label=z_label, size=20)
    cb.ax.yaxis.set_tick_params(labelsize=15)
    
    if yticks is None:
        ylocs = ax[0].get_yticks()
    else:
        ylocs = yticks[:]
        ylocs = y_bins*(ylocs - biny[0])/(biny[-1] - biny[0])

    
    if xticks is None:
        xlocs = ax[0].get_xticks()
    else:
        xlocs = xticks[:]
        xlocs = x_bins*(xlocs - binx[0])/(binx[-1] - binx[0])

    if (y_op=='Nq6') or (y_op=="sol") or (y_op=="Nq6-2"):
        ylabs = ["{:}".format(int(biny[int(i)])) for i in ylocs]
    else:
        ylabs = ["{:.2f}".format(biny[int(i)]) for i in ylocs]
    if (x_op=="Nq6") or (x_op=="sol") or (x_op=="Nq6-2"):
        xlabs = ["{:}".format(int(binx[int(i)])) for i in xlocs]
    else:
        xlabs = ["{:.2f}".format(binx[int(i)]) for i in xlocs]

    kw = dict(horizontalalignment="center", verticalalignment="center")

    if annotate:
        xscale = 1.0/x_bins 
        yscale = 1.0/y_bins
        for i in range(x_bins):
            tx = (i+0.5)*xscale
            for j in range(y_bins):
                ty = (j+0.5)*yscale
                if (count) or (natrl_count[j, i] !=0):
                    ax[0].text(tx, ty, "{:}".format(int(natrl_count[j, i])), size=20, transform=ax[0].transAxes, **kw)
                if (count) or (synth_count[j, i] !=0):
                    ax[1].text(tx, ty, "{:}".format(int(synth_count[j, i])), size=20, transform=ax[1].transAxes, **kw)
                
        
    ax[0].set_xlabel(labs[x_op], size=20)
    ax[0].set_ylabel(labs[y_op], size=20)
    ax[1].set_xlabel(labs[x_op], size=20)
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15)
    ax[0].set_yticks(ylocs, ylabs)
    ax[0].set_xticks(xlocs, xlabs)
    ax[1].set_xticks(xlocs, xlabs)
    ax[0].set_title(nlab, size=20)
    ax[1].set_title(slab, size=20)
    if savename != None:
        plt.savefig(savename+"_heatmap.png", dpi=450)
        
    plt.show()

    # Differential heatmap
    # --------------------

    if differential:

        shift = shift*2
        
        D_av = av_synth - av_natrl
        DV   = np.max(abs(D_av))
        
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), sharey='all')
        fig.subplots_adjust(left=shift, right=0.9, bottom=0.15, top=0.95, wspace=0, hspace=0)
        pt = ax.pcolormesh(D_av, cmap=cm.seismic, vmin=-1*DV, vmax=DV)
        cb = fig.colorbar(pt, orientation='vertical', ax=ax, pad=0.01)
        if z_op ==None:
            cb.set_label(label=r'$\Delta$ Count', size=20)
        else:
            cb.set_label(label=r'$\Delta'+z_label[1:], size=20)
        cb.ax.yaxis.set_tick_params(labelsize=15)
        ax.patch.set(hatch='xx', edgecolor='black')

        ax.set_xlabel(labs[x_op], size=20)
        ax.set_ylabel(labs[y_op], size=20)
        ax.set_yticks(ylocs, ylabs)
        ax.set_xticks(xlocs, xlabs)
        ax.tick_params(labelsize=15)
        if savename != None:
            plt.savefig("D_"+savename+"_heatmap.png", dpi=450)
        plt.show()

    return None



def mult_heatmap(n, s, d3, d4, x_op, y_ops, idx, x_bins, y_bins, labs, nlab, slab, lab3, lab4, xticks, yticks, savename = None, shift = None, annotate = False):


    ''' Creates a heatmap of Nq6 and OPs.

        Parameters
        ----------
        n, s, d3, d4         : the two distributions 
        x_op, y_ops          : the x/y data IDs (z_op = normalised count)
        idx                  : mapping of IDs to data locations
        x_bins, y_bins       : the number of bins in the x/y direction
        labs                 : mapping of IDs to labels
        xticks, yticks (opt) : location of the ticks on the x/y axes
        f (opt)              : prefix to use when saving file 
        shift (opt)          : shift LHS of graph to prevent truncation of axis label
        annotate (opt)       : annotate heatmap with number of seeds (True/False). '''
    

    shift = shift if shift is not None else 0.075

    count1 = np.zeros((y_bins, x_bins))
    count2 = np.zeros((y_bins, x_bins))
    count3 = np.zeros((y_bins, x_bins))
    count4 = np.zeros((y_bins, x_bins))

    x_data1 = n[:, idx[x_op]]
    x_data2 = s[:, idx[x_op]]
    x_data3 = d3[:, idx[x_op]]
    x_data4 = d4[:, idx[x_op]]        
    
    binx = get_mult_bins(x_data1, x_data2, x_data3, x_data4, x_bins)

    fig, ax = plt.subplots(len(y_ops), 4, figsize=(12.8, 9.6), sharey='all')
    fig.subplots_adjust(left=shift, right=1.05, bottom=0.075, top=0.925, wspace=0, hspace=0)

    binsy = []
    
    for o, op in enumerate(y_ops):
        y_data1 = n[:, idx[op]]
        y_data2 = s[:, idx[op]]
        y_data3 = d3[:, idx[op]]
        y_data4 = d4[:, idx[op]]        

        col = 0
        
        biny = get_mult_bins(y_data1, y_data2, y_data3, y_data4, y_bins)
        binsy.append(biny)
        
        kw = dict(horizontalalignment="center", verticalalignment="center")
        xscale = 1.0/x_bins 
        yscale = 1.0/y_bins
        
        for x_data, y_data, count in zip([x_data1, x_data2, x_data3, x_data4], [y_data1, y_data2, y_data3, y_data4],
                                         [count1, count2, count3, count4]):
            for i in range(len(x_data)):
                nx = np.where(x_data[i] >= binx)[0][-1]
                ny = np.where(y_data[i] >= biny)[0][-1]
                nx = x_bins - 1 if nx == x_bins else nx
                ny = y_bins - 1 if ny == y_bins else ny
                count[ny, nx] += 1 
                

            av = np.ma.masked_array(count, mask=count==0)
            av = av/np.max(av)
            ax[o][col].pcolormesh(av, cmap=cm.autumn, vmin=0, vmax=1)
            ax[o][col].tick_params(labelsize=15)

            if annotate:
                for i in range(x_bins):
                    tx = (i+0.5)*xscale
                    for j in range(y_bins):
                        ty = (j+0.5)*yscale
                        if (count[j, i] !=0):
                            ax[o][col].text(tx, ty, "{:}".format(int(count[j, i])), size=9, transform=ax[o][col].transAxes, **kw)

            col += 1

        
    norm = colors.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.autumn), orientation='vertical', ax=ax, pad=0.01)
    cb.set_label(label="Normalised Count", size=20)
    cb.ax.yaxis.set_tick_params(labelsize=15)
   
    for i, title in enumerate([nlab, slab, lab3, lab4]):
        ax[0][i].set_title(title, size=20)

    xlocs = xticks[:]
    xlocs = x_bins*(xlocs - binx[0])/(binx[-1] - binx[0])
    if (x_op=="Nq6") or (x_op=="sol") or (x_op=="Nq6-2"):
        xlabs = ["{:}".format(int(binx[int(i)])) for i in xlocs]
    else:
        xlabs = ["{:.2f}".format(binx[int(i)]) for i in xlocs]

    for j, y_op in enumerate(y_ops):
        ylocs = yticks[j][:]
        biny  = binsy[j]
        ylocs = y_bins*(ylocs - biny[0])/(biny[-1] - biny[0])
        if (y_op=='Nq6') or (y_op=="sol") or (y_op=="Nq6-2"):
            ylabs = ["{:}".format(int(biny[int(i)])) for i in ylocs]
        else:
            ylabs = ["{:.2f}".format(biny[int(i)]) for i in ylocs]
        print(ylocs, ylabs)
        ax[j][0].set_ylabel(labs[y_op], size=20)
        ax[j][0].set_yticks(ylocs, ylabs)
    
    for i in range(4):
        ax[-1][i].set_xlabel(labs[x_op], size=20)        
        ax[-1][i].set_xticks(xlocs, xlabs)


    if savename != None:
        plt.savefig(savename, dpi=450)
        
    plt.show()

    return None

