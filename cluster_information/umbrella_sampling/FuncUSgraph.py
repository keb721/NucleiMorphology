import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import Counter
from scipy import stats

rc('text', usetex=True)

def get_data(cluster_file, energy_file, alpha_file, idx, alpha=True, US=True):
    # Get data
    print("# INFO: generating data from {}".format(cluster_file))
    print("# INFO: somewhat assuming shape of data file - check if properties are unexpected")
    
    data = np.genfromtxt(cluster_file) # File, Nq6, QclP, QclR, NQP, NQR, QclM, NQM, SA, V

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
    if US:
        print("# INFO: US also takes into account total energy")
        en  = enf[:, 2]             ; data = np.hstack((data, en.reshape(len(en), 1)))   # Add cluster energy
        ena = enf[:, 2]/data[:, 1]  ; data = np.hstack((data, ena.reshape(len(ena), 1))) # Add cluster energy per atom
        enc = enf[:, 3]             ; data = np.hstack((data, enc.reshape(len(enc), 1))) # Add cluster-cluster energy
        enp = enf[:, 3]/enf[:, 2]   ; data = np.hstack((data, enp.reshape(len(enp), 1))) # Add percentage cluster-cluster
        ent = enf[:, 1]             ; data = np.hstack((data, ent.reshape(len(ent), 1))) # Add total energy
        ecp = enf[:, 2]/enf[:, 1]   ; data = np.hstack((data, ecp.reshape(len(ecp), 1))) # Add percentage energy from cluster
        eci = enf[:, 3]/enf[:, 1]   ; data = np.hstack((data, eci.reshape(len(eci), 1))) # Add percentage energy from cluster-cluster interactions
        ein = enf[:, 3]/data[:, 1]   ; data = np.hstack((data, ein.reshape(len(ein), 1))) # Add cluster-cluster energy per atom
    else:
        en  = enf[:, 1]             ; data = np.hstack((data, en.reshape(len(en), 1)))   # Add cluster energy
        en  = enf[:, 1]/data[:, 1]  ; data = np.hstack((data, en.reshape(len(en), 1)))   # Add cluster energy per atom
        enc = enf[:, 2]             ; data = np.hstack((data, enc.reshape(len(enc), 1))) # Add cluster-cluster energy
        enp = enf[:, 2]/enf[:, 1]   ; data = np.hstack((data, enp.reshape(len(enp), 1))) # Add percentage cluster-cluster

    return data

def plot_energy(n, s, ncol, scol, nlab, slab, idx, labs, savename=None):
    fig, ax = plt.subplots(2, 4, figsize=(19.2, 9.6), sharex='all')
    fig.subplots_adjust(left=0.0625, right=0.995, top=0.975, bottom=0.075, wspace=0.28, hspace=0)

    steps = np.linspace(1, len(n), len(n))

    ax[0][0].plot(steps, n[:, idx['tEn']],  color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
    ax[0][0].plot(steps, s[:, idex['tEn']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[0][0].tick_params(labelsize=15)
    ax[0][0].set_ylabel(labs['tEn'], size=20)

    ax[0][1].plot(steps, n[:, idx['Ecl']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
    ax[0][1].plot(steps, s[:, idx['Ecl']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[0][1].tick_params(labelsize=15)
    ax[0][1].set_ylabel(labs['Ecl'], size=20)

    ax[0][2].plot(steps, n[:, idx['pEc']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
    ax[0][2].plot(steps, s[:, idx['pEc']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[0][2].tick_params(labelsize=15)
    ax[0][2].set_ylabel(labs['pEc'], size=20)

    ax[0][3].plot(steps, n[:, idx['EcN']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
    ax[0][3].plot(steps, s[:, idx['EcN']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[0][3].tick_params(labelsize=15)
    ax[0][3].set_ylabel(labs['EcN'], size=20)

    ax[1][0].plot(steps, n[:, idx['Ecl-cl']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
    ax[1][0].plot(steps, s[:, idx['Ecl-cl']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[1][0].tick_params(labelsize=15)
    ax[1][0].set_xlabel("Steps", size=20)
    ax[1][0].set_ylabel(labs['Ecl-cl'], size=20)

    ax[1][1].plot(steps, n[:, idx['Ecl-clP']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
    ax[1][1].plot(steps, s[:, idx['Ecl-clP']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[1][1].tick_params(labelsize=15)
    ax[1][1].set_xlabel("Steps", size=20)
    ax[1][1].set_ylabel(labs['Ecl-clP'], size=20)

    ax[1][2].plot(steps, n[:, idx['Ecl-clPT']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5) # percentage of total
    ax[1][2].plot(steps, s[:, idx['Ecl-clPT']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[1][2].tick_params(labelsize=15)
    ax[1][2].set_xlabel("Steps", size=20)
    ax[1][2].set_ylabel(labs['Ecl-clPT'], size=20)

    ax[1][3].plot(steps, n[:, idx['Ecl-clN']], color=ncol, label=nlab, linewidth=2.5, alpha=0.5) # per atom
    ax[1][3].plot(steps, s[:, idx['Ecl-clN']], color=scol, label=slab, linewidth=2.5, alpha=0.5)
    ax[1][3].tick_params(labelsize=15)
    ax[1][3].set_xlabel("Steps", size=20)
    ax[1][3].set_ylabel(labs['Ecl-clN'], size=20)

    if savename != None:
        plt.savefig(savename, dpi=450)
        
    plt.show()
    return None


def prop_evo(n, s, ncol, scol, nlab, slab, idx, labs, ops, savename=None):
    
    fig, ax = plt.subplots(2, 3, figsize=(19.2, 9.6), sharex='all')
    fig.subplots_adjust(left=0.05, right=0.995, top=0.975, bottom=0.075, wspace=0.175, hspace=0)
	
    steps = np.linspace(1, len(n), len(n))


    for j in range(2):
        for i in range(3):
            ax[j][i].plot(steps, n[:, idx[ops[j*3+i]]], color=ncol, label=nlab, linewidth=2.5, alpha=0.5)
            ax[j][i].plot(steps, s[:, idx[ops[j*3+i]]], color=scol, label=slab, linewidth=2.5, alpha=0.5)
            ax[j][i].tick_params(labelsize=15)
            ax[j][i].set_ylabel(labs[ops[j*3]+i], size=20)

            if j==1:
                ax[j][i].set_xlabel("Steps", size=20)

    if savename != None:
        plt.savefig(savename, dpi=450)
        
    plt.show()
	

def get_bins(natrl, synth, Nbins, d3 = None, d4 = None):
    mn = np.min(natrl) if np.min(natrl) < np.min(synth) else np.min(synth)
    mx = np.max(natrl) if np.max(natrl) > np.max(synth) else np.max(synth)
    if d3 is not None:
        mn = mn if mn < np.min(d3) else np.min(d3)
        mx = mx if mx > np.max(d3) else np.max(d3)
    if d4 is not None:
        mn = mn if mn < np.min(d4) else np.min(d4)
        mx = mx if mx > np.max(d4) else np.max(d4)
    bn = np.linspace(mn, mx, Nbins+1)
    return bn


def histograms(n, s, ncol, scol, nlab, slab, Nbins, idx, labs, ops, savename=None, data3=None, data4=None):

    fig, ax = plt.subplots(1, 6, figsize=(12.8, 4.8), sharey='all', sharex='col')
    fig.subplots_adjust(left=0.075, right=0.995, top=0.975, bottom=0.15, wspace=0, hspace=0)
    
    for i, op in enumerate(ops):
        if data3!= None and data4!=None:
            dat3 = data3[0]
            dat4 = data4[0]
            bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins, dat3[:, idx[op]], dat4[:, idx[op]]) 
            di, bns = np.histogram(dat3[:, idx[op]], bns)       
            ax[0].stairs(di, edges=bns, color=data3[1], label=data3[2], linewidth=2.5, alpha=0.8)
            di, bns = np.histogram(dat4[:, idx[op]], bns)       
            ax[0].stairs(di, edges=bns, color=data3[1], label=data3[2], linewidth=2.5, alpha=0.8)
        elif data3!=None:
            dat = data3[0]
            bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins, dat[:, idx[op]]) 
            di, bns = np.histogram(dat[0][:, idx[op]], bns)       
            ax[0].stairs(di, edges=bns, color=data3[1], label=data3[2], linewidth=2.5, alpha=0.8)
        elif data4!=None:
            dat = data4[0]
            bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins, dat[:, idx[op]]) 
            di, bns = np.histogram(dat[0][:, idx[op]], bns)       
            ax[0].stairs(di, edges=bns, color=data4[1], label=data4[2], linewidth=2.5, alpha=0.8)
        else:
            bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins) 

        ni, bns = np.histogram(n[:, idx[op]], bns)
        si, bns = np.histogram(s[:, idx[op]], bns)
        
        ax[i].stairs(ni, edges=bns, color=ncol, label=nlab, linewidth=2.5, alpha=0.8)
        ax[i].stairs(si, edges=bns, color=scol, label=slab, linewidth=2.5, alpha=0.8)
        ax[i].tick_params(labelsize=15)
        ax[i].set_xlabel(labs[op], size=20)

    ax[0].set_ylabel("Count", size=20)
    ax[-1].legend(prop={'size':13})

    if savename != None:
        plt.savefig(savename, dpi = 450)

    plt.show()
    return None
    
def resample_seed(US_data, seed_data):
    # Want to resample seed_data to reflect the same Nq6 distribution as US_data
    c = Counter(US_data[:, 1])
    
    window_seed = []
    window_wts  = []

    for i in c:
        seed_N = seed_data[np.where(seed_data[:, 1]==i)[0]]
        window_seed.extend(seed_N)
        window_wts.extend([c[i]/len(seed_N)]*len(seed_N))
        
    return np.array(window_seed), window_wts

def reweight(data):
    weights = {}
    for i in range(125, 300):
        num = len(np.where(data[:, 1]==i)[0])
        try:
            weights[i] = 1.0/(num*175)
        except ZeroDivisionError:
            weights = 0
    nw = []
    for d in data:
        nw.append(weights[d[0]])
    return np.array(nw)
        


def equil_test(n, s, ncol1, ncol, scol1, scol,  nlab, slab, Nbins, idx, labs, ops, savename = None, offset=2500, scale=500, dt=0.005, reweight=False):

    bns = []
    for op in ops:
        bns.append(get_bins(n[:, idx[op]], s[:, idx[op]], Nbins))

    fig, ax = plt.subplots(4, 6, figsize=(12.8, 7.2), sharey='row', sharex='col')
    fig.subplots_adjust(left=0.075, right=0.74, top=0.995, bottom=0.12, wspace=0, hspace=0)

    plot_props = zip([n, s], [ncol1, scol1], [nlab, slab])
    for p in plot_props:
        dat = p[0] ; col = p[1] ; lb  = p[2]
        full_weights = reweight(dat)*len(dat) if reweight else [1]*len(dat)
        for k, op in enumerate(ops):
            for i in range(4):
                bndat, _ = np.histogram(dat[:, idx[op]], bns[k], weights=full_weights)
                ax[i][k].stairs(bndat, edges=bns[k], color=col, label=lb, linewidth=2.5, alpha=0.8)
                ax[i][k].tick_params(labelsize=15)
#        ax[3][0].set_xlabel(labs[op], size=20)
        
    plot_props = zip([n, s], [ncol, scol], [nlab, slab])
    for p in plot_props:
        dat = p[0] ; col = p[1] ; lb  = p[2]
        for i in range(4):
            d = dat[(offset*(i+1)+1):]
            full_weights = reweight(d)*len(d) if reweight else [1]*len(d)
            for k, op in enumerate(ops):
                # ax[i][k].hist(dat[:, k], bns[k], color=col, label=lb, alpha=0.4)
                bndat, _ = np.histogram(d[:, idx[op]], bns[k], weights=full_weights)
                ax[i][k].stairs(bndat, edges=bns[k], color=col, label=lb+', $t^*_U \geq '+'{:,}'.format(int(dt*(offset*(i+1))*scale))+'$', linewidth=2.5, alpha=0.8)
                ax[3][k].set_xlabel(labs[op], size=20)

    pans = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
            '(g)', '(h)', '(i)', '(j)', '(k)', '(l)',
            '(m)', '(n)', '(o)', '(p)', '(q)', '(r)',
            '(s)', '(t)', '(u)', '(v)', '(w)', '(x)']
   
    for i in range(4):
        for j in range(6):
            if j == 2 or j ==3:
                ax[i][j].text(0.775, 0.8, pans[i*6+j], size=20, transform=ax[i][j].transAxes)              
            else:
                ax[i][j].text(0.05, 0.8, pans[i*6+j], size=20, transform=ax[i][j].transAxes)
            if j == 0:
                ax[i][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':13})
                ax[i][0].set_ylabel("Count", size=20)

    ax[3][0].set_xticks([150, 250])

    if savename != None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None


def reweighted_plot(dat, dcol, dlab, ref, rcol, rlab, Nbins, idx, labs, ops, dat2 = None, ref2 = None, savename = None, KS=True):
    
    bns = []
    for op in ops:
        if ref2!=None and dat2!=None:
            dt2 = dat2[0]
            rf2 = ref2[0]
            bns.append(get_bins(dat[:, idx[op]], ref[:, idx[op]], Nbins, dt2[:, idx[op]], rf2[:, idx[op]]))
        elif ref2 != None:
            rf2 = ref2[0]
            bns.append(get_bins(dat[:, idx[op]], ref[:, idx[op]], Nbins, rf2[:, idx[op]]))
        elif dat2 != None:
            dt2 = dat2[0]
            bns.append(get_bins(dat[:, idx[op]], ref[:, idx[op]], Nbins, dt2[:, idx[op]]))
        else:
            bns.append(get_bins(dat[:, idx[op]], ref[:, idx[op]], Nbins))

    fig, ax = plt.subplots(1, 6, figsize=(12.8, 3.6), sharey='row', sharex='col')
    fig.subplots_adjust(left=0.0725, right=0.83, top=0.995, bottom=0.1825, wspace=0, hspace=0)

    for k, op in enumerate(ops):
        bndat, _ = np.histogram(dat[:, idx[op]], bns[k])
        ax[k].stairs(bndat, edges=bns[k], color=dcol, label=dlab, linewidth=2.5, alpha=0.8)
        ax[k].tick_params(labelsize=15)
        ax[k].set_xlabel(labs[op], size=20)
        
    if dat2 != None:
        d = dat2[0]
        for k, op in enumerate(ops):
            bndat, _ = np.histogram(d[:, idx[op]], bns[k])
            ax[k].stairs(bndat, edges=bns[k], color=dat2[1], label=dat2[2], linewidth=2.5, alpha=0.8)
            if KS:
                print("# INFO: Kolmogorov-Smirnov statistic for {} between two sets of data is {}".format(op, stats.ks_2samp(dat[:, idx[op]], d[:, idx[op]], alternative='two-sided').pvalue))


    ref_window, ref_weight = resample_seed(dat, ref)
    for k in range(1, 6):
        bndat, _ = np.histogram(ref_window[:, idx[ops[k]]], bns[k], weights=ref_weight)
        ax[k].stairs(bndat, edges=bns[k], color=rcol, label=rlab, linewidth=2.5, alpha=0.8)
        if KS:
            print("# INFO: Kolmogorov-Smirnov statistic for {} between data and FULL reference is {}".format(op, stats.ks_2samp(dat[:, idx[ops[k]]], ref_window[:, idx[ops[k]]]).pvalue))

    if ref2 != None:
        r2 = ref2[0]
        r2_window, r2_weight = resample_seed(dat, r2)
        for k in range(1, 6):
            bndat, _ = np.histogram(r2_window[:, idx[ops[k]]], bns[k], weights=r2_weight)
            ax[k].stairs(bndat, edges=bns[k], color=ref2[1], label=ref2[2], linewidth=2.5, alpha=0.8)
            if KS:
                print("# INFO: Kolmogorov-Smirnov statistic for {} between data and FULL reference 2 is {}".format(op, stats.ks_2samp(dat[:, idx[ops[k]]], ref_window[:, idx[ops[k]]]).pvalue))
    
    # ax[0].set_xticks([150, 250])
    
    ax[0].set_ylabel("Count", size=20)
    ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':13})
    #ax[-1].legend(loc='upper left', prop={'size':11})

    pans = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for k in range(6):
        if k ==3 :
            ax[k].text(0.8, 0.9, pans[k], size=18, transform=ax[k].transAxes)
        else:
            ax[k].text(0.025, 0.9, pans[k], size=18, transform=ax[k].transAxes)

    

    if savename != None:
        plt.savefig(savename, dpi=450)

    plt.show()
    return None

def single_hist(n, s, ncol, scol, nlab, slab, Nbins, idx, labs, op, dat3= None, dat4= None, savename=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), sharex='all')
    fig.subplots_adjust(left=0.14, right=0.995, top=0.975, bottom=0.14)

    if dat3 !=None and dat4!=None:
        d3 = dat3[0] ; d4 = dat4[0]
        bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins, d3[:, idx[op]], d4[:, idx[op]])
        plot_props = zip([n[:, idx[op]], s[:, idx[op]], d3[:, idx[op]], d4[:, idx[op]]],
                         [ncol, scol, dat3[1], dat4[1]], [nlab, slab, dat3[2], dat4[2]])
    elif dat3 != None:
        d3 = dat3[0] ; d4 = dat4[0]
        bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins, d3[:, idx[op]])
        plot_props = zip([n[:, idx[op]], s[:, idx[op]], d3[:, idx[op]]],
                         [ncol, scol, dat3[1]], [nlab, slab, dat3[2]])
    elif dat4 != None:
        d3 = dat3[0] ; d4 = dat4[0]
        bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins, d4[:, idx[op]])
        plot_props = zip([n[:, idx[op]], s[:, idx[op]], d4[:, idx[op]]],
                         [ncol, scol, dat4[1]], [nlab, slab, dat4[2]])
    else:
        bns = get_bins(n[:, idx[op]], s[:, idx[op]], Nbins)
        plot_props = zip([n[:, idx[op]], s[:, idx[op]]], [ncol, scol], [nlab, slab])
    

    for p in plot_props:
        print(len(p[0]))
        bndat, _ = np.histogram(p[0], bns)
        print(bndat)
        ax.stairs(bndat, edges=bns, color=p[1], label=p[2], linewidth=2.5, alpha=0.8)

        
    ax.set_xlabel(labs[op], size=20)
    ax.set_ylabel("Count", size=20)
    ax.legend(loc='upper left', prop={'size':13})
    ax.tick_params(labelsize=15)

    if savename != None:
        plt.savefig(savename, dpi=450)
    
    plt.show()
    return None
