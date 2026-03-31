####################################################
#                                                  #
#    Generate cluster IDs for connected cluster    #
#                                                  #
####################################################

# Import packages
# ---------------

import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seeds',   help='Number of seeds', type=int)

args = parser.parse_args()


cluster_file = 'oriented_{}_cluster.txt'
seed_file    = '../{}_seed.data'
IDs_file     = '_cluster_IDs_{}.txt'

epsilon = 1e-6

for i in range(1, args.seeds+1):
    cluster = cluster_file.format(i)
    seed    = seed_file.format(i)

    # print(cluster, seed)
    
    uc_min, uc_max, _, _ = np.genfromtxt(seed, skip_header=7, max_rows=1) # Use cubic unit cell

    uc_length = uc_max - uc_min  ; inv_uc = 1.0/uc_length
    
    coords   = np.genfromtxt(cluster)
    all_info = np.genfromtxt(seed, skip_header=19, max_rows=6912)
    all_pos  = all_info[:, 2:5]

    IDs = []

    output = open(IDs_file.format(i), 'w')
    output.write(str(len(coords)))
    
    for atom in coords:
        # Test each atom in the cluster
        for i in range(3):
            # Wrap into unit cell
            atom[i] = atom[i] - uc_length*round(atom[i]*inv_uc)

        test =  all_pos - atom
        IDs.append(int(all_info[np.where(np.sum(abs(test), axis=1) < epsilon)[0][0], 0]))

        output.write('\n'+str(IDs[-1])+'\t 42')
    
    output.close()
    assert len(IDs) == len(coords)
