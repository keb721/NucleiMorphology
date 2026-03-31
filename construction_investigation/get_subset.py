import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seeds',   help='Number of seeds', type=int)
parser.add_argument('-f', '--outfile', help='Outputfile for seeds where largest oriented cluster not a subset of largest proximal cluster.')

args = parser.parse_args()

oriented_IDs_file   = 'oriented_cluster_IDs_{}.txt'
proximal_IDs_file = 'proximal_IDs_{}_cluster.txt'

not_subset = []

for i in range(1, args.seeds+1):
    oriented = np.genfromtxt(oriented_IDs_file.format(i), skip_header=1, dtype=int)[:, 0]
    proximal = np.genfromtxt(proximal_IDs_file.format(i))
    proximal = np.array(proximal, dtype=int)
    
    if not set(oriented).issubset(proximal):
        not_subset.append(i)
        print(i)
        print('oriented')
        print(sorted(oriented))
        print('proximal')
        print(proximal)

np.savetxt(args.outfile, np.array(not_subset), fmt='%i')
