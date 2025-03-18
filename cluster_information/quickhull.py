import numpy as np
from scipy.spatial import ConvexHull
import argparse

par = argparse.ArgumentParser()
par.add_argument('filename')
args = par.parse_args()

f = args.filename.split("_")[0]
f = f + "_cluster.txt"

pos  = np.genfromtxt("cluster_wrap.txt") # File containing the position of atoms in the cluster
hull = ConvexHull(pos)
np.savetxt(f, pos)

print(hull.area, hull.volume)
