from lammps import PyLammps
import numpy as np
import argparse

# Set parameters
# --------------

parser = argparse.ArgumentParser()

parser.add_argument("-N", "--Nsteps",   help="Number of MD timesteps per MC sweep. Default = 100", type=int)
parser.add_argument("-s", "--segments", help="Total number of trajectory segments to obtain. Default = 1000000", type=int)
parser.add_argument("-T", "--temp",     help="Temperature of the simulation. Default = 0.765", type=float)
parser.add_argument("-p", "--press",    help="Pressure of the simulation. Default = 5.68", type=float)
parser.add_argument("-d", "--datafile", help="Input file (LAMMPS data file) to start the simulation. Default = '1_seed.data'", type=str)

args = parser.parse_args()

N = 100         if args.Nsteps   is None else args.Nsteps
s = 1000000     if args.segments is None else args.segments
T = 0.765       if args.temp     is None else args.temp
p = 5.68        if args.press    is None else args.press
d = '1_seed.data' if args.datafile is None else args.datafile

base = d.split(".data")
base = [i for i in base if len(i)!=0][0]
    
start        = int(base.split("_seed")[0])

    
# Initialise first system
# -----------------------

L = PyLammps()

L.variable("N equal ", N)
L.variable("temp equal", T)
L.variable("press equal", p)
L.variable("fele string", d)

# Define LJ properties as not contained in data file
L.units("lj") ; L.atom_style("atomic") ; L.atom_modify("map hash")
L.pair_style("lj/cut 3.5") ; L.pair_modify("shift yes")

L.read_data(d)
L.pair_coeff("1 1 1.0 1.0")
L.variable("oriented atomfile oriented_cluster_IDs_"+str(start)+".txt")

L.file("in.oriented_energy")                                     # Define the size of largest cluster RC
L.run(0)

L.variable('oriented delete')

for i in range(start+1, start+s+1):
    print(i)
    L.clear()
    L.variable("N equal ", N)
    L.variable("temp equal", T)
    L.variable("press equal", p)

    # Define LJ properties as not contained in data file
    L.units("lj") ; L.atom_style("atomic") ; L.atom_modify("map hash")
    L.pair_style("lj/cut 3.5") ; L.pair_modify("shift yes")

    f = str(i)+"_seed.data"
    print(f)
    
    L.read_data(f)
    L.variable("fele string", f)
    L.variable("oriented atomfile oriented_cluster_IDs_"+str(i)+".txt")
    L.file("in.oriented_energy")                                     # Define the size of largest cluster RC

    L.variable('oriented delete')
    
