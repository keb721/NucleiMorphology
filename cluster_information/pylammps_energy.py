from lammps import PyLammps
import numpy as np
import argparse

# Set parameters
# --------------

parser = argparse.ArgumentParser()

parser.add_argument("-N", "--Nsteps",   help="Number of MD timesteps per MC sweep. Default = 100", type=int)
parser.add_argument("-s", "--segments", help="Total number of trajectory segments to obtain. Default = 1000000", type=int)
parser.add_argument("-i", "--repID",    help="ID for current run. Default = 1", type=int)
parser.add_argument("-u", "--upper",    help="Upper edge of the window in Nq6. Default = 250", type=int)
parser.add_argument("-l", "--lower",    help="Lower edge of the window in Nq6. Default = 125", type=int) 
parser.add_argument("-T", "--temp",     help="Temperature of the simulation. Default = 0.765", type=float)
parser.add_argument("-p", "--press",    help="Pressure of the simulation. Default = 5.68", type=float)
parser.add_argument("-d", "--datafile", help="Input file (LAMMPS data file) to start the simulation. Default = 'seed.data'", type=str)

args = parser.parse_args()

N = 100         if args.Nsteps   is None else args.Nsteps
s = 1000000     if args.segments is None else args.segments
i = 1           if args.repID    is None else args.repID
u = 300         if args.upper    is None else args.upper
l = 125         if args.lower    is None else args.lower
T = 0.765       if args.temp     is None else args.temp
p = 5.68        if args.press    is None else args.press
d = 'seed.data' if args.datafile is None else args.datafile

base = d.split(".data")
base = [i for i in base if len(i)!=0][0]
    
if base.find("step")<0:
    start         = -1
    restart_name = base+"_ID"+str(i)+"_N"+str(N)+"_T"+str(T)+"_p"+str(p)
else:
    start        = int(base.split("step")[1].split("_")[0])
    restart_name = base.split("_step")[0]

    
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
L.file("in.energy")                                     # Define the size of largest cluster RC
L.run(0)

for i in range(start+1, start+s+1):
    print(i)
    L.clear()
    L.variable("N equal ", N)
    L.variable("temp equal", T)
    L.variable("press equal", p)

    # Define LJ properties as not contained in data file
    L.units("lj") ; L.atom_style("atomic") ; L.atom_modify("map hash")
    L.pair_style("lj/cut 3.5") ; L.pair_modify("shift yes")

    f = restart_name+"_step"+str(i)+".data"
    print(f)
    
    L.read_data(f)
    L.variable("fele string", f)
    L.file("in.energy")                                     # Define the size of largest cluster RC
