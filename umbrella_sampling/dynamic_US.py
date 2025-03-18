from lammps import PyLammps
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# Set parameters
# --------------

parser = argparse.ArgumentParser()

parser.add_argument("-N", "--Nsteps",   help="Number of MD timesteps per MC sweep. Default = 100", type=int, default=100)
parser.add_argument("-s", "--segments", help="Total number of trajectory segments to obtain. Default = 20000", type=int, default=20000)
parser.add_argument("-i", "--repID",    help="ID for current run. Default = 1", type=int, default=1)
parser.add_argument("-u", "--upper",    help="Upper edge of the window in Nq6. Default = 300", type=int, default=300)
parser.add_argument("-l", "--lower",    help="Lower edge of the window in Nq6. Default = 125", type=int, default=125) 
parser.add_argument("-T", "--temp",     help="Temperature of the simulation. Default = 0.765", type=float, default=0.765)
parser.add_argument("-p", "--press",    help="Pressure of the simulation. Default = 5.68", type=float, default=5.68)
parser.add_argument("-d", "--datafile", help="Input file (LAMMPS data file) to start the simulation. Default = 'seed.data'", type=str, default='seed.data')
parser.add_argument("-r", "--ratio",    help="Proportion of old velocity to lose after each sweep. Default=0.5", type=float, default=0.5)

args = parser.parse_args()

N = args.Nsteps
s = args.segments
i = args.repID
u = args.upper
l = args.lower
T = args.temp
p = args.press
d = args.datafile
r = args.ratio * args.ratio

print(r, args.ratio)

vrand = np.random.randint(10e7)  
trand = np.random.randint(10e7)

base = d.split("data") ; base = [i.split(".") for i in base if len(i)>=2][0]
base = [i for i in base if len(i)!=0][0]

print(base)

restart_name = base+"_ID"+str(i)+"_N"+str(N)+"_T"+str(T)+"_p"+str(p)+"_r"+str(args.ratio)

# Initialise first system
# -----------------------

L = PyLammps()

L.variable("N equal ", N)
L.variable("temp equal", T)
L.variable("press equal", p)
L.variable("vrand equal", vrand)
L.variable("trand equal", trand)

# Define LJ properties as not contained in data file
L.units("lj") ; L.atom_style("atomic") ; L.atom_modify("map hash")
L.pair_style("lj/cut 3.5") ; L.pair_modify("shift yes")

L.read_data(d)
L.file("in.vmaxn")                                     # Define the size of largest cluster RC
L.thermo_style("custom step temp press pe v_max_n")    # Note: for 23Jun22 need to put RC in thermo_style
L.thermo(N)

L.velocity("all create", T, vrand, "mom yes dist gaussian")
L.write_restart(restart_name+".restart")               # Can leave window in first step


# Set up thermo sweep
# -------------------

accepted = 0
sweeps   = 0

traj     = []                      # Keep track of the Nq6 values in the constructed trajectory
ratio    = []                      # Keep track of dynamic acceptance ratio 


max_repeats = 0
max_curr    = 0

while (len(traj) < s):
    trand = np.random.randint(10e7)                            # Reroll thermostat random seed
    L.fix("1rr all temp/csvr", T, T, 0.05, trand)
    L.fix("2rr all nph iso", p, p, 0.5, "mtk yes pchain", 5)

    L.run(N)    
    
    sweeps += 1
    size    = L.runs[-1].thermo.v_max_n[-1]                    # Can only access size through thermostyle
    
    if (size < u) and (size >= l) :
        # Accept this run and add it to the trajectory
        accepted += 1
        L.unfix("1rr")             # Need to unfix for restarts
        L.unfix("2rr")
        L.write_restart(restart_name+".restart")
        # Write information about the system for post-processing
        L.write_data(restart_name+"_step"+str(len(traj)+1)+".data") 

        max_curr = 0
        
    else:
        # Reject this run and restart from end of last accepted segment
        L.clear()
        L.read_restart(restart_name+".restart")
        L.write_data(restart_name+"_step"+str(len(traj)+1)+".data")
        try:
            size = traj[-1]
        except IndexError:
            size = L.runs[-1].thermo.v_max_n[0]
        L.file("in.vmaxn")

        L.thermo_style("custom step temp press pe v_max_n")
        L.thermo(N)

        max_curr += 1
        max_repeats = max_repeats if max_repeats > max_curr else max_curr
        
    traj.append(size)          # Add to trajectory

    ratio.append(1.0*accepted/sweeps)

    vrand = np.random.randint(10e7)
    L.velocity("all scale", (1-r)*T)
    L.velocity("all create", r*T, vrand, "mom yes dist gaussian sum yes")

np.savetxt(restart_name+"_acceptance.txt", ratio)

print('The maximum number of timesteps before generating a new configuration which fell within the window was {}.'.format(max_repeats))
