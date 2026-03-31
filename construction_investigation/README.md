Codes for investigating cluster construction


### Energy files

`proximal_energy_pylampps.py` uses `in.proximal_energy` to calculate the energy of proximal clusters. Internal cutoffs must be set in the files.

`oriented_energy_pylammps.py` uses `in.oriented_energy` to get the energy of an oriented cluster, where the atoms in the cluster are specified using `get_cluster_IDs.py`.
