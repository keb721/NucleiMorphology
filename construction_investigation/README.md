Codes for investigating cluster construction

### Fortran codes

`test_neigh_cutoff.f90` is compiled by the `Makefile`, giving proximal and oriented cluster information for the specified neighbour cutoff ($\sigma_N$).
`step_neigh_cutoff.f90` steps through a range of $\sigma_N$ to get the size of the largest proximal and oriented clusters at this value of $\sigma_N$.


### Energy files

`proximal_energy_pylampps.py` uses `in.proximal_energy` to calculate the energy of proximal clusters. Internal cutoffs must be set in the files.

`oriented_energy_pylammps.py` uses `in.oriented_energy` to get the energy of an oriented cluster, where the atoms in the cluster are specified using `get_cluster_IDs.py`.
