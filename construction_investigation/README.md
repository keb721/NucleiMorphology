Codes for investigating cluster construction

### Cluster analysis codes

`test_neigh_cutoff.f90` is compiled by the `Makefile`, giving proximal and oriented cluster information for the specified neighbour cutoff ($\sigma_N$). This also requires `quickhull.py`.


`step_neigh_cutoff.f90` steps through a range of $\sigma_N$ to get the size of the largest proximal and oriented clusters at this value of $\sigma_N$.


`get_subset.py` utilises the IDs of cluster constituents (via `get_cluster_IDs.py`) to determine if largest oriented cluster is a subset of largest proximal cluster.

`alpha.m` computes alpha_shape on files.

`separate_all.py` splits files containing both oriented and proximal data into just oriented or just proximal data.


### Energy files

`proximal_energy_pylampps.py` uses `in.proximal_energy` to calculate the energy of proximal clusters. Internal cutoffs must be set in the files.

`oriented_energy_pylammps.py` uses `in.oriented_energy` to get the energy of an oriented cluster, where the atoms in the cluster are specified using `get_cluster_IDs.py`.
