#-*- mode: makefile; mode: font-lock; vc-back-end: RCS -*-
SHELL = /bin/sh

# Where you want the binary

# Compiler and settings
F90       = gfortran
LD        = gfortran
FFLAGS    = -O3 -g
INCLUDE   = 

.PRECIOUS: %.o
.PHONY:  clean

%: %.o
%.o: %.f90
	$(F90) $(FFLAGS) $(INCLUDE) -c -o $@ $<

%: %.o
	$(F90) $(FFLAGS) $(INCLUDE)  -o $@ $^

all :  test_neighbours

CLUSTER_OBJECTS = read_dcd.o harmonics.o cluster_funcs.o clusterinf.o test_neigh_cutoff.o

test_neighbours: $(CLUSTER_OBJECTS)
	$(LD) -o test_neighbours $(CLUSTER_OBJECTS)


clean : 

	rm -f *.mod *.d *.il *.o work.* test_neighbours


