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

all :  cluster_dump_conn

CLUSTER_OBJECTS = read_dcd.o harmonics.o cluster_funcs.o clusterinf.o Wolde_conn.o 

cluster_dump_conn : $(CLUSTER_OBJECTS)
	$(LD) -o cluster_dump_conn $(CLUSTER_OBJECTS)


clean : 

	rm -f *.mod *.d *.il *.o work.* cluster_dump_conn


