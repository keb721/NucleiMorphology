PROGRAM MAIN

  USE CLUSTERINF

  IMPLICIT NONE

  CHARACTER(LEN=500)                             :: file, dump 
  INTEGER                                        :: N, fnum, solid_threshold, l, atom, tot_sol, sn_clus, &
                                                    i, j, k, Nq6, bg_clus, curr_clus, outfile, ierr
  LOGICAL                                        :: uc_present
  REAL                                           :: q6_threshold
  REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: positions, cls_pos
  REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: vec_ql_mat_rl 
  REAL(KIND=REAL64)                              :: uc_len, uc_inv, distance_cutoff, mnq, theta, phi, &
                                                    xlo, xhi, xp, yp, zp, Peters, Rogal, Moroni, area, vol
  REAL(KIND=REAL64), DIMENSION(:), ALLOCATABLE   :: rl, im, angles
  INTEGER, ALLOCATABLE                           :: cluster_mat(:), neighbours(:) 
                                                 !! Dealing with vectors to be assigned later
  INTEGER, DIMENSION(:), ALLOCATABLE             :: con_mat, cluster, cls_tmp

  l               = 6
  q6_threshold    = 0.5
  distance_cutoff = 1.432_REAL64 ! LAMMPS is working in LJ units not lattice units
  solid_threshold = 8          ! Need 8 neighbours to be solid


  CALL GET_COMMAND_ARGUMENT(1, file)

  fnum = 97
  OPEN(fnum, file=TRIM(file), status='OLD')
  READ(fnum,*) dump ; READ(fnum,*) N, dump ; READ(fnum,*) dump
  READ(fnum,*) xlo, xhi, dump ;  READ(fnum,*) dump ; READ(fnum,*) dump 
  READ(fnum,*) dump ; READ(fnum,*) dump ; READ(fnum,*) dump ; READ(fnum,*) dump ; READ(fnum,*) dump

  uc_len = xhi - xlo ; uc_inv = 1/uc_len 
 
  ALLOCATE(positions(3, N))
  
  DO i=1, N
     READ(fnum,*) atom, dump, xp, yp, zp, dump, dump, dump
     positions(1, atom) = xp ; positions(2, atom) = yp ; positions(3, atom) = zp
  END DO

  CLOSE(fnum)

  ALLOCATE(vec_ql_mat_rl(N, -l:l))
  ALLOCATE(con_mat(N))
  ALLOCATE(rl(-l:l))
       
  vec_ql_mat_rl  = 0.0_REAL64
  con_mat = 0
  
  ! Unsophisticated measure of getting neighbours relies on using each atom individually
  
  DO atom = 1, N       
     neighbours             = GET_NEIGHS(positions, uc_len, uc_inv, distance_cutoff, atom)       
     rl = VEC_Ql(l, positions, atom, uc_len, uc_inv, neighbours)
     vec_ql_mat_rl(atom, :) = rl
  END DO
  
  DO atom = 1, N
     neighbours             = GET_NEIGHS(positions, uc_len, uc_inv, distance_cutoff, atom)       
     con_mat(atom) = CONNECTIONS(atom, neighbours, vec_ql_mat_rl, q6_threshold)
  END DO
  
  cluster_mat = CLUSTERNUMS(con_mat, solid_threshold, q6_threshold, vec_ql_mat_rl, positions, uc_len, uc_inv, distance_cutoff)
        
  bg_clus = 0
  Nq6     = 0
  tot_sol = 0
  sn_clus = 0
  
  DO i = 1, MAXVAL(cluster_mat)
     curr_clus = COUNT(cluster_mat .EQ. i)
     tot_sol   = tot_sol + curr_clus
     IF (curr_clus .GT. Nq6) THEN
        bg_clus   = MERGE(i, bg_clus, curr_clus .GT. Nq6)
        sn_clus   = MERGE(Nq6, sn_clus, curr_clus .GT. Nq6)
        Nq6       = MERGE(curr_clus, Nq6, curr_clus .GT. Nq6)
     ELSE
        sn_clus   = MERGE(curr_clus, sn_clus, curr_clus .GT. sn_clus)
     END IF
  END DO

  WRITE(*,*) TRIM(file), tot_sol, sn_clus


  
END PROGRAM MAIN
