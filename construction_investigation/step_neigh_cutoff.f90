PROGRAM MAIN

  USE CLUSTERINF

  IMPLICIT NONE

  CHARACTER(LEN=500)                             :: file, dump
  INTEGER                                        :: N, fnum, solid_threshold, l, atom, tot_sol, sn_clus, &
                                                    i, j, k, Nq6, bg_clus, curr_clus, outfile, ierr, d
  LOGICAL                                        :: uc_present
  REAL                                           :: q6_threshold
  REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: positions, cls_pos
  REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: vec_ql_mat_rl 
  REAL(KIND=REAL64)                              :: uc_len, uc_inv, distance_cutoff, mnq, theta, phi, &
                                                    xlo, xhi, xp, yp, zp, Peters, Rogal, Moroni, area, vol, dtp
  REAL(KIND=REAL64), DIMENSION(:), ALLOCATABLE   :: rl, im, angles
  INTEGER, ALLOCATABLE                           :: conn_cluster_mat(:), cluster_mat(:), neighbours(:) 
                                                 !! Dealing with vectors to be assigned later
  INTEGER, DIMENSION(:), ALLOCATABLE             :: con_mat, cluster, cls_tmp

  
  ! g(r)s information
  INTEGER, PARAMETER                        :: hist_size = 45
  REAL(KIND=REAL64), PARAMETER              :: ftp = 16.0_REAL64*ATAN(1.0_REAL64)/3.0_REAL64, edge_size = 18.0_REAL64
  REAL(KIND=REAL64), DIMENSION(hist_size+1) :: hist_edges
  REAL(KIND=REAL64), DIMENSION(hist_size)   :: grs, vols
  REAL(KIND=REAL64), DIMENSION(3)           :: tmp
  REAL(KIND=REAL64)                         :: dist, volin, volout, hist_diff = 0.5_REAL64/hist_size

  l               = 6
  q6_threshold    = 0.5
!  distance_cutoff = 1.432_REAL64 ! LAMMPS is working in LJ units not lattice units
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

  DO d = 1, 351
     distance_cutoff = 1.198_REAL64 + d*0.002_REAL64
  
     ! Unsophisticated measure of getting neighbours relies on using each atom individually
  
     DO atom = 1, N       
        neighbours             = GET_NEIGHS(positions, uc_len, uc_inv, distance_cutoff, atom)
        rl = VEC_Ql(l, positions, atom, uc_len, uc_inv, neighbours)
        vec_ql_mat_rl(atom, :) = rl
     END DO
  
     DO atom = 1, N
        neighbours    = GET_NEIGHS(positions, uc_len, uc_inv, distance_cutoff, atom)       
        con_mat(atom) = CONNECTIONS(atom, neighbours, vec_ql_mat_rl, q6_threshold)
     END DO
  
     conn_cluster_mat = CONNECTED_CLUSTERNUMS(con_mat, solid_threshold, q6_threshold, vec_ql_mat_rl, &
          positions, uc_len, uc_inv, distance_cutoff)

  Nq6     = 0
  
  DO i = 1, MAXVAL(conn_cluster_mat)
     curr_clus = COUNT(conn_cluster_mat .EQ. i)
     IF (curr_clus .GT. Nq6) THEN
        Nq6       = MERGE(curr_clus, Nq6, curr_clus .GT. Nq6)
     END IF
  END DO


     
     
     ! Only want size
     WRITE(*,*) distance_cutoff, "connected", Nq6



   
     ! Unconnected cluster
     
     cluster_mat = CLUSTERNUMS(con_mat, solid_threshold, q6_threshold, vec_ql_mat_rl, positions, uc_len, uc_inv, distance_cutoff)

     Nq6 = 0
     
     DO i = 1, MAXVAL(cluster_mat)
        curr_clus = COUNT(cluster_mat .EQ. i)
        IF (curr_clus .GT. Nq6) THEN
           Nq6       = MERGE(curr_clus, Nq6, curr_clus .GT. Nq6)
        END IF
     END DO



     
     WRITE(*,*) distance_cutoff, "unconnected", Nq6
  
  END DO
 
END PROGRAM MAIN
