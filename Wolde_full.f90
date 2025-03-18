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
                                                    xlo, xhi, xp, yp, zp, Peters, Rogal, Moroni, area, vol, dtp
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
     neighbours    = GET_NEIGHS(positions, uc_len, uc_inv, distance_cutoff, atom)       
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
  
  ALLOCATE(cluster(Nq6))
  
  j = 1
  DO i = 1, atom
     IF (cluster_mat(i) .EQ. bg_clus) THEN
        cluster(j) = i
        j          = j + 1
     END IF
  END DO
  
  Peters = Ql_CLUSTER_PETERS(cluster, positions, uc_len, uc_inv, distance_cutoff, l)
  Rogal  = Ql_CLUSTER_ROGAL(cluster, vec_ql_mat_rl, positions, uc_len, uc_inv, distance_cutoff, l)
  Moroni = Ql_CLUSTER_Moroni(cluster, positions, uc_len, uc_inv, distance_cutoff, l)
  
  ALLOCATE(cls_tmp(Nq6))
  ALLOCATE(cls_pos(3, Nq6))

  cls_tmp = 0
  cls_tmp(1)    = cluster(1)
  cls_pos(:, 1) = positions(:, cls_tmp(1))
  j             = 2
  
  DO atom = 1, Nq6
     DO i = 1, Nq6
        IF (COUNT(cls_tmp .EQ. cluster(i)) .EQ. 1) CYCLE ! Don't check if the atom is already in the cluster
        tmp = positions(:, cluster(i)) - cls_pos(:, atom)
        DO k=1, 3
           tmp(k) = tmp(k) - uc_len*ANINT(tmp(k)*uc_inv)
        END DO
        dist = SQRT(DOT_PRODUCT(tmp, tmp))
        IF (dist .LE. distance_cutoff) THEN
           cls_pos(:, j) = cls_pos(:, atom) + tmp(:)
           cls_tmp(j)    = cluster(i)
           j             = j + 1
        END IF   
     END DO
  END DO

  OPEN(48, file="cluster_wrap.txt", status='UNKNOWN')
  
  DO i = 1, Nq6
     WRITE(48, *) cls_pos(1, i), cls_pos(2, i), cls_pos(3, i)
  END DO

  CLOSE(48)

  DEALLOCATE(cls_tmp)
  DEALLOCATE(cls_pos)
  
  
  CALL EXECUTE_COMMAND_LINE("python quickhull.py "//TRIM(file)//" > area.txt")

  OPEN(49, file="area.txt", status='OLD')
  READ(49, *) area, vol
  CLOSE(49)

  WRITE(*,*) TRIM(file), Nq6, Peters, Rogal, Nq6*Peters, Nq6*Rogal, Moroni, Nq6*Moroni, area, vol, tot_sol, sn_clus

  DEALLOCATE(cluster)
 
END PROGRAM MAIN
