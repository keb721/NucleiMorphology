MODULE CLUSTERINF

  USE cluster_funcs

  IMPLICIT NONE

  !##################################################!
  !                                                  !
  !    ANALYSE CLUSTERS FOR CLUSTERINF DUMP FILES    !
  !                                                  !
  !##################################################!

  ! Clusterinf files have two per-atom numbers
  ! 1. The number of connections
  ! 2. The cluster number of an atom

  ! THE QUANTITIES IN THIS FILE ARE INDEPENDENT OF THE QL USED, I.E. ONLY RELY ON 
  ! OUTPUTS FROM CLUSTER_FUNCS. 
  
  ! Cluster_funcs included until a better 
  ! neighbour-handling-thing is implemented


  CONTAINS
    
    FUNCTION LOCAL_Ql(atom, neighbours, vecql_rl, vecql_im) RESULT(tot)
      ! ========================================================== !
      ! Computes the average over all bonds, as in PLUMED LOCAL_Q6 !
      ! ========================================================== !
      INTEGER, INTENT(IN)                           :: atom
      INTEGER, DIMENSION(:), INTENT(IN)             :: neighbours
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: vecql_rl, vecql_im
      INTEGER, DIMENSION(1)                         :: nshape
      REAL(KIND=REAL64)                             :: tot, mean_rl, mean_im, dotpro
      INTEGER                                       :: i, neigh

      ! Loop over all bonds, and find the average value of the bonds

      nshape  = SHAPE(neighbours)
      tot     = 0.0_REAL64

      mean_rl = 0.0_REAL64
      mean_im = 0.0_REAL64

     DO i=1, nshape(1)
         neigh   = neighbours(i)
         dotpro  = DOT_PRODUCT(vecql_rl(atom, :), vecql_rl(neigh, :))
         mean_rl = mean_rl + dotpro
         dotpro  = DOT_PRODUCT(vecql_im(atom, :), vecql_im(neigh, :))
         mean_rl = mean_rl + dotpro
         dotpro  = DOT_PRODUCT(vecql_rl(atom, :), vecql_im(neigh, :))
         mean_im = mean_im + dotpro
         dotpro  = DOT_PRODUCT(vecql_im(atom, :), vecql_rl(neigh, :))
         mean_im = mean_im + dotpro
!         print*, mean_im
!        mean_rl = mean_rl - vecql_rl(atom, 7)*vecql_rl(neigh, 7)

      END DO

      tot = mean_rl

 !     tot = SQRT(mean_rl**2 + mean_im**2)
      tot = tot/nshape(1)

    END FUNCTION LOCAL_Ql

    FUNCTION ALL_Ql_CLUSTER_ROGAL(cluster, vecql_rl, vecql_im, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(ql_cl)
      ! ========================================================================================== !
      ! Compute the average Q6 crystallinity of a cluster as defined in Liang et al. 2020, JCP 152 !
      ! ========================================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)               :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                             :: l
      REAL(KIND=REAL64), DIMENSION(:,-l:), INTENT(IN) :: vecql_rl, vecql_im
      REAL(KIND=REAL64), INTENT(IN)                   :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)   :: pos
      REAL(KIND=REAL64)                               :: ql_cl, num_rl, num_im, denom
      REAL(KIND=REAL64), PARAMETER                    :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      INTEGER, ALLOCATABLE                            :: neighbours(:)
      INTEGER                                         :: atom, neighs, i, m, N
      INTEGER, DIMENSION(1)                           :: Nsize
     
      Nsize = SHAPE(cluster)
      N     = Nsize(1)
      
      ql_cl = 0.0_REAL64

      ! Start outer loop
      DO m = -1*l, l
         
         ! Initialise real and imaginary parts of numerator, and the denominator
         num_rl = 0.0_REAL64 ; num_im = 0.0_REAL64 ; denom = 0.0_REAL64

         ! For each atom in the cluster, add qlm(i)*neighs(i) to numerator and neighs(i) to denominator
         DO i = 1, N

            atom       = cluster(i)
            neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
            nsize      = SHAPE(neighbours)
            neighs     = nsize(1)

            num_rl     = num_rl + vecql_rl(atom, m)*neighs
            num_im     = num_im + vecql_im(atom, m)*neighs
            denom      = denom  + neighs*1.0_REAL64

         END DO
         
         num_rl = num_rl / denom
         num_im = num_im / denom
         
         ! Now want to compute the square of the magnitude of this sum, and add to ql_cl

         ql_cl = ql_cl + (num_rl**2 + num_im**2)
         
      END DO

      ql_cl = SQRT(ql_cl * frpi / (2.0_REAL64*l+1.0_REAL64))

    END FUNCTION ALL_Ql_CLUSTER_ROGAL

    FUNCTION Ql_CLUSTER_ROGAL(cluster, vecql_rl, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(ql_cl)
      ! ========================================================================================== !
      ! Compute the average Q6 crystallinity of a cluster as defined in Liang et al. 2020, JCP 152 !
      ! ========================================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)               :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                             :: l
      REAL(KIND=REAL64), DIMENSION(:,-l:), INTENT(IN) :: vecql_rl
      REAL(KIND=REAL64), INTENT(IN)                   :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)   :: pos
      REAL(KIND=REAL64)                               :: ql_cl, num_rl, num_im, denom
      REAL(KIND=REAL64), PARAMETER                    :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      INTEGER, ALLOCATABLE                            :: neighbours(:)
      INTEGER                                         :: atom, neighs, i, m, N
      INTEGER, DIMENSION(1)                           :: Nsize

      ! As ALL_Ql_CLUSTER_ROGAL above, but using real spherical harmonics
      
      Nsize = SHAPE(cluster)
      N     = Nsize(1)
      
      ql_cl = 0.0_REAL64
      
      ! Start outer loop
      DO m = -1*l, l
         
         ! Initialise numerator and  denominator
         num_rl = 0.0_REAL64 ; denom = 0.0_REAL64

         ! For each atom in the cluster, add qlm(i)*neighs(i) to numerator and neighs(i) to denominator
         DO i = 1, N

            atom       = cluster(i)
            neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
            nsize      = SHAPE(neighbours)
            neighs     = nsize(1)
            num_rl     = num_rl + vecql_rl(atom, m)*neighs
            denom      = denom  + neighs*1.0_REAL64
            
         END DO
         
         num_rl = num_rl / denom
         
         ! Now want to compute the square of the magnitude of this sum, and add to ql_cl

         ql_cl = ql_cl + (num_rl**2)
         
      END DO

      ql_cl = SQRT(ql_cl * frpi / (2.0_REAL64*l+1.0_REAL64))

    END FUNCTION Ql_CLUSTER_ROGAL

    FUNCTION ALL_Ql_CLUSTER_PETERS(cluster, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(ql_cl)
      ! =============================================================================================== !
      ! Compute the average Q6 crystallinity of a cluster as defined in Beckham and Peters 2011, JPCL 2 !
      ! =============================================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)             :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                           :: l
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64)                             :: ql_cl
      REAL(KIND=REAL64), PARAMETER                  :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(:), ALLOCATABLE  :: qlm_bar_rl, qlm_bar_im
      INTEGER                                       :: m
       
      ql_cl = 0.0_REAL64

      CALL ALL_MEAN_qlm_CLUSTER(cluster, pos, uc_len, uc_inv, distance_cutoff, l, qlm_bar_rl, qlm_bar_im)
          
      DO m = -1*l, l
         ql_cl = ql_cl + (qlm_bar_rl(m)**2 + qlm_bar_im(m)**2)
      END DO

      ql_cl = SQRT(ql_cl * frpi / (2.0_REAL64*l+1.0_REAL64))

    END FUNCTION ALL_Ql_CLUSTER_PETERS

    FUNCTION Ql_CLUSTER_PETERS(cluster, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(ql_cl)
      ! =============================================================================================== !
      ! Compute the average Q6 crystallinity of a cluster as defined in Beckham and Peters 2011, JPCL 2 !
      ! =============================================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)             :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                           :: l
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64)                             :: ql_cl
      REAL(KIND=REAL64), PARAMETER                  :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(-l:l)            :: qlm_bar
      INTEGER                                       :: m

      ! As ALL_Ql_CLUSTER_PETERS, using real spherical harmonics instead of full spherical harmonics
            
      ql_cl = 0.0_REAL64

      qlm_bar = MEAN_qlm_CLUSTER(cluster, pos, uc_len, uc_inv, distance_cutoff, l)
          
      DO m = -1*l, l
         ql_cl = ql_cl + qlm_bar(m)**2
      END DO

      ql_cl = SQRT(ql_cl * frpi / (2.0_REAL64*l+1.0_REAL64))

    END FUNCTION Ql_CLUSTER_PETERS

    FUNCTION Ql_CLUSTER_MORONI(cluster, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(ql_cl)
      ! =============================================================================================== !
      ! Compute the average Q6 crystallinity of a cluster as defined in Moroni et al 2005 PRL 94 235703 !
      ! =============================================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)             :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                           :: l
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64)                             :: ql_cl
      REAL(KIND=REAL64), PARAMETER                  :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(-l:l)            :: qlm_bar
      INTEGER                                       :: m

      ql_cl = 0.0_REAL64

      qlm_bar = Qlm_CLUSTER_MORONI(cluster, pos, uc_len, uc_inv, distance_cutoff, l)
          
      DO m = -1*l, l
         ql_cl = ql_cl + qlm_bar(m)**2
      END DO

      ql_cl = SQRT(ql_cl * frpi / (2.0_REAL64*l+1.0_REAL64))

    END FUNCTION Ql_CLUSTER_MORONI
    
    FUNCTION Q_l_BAR(l, neighbours, atom_number, avg_qlms)
      !===================================================================!
      ! Compute the q_l(bar) as eq.5 in Lechner and Dellago 2008, JCP 129 !
      ! ================================================================= !
      REAL(KIND=REAL64), DIMENSION(:, -l:), INTENT(IN) :: avg_qlms ! An array with the mean qlm value for each atom
      INTEGER, DIMENSION(:), INTENT(IN)                :: neighbours
      INTEGER, INTENT(IN)                              :: atom_number, l
      REAL(KIND=REAL64)                                :: sum, q_l_bar, theta, phi, q_lm_bar, inv_neigh
      REAL(KIND=REAL64), PARAMETER                     :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      INTEGER                                          :: m, neighs, i, j
      INTEGER, DIMENSION(1)                            :: nshape

      nshape = SHAPE(neighbours)
      neighs = nshape(1) + 1 ! Now including particle i

      inv_neigh = 1.0_REAL64/(1.0_REAL64*neighs)

      sum = 0.0_REAL64
      
      DO m=-1*l, l
         q_lm_bar = avg_qlms(atom_number, m)
         DO j=1, neighs - 1
            q_lm_bar = q_lm_bar + avg_qlms(neighbours(j), m) ! For every nieghbour increment by its qlm value
         END DO
         q_lm_bar = q_lm_bar * inv_neigh
         sum = sum + q_lm_bar**2
      END DO

      sum = sum*frpi/(2_REAL64*l+1_REAL64)
      
      Q_l_bar = SQRT(sum)
 
    END FUNCTION Q_l_BAR


    FUNCTION CONNECTIONS(atom, neighbours, vecql_rl, q6_threshold) RESULT(cons)
      ! ======================================= !
      ! Compute number of atom-atom connections !
      ! ======================================= !
      REAL, INTENT(IN)                              :: q6_threshold ! Value over which atoms are connected
      INTEGER, DIMENSION(:), INTENT(IN)             :: neighbours
      INTEGER, INTENT(IN)                           :: atom
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: vecql_rl
      INTEGER                                       :: neigh, j, cons
      INTEGER, DIMENSION(1)                         :: nshape
      REAL(KIND=REAL64)                             :: dotpro

      ! From ten Wolde, connections are when q(i) . q(j) exceeds a threhold value
      ! Assuming this means the real part of that only.
      ! Here we have input arrays, vecql, of shape(2l+1) which store the 
      ! q_l vector for thid atoms, as well as an output, cons, storing
      ! the number of connections for this atom
 
      ! Get number of neighbours, neighs, and start a counter for connections

      nshape = SHAPE(neighbours)
      cons   = 0

      DO j=1, nshape(1)
         neigh = neighbours(j)
         dotpro = DOT_PRODUCT(vecql_rl(atom, :), vecql_rl(neigh, :))
         IF (dotpro .GT. q6_threshold) cons = cons+1
      END DO

    END FUNCTION CONNECTIONS
 
   
    FUNCTION ALL_CONNECTIONS(atom, neighbours, vecql_rl, vecql_im, q6_threshold) RESULT(cons)
      ! ======================================= !
      ! Compute number of atom-atom connections !
      ! ======================================= !
      REAL, INTENT(IN)                              :: q6_threshold ! Value over which atoms are connected
      INTEGER, DIMENSION(:), INTENT(IN)             :: neighbours
      INTEGER, INTENT(IN)                           :: atom
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: vecql_rl, vecql_im
      INTEGER                                       :: neigh, j, cons
      INTEGER, DIMENSION(1)                         :: nshape
      REAL(KIND=REAL64)                             :: dotpro

      ! From ten Wolde, connections are when q(i) . q(j) exceeds a threhold value
      ! Assuming this means the real part of that only.
      ! Here we have input arrays, vecql, of shape(2l+1) which store the 
      ! q_l vector for thid atoms, as well as an output, cons, storing
      ! the number of connections for this atom
 
      ! Get number of neighbours, neighs, and start a counter for connections
      
      nshape = SHAPE(neighbours)
      cons   = 0

      DO j=1, nshape(1)
         neigh = neighbours(j)
         dotpro = DOT_PRODUCT(vecql_rl(atom, :), vecql_rl(neigh, :))
         ! Real part is symmetric, imaginary part is anti-symmetric, therefore don't
         ! need to worry about cross terms
         dotpro = dotpro + DOT_PRODUCT(vecql_im(atom, :), vecql_im(neigh, :))
         IF (dotpro .GT. q6_threshold) cons = cons+1
      END DO

    END FUNCTION ALL_CONNECTIONS


    FUNCTION CLUSTERNUMS(con_mat, solid_threshold, q6_threshold, vecql_rl, pos, uc_len, uc_inv, distance_cutoff)
      ! ========================================== !
      ! Determine which cluster an atom belongs to !
      ! ========================================== !
      INTEGER, DIMENSION(:), INTENT(IN)             :: con_mat
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: vecql_rl!, vecql_im
      REAL, INTENT(IN)                              :: q6_threshold ! q(i).q(j) value to be connected
      INTEGER, INTENT(IN)                           :: solid_threshold ! number of sconnections to be solid
      INTEGER, DIMENSION(:), ALLOCATABLE            :: clusternums
      INTEGER                                       :: atom, clusters, neigh, i, j, N, nnlen, currlen
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, distance_cutoff
      INTEGER, ALLOCATABLE                          :: neighbours(:)
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64)                             :: dotpro !, sm, square
      INTEGER, DIMENSION(2)                         :: posshape
      INTEGER, DIMENSION(1)                         :: nshape, nnshape
      INTEGER, DIMENSION(:), ALLOCATABLE            :: neighneighs, tmp_neigh

      posshape = SHAPE(pos)
      N        = posshape(2)

      ALLOCATE(clusternums(N))

      ! Set every atom to unassigned
      clusternums = -1 

      ! Initialise counter of number of clusters
      clusters = 0

      ! Can't just iterate through atoms and assign clusters if over the threshold,
      ! as this leads to multiple IDs for the cluster when picking two unconnected points on
      ! the same cluster one after the other. 
            
      DO atom=1, N

!         PRINT*, "ATOM", atom
!         PRINT*, con_mat(atom)
!         PRINT*, clusternums(atom)

         IF (clusternums(atom) .NE. -1) CYCLE
         ! Already been assigned

         IF (con_mat(atom) .LT. solid_threshold) THEN
            ! Skip if number of connections under threshold, i.e. liquid
            clusternums(atom) = 0
            CYCLE
         END IF

         clusters          = clusters + 1  
         clusternums(atom) = clusters
         
!         PRINT*, clusternums(atom)

         ! Now comes the hard part. We have an atom which is part of a cluster of unknown extent.
         ! We need to iterate over the neighbours of this atom, their neighbours, their neighbours
         ! neighbours etc. 

         ! Use some kind of a while loop?
         
         neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
         nshape     = SHAPE(neighbours)
         nnlen      = nshape(1)

         ALLOCATE(neighneighs(nnlen))
         
         neighneighs = neighbours

         DO WHILE (nnlen .GT. 0)
            ! Need this loop to complete as many times as needed to get to nnlen = 0

            ! 1. Get a list of all of the nieghbours of the atom to consider (outside loop
            !    for original atom, as #4. for neighs of neighs etc)
            ! 2. Check if neighbour in cluster 
            ! 3. If not in cluster, remove from list to consider
            ! 4. Generate new list of neighbours of atoms in list (1. again)
            
            DO i = 1, nnlen

               IF (clusternums(neighneighs(i)) .NE. -1) THEN
                  ! Already assigned - skip and remove from list
                  neighneighs(i) = 0
                  CYCLE
               END IF                  

               IF (con_mat(neighneighs(i)) .LT. solid_threshold) THEN
                  ! Assign and remove from list if liquid
                  clusternums(neighneighs(i)) = 0
                  neighneighs(i)              = 0
                  CYCLE
               END IF
               
               clusternums(neighneighs(i)) = clusters
            
            END DO

            ! Now we have assigned every member of neighneighs to either liquid
            ! or part of the cluster, and the nieghneighs list is purged of
            ! irrelevant entries

            ! Now need to generate a new array of the neighbours of these

            nnlen = nnlen - COUNT(neighneighs .EQ. 0)
            
            IF (nnlen .EQ. 0) CYCLE            
            
            ALLOCATE(tmp_neigh(nnlen))
            nnshape = SHAPE(tmp_neigh)
            
            nshape = SHAPE(neighneighs)
            
            j = 1
            DO i=1, nshape(1)
               IF (neighneighs(i) .EQ. 0) CYCLE
               tmp_neigh(j) = neighneighs(i)
               j            = j +1
            END DO

            ! This is a temporary array holding all of the neighbours whos neighbours we need
            ! to consider
            
            DEALLOCATE(neighneighs)            
            nnlen = 0
                        
            ! Now find how many neighbours of neighbours there are
            DO j = 1, nnshape(1)
               neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, tmp_neigh(j))
               nshape     = SHAPE(neighbours)
               nnlen      = nnlen + nshape(1)
            END DO
            
            ALLOCATE(neighneighs(nnlen))

            currlen = 1
            
            DO j = 1, nnshape(1)
               neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, tmp_neigh(j))
               nshape     = SHAPE(neighbours)

               neighneighs(currlen:currlen+nshape(1)) = neighbours
               currlen                                = currlen + nshape(1)
            END DO
            DEALLOCATE(tmp_neigh)
         END DO
         DEALLOCATE(neighneighs)
      END DO

    END FUNCTION CLUSTERNUMS

    FUNCTION CONNECTED_CLUSTERNUMS(con_mat, solid_threshold, q6_threshold, vecql_rl, pos, uc_len, uc_inv, distance_cutoff) & 
             RESULT(clusternums)
      ! ========================================== !
      ! Determine which cluster an atom belongs to !
      ! ========================================== !
      INTEGER, DIMENSION(:), INTENT(IN)             :: con_mat
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: vecql_rl!, vecql_im
      REAL, INTENT(IN)                              :: q6_threshold ! q(i).q(j) value to be connected
      INTEGER, INTENT(IN)                           :: solid_threshold ! number of sconnections to be solid
      INTEGER, DIMENSION(:), ALLOCATABLE            :: clusternums
      INTEGER                                       :: atom, clusters, neigh, i, j, N, nlen, currlen, clnm
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, distance_cutoff
      INTEGER, ALLOCATABLE                          :: neighbours(:)
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64)                             :: dotpro !, sm, square
      INTEGER, DIMENSION(2)                         :: posshape
      INTEGER, DIMENSION(1)                         :: nshape
      INTEGER, DIMENSION(:), ALLOCATABLE            :: clst

      posshape = SHAPE(pos)
      N        = posshape(2)

      ALLOCATE(clusternums(N))
      ALLOCATE(clst(N))

      ! Set every atom to unassigned
      clusternums = -1 

      ! Initialise counter of number of clusters
      clusters = 0

      ! Can't just iterate through atoms and assign clusters if over the threshold,
      ! as this leads to multiple IDs for the cluster when picking two unconnected points on
      ! the same cluster one after the other. 
            
      DO atom=1, N

!         PRINT*, "ATOM", atom
!         PRINT*, con_mat(atom)
!         PRINT*, clusternums(atom)

         IF (clusternums(atom) .NE. -1) CYCLE
         ! Already been assigned

         IF (con_mat(atom) .LT. solid_threshold) THEN
            ! Skip if number of connections under threshold, i.e. liquid
            clusternums(atom) = 0
            CYCLE
         END IF

         clusters          = clusters + 1  
         clusternums(atom) = clusters
         
!         PRINT*, "solid, cluster", clusternums(atom)
         
         clst    = 0
         clst(1) = atom
         clnm    = 2
         currlen = 1

         ! Now comes the hard part. We have an atom which is part of a cluster of unknown extent.
         ! We need to iterate over the neighbours of this atom, their neighbours, their neighbours
         ! neighbours etc. 

         ! Use some kind of a while loop?

         neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
         nshape     = SHAPE(neighbours)
         nlen       = nshape(1)

         DO WHILE (nlen .GT. 0)
            ! Need this loop to complete as many times as needed to get to nnlen = 0

            ! 1. Get a list of all of the nieghbours of the atom to consider (outside loop
            !    for original atom, as #4. for neighs of neighs etc)
            ! 2. Check if neighbour in cluster 
            ! 3. If not in cluster, remove from list to consider
            ! 4. Generate new list of neighbours of atoms in list (1. again)
            
            DO i = 1, nlen

               IF (clusternums(neighbours(i)) .NE. -1) THEN
                  ! Already assigned - skip and remove from list
                  neighbours(i) = 0
                  CYCLE
               END IF                  

               IF (con_mat(neighbours(i)) .LT. solid_threshold) THEN
                  ! Assign and remove from list if liquid
                  clusternums(neighbours(i)) = 0
                  neighbours(i)              = 0
                  CYCLE
               END IF
 
               ! Now need to check if this is connected
               dotpro = DOT_PRODUCT(vecql_rl(clst(currlen), :), vecql_rl(neighbours(i), :))
!               WRITE(*,*) clst(currlen), neighbours(i), dotpro
               IF (dotpro .GT. q6_threshold) THEN
                  ! Connected, therefore add to cluster
                  clusternums(neighbours(i)) = clusters
                  clst(clnm)                 = neighbours(i)
                  clnm                       = clnm + 1 
               END IF
            END DO

            ! Now we have assigned every member of neighbours of atom currlen to cluster or not
            ! Move on to the next atom in the cluster (or end loop)

!            write(*,*) "NEXT: ", currlen+1, clst(currlen), clst(currlen+1)
            currlen = currlen + 1 
            
            IF (clst(currlen) .EQ. 0) THEN
               nlen = 0
               CYCLE
            END IF

            neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, clst(currlen))
            nshape     = SHAPE(neighbours)
            nlen       = nshape(1)

         END DO
      END DO
      DEALLOCATE(clst)
    END FUNCTION CONNECTED_CLUSTERNUMS

    FUNCTION CLUSTER_SHAPE(clusternums, lg_cluster, pos, uc_len, uc_inv, distance_cutoff)
      ! ====================================================== !
      ! Determine the moment of inertia of the largest cluster !
      ! ====================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)              :: clusternums
      INTEGER, INTENT(IN)                            :: lg_cluster
      REAL(KIND=REAL64), INTENT(IN)                  :: uc_len, uc_inv, distance_cutoff
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)  :: pos
      REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: cpos
      INTEGER                                        :: atom, size, ii, jj, kk, O
      INTEGER, DIMENSION(1)                          :: mnloc, nshape
      INTEGER, DIMENSION(:), ALLOCATABLE             :: catoms
      INTEGER, ALLOCATABLE                           :: neighbours(:)
      REAL(KIND=REAL64), DIMENSION(3)                :: CoM, eig, es
      REAL(KIND=REAL64), DIMENSION(3, 3)             :: inertia_tensor
      REAL(KIND=REAL64)                              :: a, b, c, Q, R, phi, rootq, cluster_shape, svar
      LOGICAL                                        :: connatom

      ! Constants
   
      REAL(KIND=REAL64), PARAMETER :: onhlf   = 1.0_REAL64/2.0_REAL64
      REAL(KIND=REAL64), PARAMETER :: onthd   = 1.0_REAL64/3.0_REAL64
      REAL(KIND=REAL64), PARAMETER :: onsth   = 1.0_REAL64/6.0_REAL64
      REAL(KIND=REAL64), PARAMETER :: onnth   = 1.0_REAL64/9.0_REAL64
      REAL(KIND=REAL64), PARAMETER :: ontws   = 1.0_REAL64/27.0_REAL64
      REAL(KIND=REAL64), PARAMETER :: twthdpi = 2.0_REAL64*(4.0_REAL64*ATAN(1.0_REAL64))*onthd

      ! First calculate the CoM of the cluster
      
      size     = COUNT(clusternums .EQ. lg_cluster)
      
      ALLOCATE(cpos(3, size)) 
      ALLOCATE(catoms(size))

      ii        = 1
      catoms    = 0 
      cpos      = 0.0_REAL64
      CoM       = 0.0_REAL64      

      mnloc       = MINLOC(clusternums, (clusternums .EQ. lg_cluster)) 
      O           = mnloc(1)
      catoms(ii)  = O

      cpos(:, ii)     = pos(:, O) ! This atom is the "origin" of the cluster
      CoM             = cpos(:, ii)
      ii              = ii + 1
      kk              = 1

      DO WHILE (COUNT(catoms .EQ. 0) .NE. 0)
         ! Don't make any assumptions about how the cluster was assigned
         neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, O)   
         nshape     = SHAPE(neighbours)
         DO jj = 1, nshape(1)
            ! Assign all neighbouring atoms that are in the cluster with correct 
            IF (clusternums(neighbours(jj)) .NE. lg_cluster) CYCLE ! Atom not in the largest cluster
            IF (COUNT(catoms .EQ. neighbours(jj)) .NE. 0) CYCLE    ! Atom already accounted for

            cpos(:, ii) = CONNECTED_ATOM(cpos(:, kk), pos(:, neighbours(jj)), uc_len, uc_inv)
            CoM         = CoM + cpos(:, ii)
            catoms(ii)  = neighbours(jj)
            ii          = ii + 1
         END DO
         
         ! Move on to the next atom - the position for this has already been assigned!
         kk = kk + 1
         O  = catoms(kk)
      END DO
      
      ii = ii - 1 

      CoM = CoM/(1.0_REAL64*ii)  ! Location of CoM  

      inertia_tensor = 0.0_REAL64
      
      DO jj=1, 3
         CoM(jj) = SUM(pos(jj, :))/size
      END DO

      DO ii = 1, size
         cpos(:, ii) = cpos(:, ii) - CoM  ! Move cluster so that the CoM is at the centre of the box

         ! Assume all masses are unity for ease of calculation
         inertia_tensor(1, 1) = inertia_tensor(1, 1) + cpos(2, ii)**2 + cpos(3, ii)**2
         inertia_tensor(2, 2) = inertia_tensor(2, 2) + cpos(1, ii)**2 + cpos(3, ii)**2
         inertia_tensor(3, 3) = inertia_tensor(3, 3) + cpos(1, ii)**2 + cpos(2, ii)**2
         
         inertia_tensor(1, 2) = inertia_tensor(1, 2) - cpos(1, ii)*cpos(2, ii)
         inertia_tensor(1, 3) = inertia_tensor(1, 3) - cpos(1, ii)*cpos(3, ii)
         inertia_tensor(2, 3) = inertia_tensor(2, 3) - cpos(2, ii)*cpos(3, ii)
      END DO
               
      DEALLOCATE(cpos)

      ! Eigenvalues of MoI tensor are solutions to equation of form x^3+ax^2+bx+c = 0

      a = inertia_tensor(1, 1) + inertia_tensor(2, 2) + inertia_tensor(3, 3) 
      a = a * (-1)

      b = inertia_tensor(1, 1)*inertia_tensor(2, 2) + inertia_tensor(3, 3)*inertia_tensor(2, 2) +         &
          inertia_tensor(1, 1)*inertia_tensor(3, 3) - inertia_tensor(1, 2)**2 - inertia_tensor(1, 3)**2 - &
          inertia_tensor(2, 3)**2
           
      c = (inertia_tensor(1, 2)**2)*inertia_tensor(3, 3) + (inertia_tensor(1, 3)**2)*inertia_tensor(2, 2) +                  &
          (inertia_tensor(2, 3)**2)*inertia_tensor(1, 1) - 2*inertia_tensor(1, 2)*inertia_tensor(1, 3)*inertia_tensor(2, 3) - &
          inertia_tensor(1, 1)*inertia_tensor(2, 2)*inertia_tensor(3, 3)

      !-------------------!
      ! Calculate Q and R !
      !-------------------!

      ! From p. 184, numerical recipes in C

      Q = onnth*a**2 - onthd*b
      
      R = ontws*a**3 - onsth*a*b + onhlf*c
      

      IF (Q**3<R**2) STOP 'Error: eignevalues not real'

      !--------------!
      ! Compute phi  !                                                                                                                      
      !--------------!                                                                                                                      

      svar = SQRT(Q**3)
      phi  = ACOS(R/svar)

      IF (Q .LT. 1e-7) phi = 0   ! Deal with repeated roots

      !-------------------------------------!
      ! Compute and sort the 3 eigenvalues  !
      !-------------------------------------!
      
      rootq = SQRT(Q)

      eig(1) = -2.0_REAL64*rootq*COS(onthd*phi)           - onthd*a
      eig(2) = -2.0_REAL64*rootq*COS(onthd*phi + twthdpi) - onthd*a
      eig(3) = -2.0_REAL64*rootq*COS(onthd*phi - twthdpi) - onthd*a

      DO ii = 1,3
         jj      = MINLOC(eig, 1)
         es(ii)  = eig(jj)
         eig(jj) = HUGE(1.0_REAL64)
      END DO
      eig = es

      eig = eig/eig(3)

      !WRITE(*,*) es
    
      !cluster_shape = (eig(3) - eig(1))/eig(3)

      cluster_shape = eig(3)**2 - 0.5*(eig(1)**2+eig(2)**2)

      !WRITE(*,*) cluster_shape

      
    END FUNCTION CLUSTER_SHAPE


    FUNCTION CONNECTED_ATOM(O_pos, neigh_pos, uc_len, uc_inv)
      ! =============================================== !
      ! Determine the closest image of a connected atom !
      ! =============================================== !
      REAL(KIND=REAL64), INTENT(IN)               :: uc_len, uc_inv
      REAL(KIND=REAL64), DIMENSION(3), INTENT(IN) :: O_pos, neigh_pos
      REAL(KIND=REAL64), DIMENSION(3)             :: tmpvec, connected_atom
      INTEGER                                     :: i
      

      tmpvec = neigh_pos - O_pos
      DO i = 1, 3
         tmpvec(i) = tmpvec(i) - 1.0_REAL64*uc_len*ANINT(tmpvec(i)*1.0_REAL64*uc_inv)
      END DO

      connected_atom = O_pos + tmpvec
      
    END FUNCTION CONNECTED_ATOM
               
END MODULE CLUSTERINF





