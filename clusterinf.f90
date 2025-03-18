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

      END DO

      tot = mean_rl

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
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)                :: pos
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
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)                :: pos
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
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)              :: pos
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
      INTEGER, DIMENSION(:), INTENT(IN)  :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                :: l
      REAL(KIND=REAL64), INTENT(IN)      :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)   :: pos
      REAL(KIND=REAL64)                  :: ql_cl
      REAL(KIND=REAL64), PARAMETER       :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(-l:l) :: qlm_bar
      INTEGER                            :: m

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
      INTEGER, DIMENSION(:), INTENT(IN)  :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                :: l
      REAL(KIND=REAL64), INTENT(IN)      :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)   :: pos
      REAL(KIND=REAL64)                  :: ql_cl
      REAL(KIND=REAL64), PARAMETER       :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(-l:l) :: qlm_bar
      INTEGER                            :: m

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
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)              :: pos
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
         IF (clusternums(atom) .NE. -1) CYCLE
         ! Already been assigned

         IF (con_mat(atom) .LT. solid_threshold) THEN
            ! Skip if number of connections under threshold, i.e. liquid
            clusternums(atom) = 0
            CYCLE
         END IF

         clusters          = clusters + 1  
         clusternums(atom) = clusters
         
         ! Atom is part of a cluster of unknown extent.
         ! Need to iterate over the neighbours of this atom, their neighbours, their neighbours neighbours etc. 

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

    FUNCTION CONNECTED_ATOM(O_pos, neigh_pos, uc_len, uc_inv)
      ! =============================================== !
      ! Determine the closest image of a connected atom !
      ! =============================================== !
      REAL(KIND=REAL64), INTENT(IN)  :: uc_len, uc_inv
      REAL(KIND=REAL64), DIMENSION(3), INTENT(IN) :: O_pos, neigh_pos
      REAL(KIND=REAL64), DIMENSION(3)             :: tmpvec, connected_atom
      INTEGER                        :: i
      

      tmpvec = neigh_pos - O_pos
      DO i = 1, 3
         tmpvec(i) = tmpvec(i) - 1.0_REAL64*uc_len*ANINT(tmpvec(i)*1.0_REAL64*uc_inv)
      END DO

      connected_atom = O_pos + tmpvec
      
    END FUNCTION CONNECTED_ATOM
               
END MODULE CLUSTERINF





