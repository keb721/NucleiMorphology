MODULE CLUSTER_FUNCS

  USE harmonics

  IMPLICIT NONE

  !#########################################################!
  !                                                         !
  !    ANALYSE CLUSTERS FOR ORDER PARAMETER CALCULATIONS    !
  !                                                         !
  !#########################################################!

  CONTAINS

    ! Need to ensure consisitency between cutoff in lattice units or LJ units
    
    FUNCTION GET_NEIGHS(pos, uc_len, uc_inv, cutoff, atom_number) RESULT(neighbours)
      ! ===================================== !
      ! Return indices of atoms within cutoff !
      ! ===================================== !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, cutoff
      REAL(KIND=REAL64), DIMENSION(3)               :: tmp_disp
      REAL                                          :: dist
      INTEGER, DIMENSION(:), ALLOCATABLE            :: neighbours
      INTEGER, INTENT(IN)                           :: atom_number
      INTEGER, DIMENSION(:), ALLOCATABLE            :: tmp
      INTEGER                                       :: neighs, i, j, lenn, N
      INTEGER, DIMENSION(2)                         :: posshape

      ! Assume a cubic simulation box, easy to change if not
      ! Assume neighbour if atom-atom distance strictly LESS THAN cutoff
      
      neighs = 0
      ALLOCATE(tmp(60))
      tmp = 0
      
      posshape = SHAPE(pos)
      N        = posshape(2)

      DO i=1,N
         IF (atom_number .EQ. i) CYCLE
         tmp_disp = pos(:,atom_number)-pos(:,i)
         DO j=1,3
            tmp_disp(j) = tmp_disp(j) - uc_len*ANINT(tmp_disp(j)*uc_inv)
         END DO
         dist = DOT_PRODUCT(tmp_disp, tmp_disp)
         dist = SQRT(dist)
         IF (dist .LE. cutoff) THEN
            tmp(neighs+1) = i
            neighs = neighs + 1
         END IF
      END DO

      ALLOCATE(neighbours(neighs))
      neighbours = tmp(1:neighs)

      DEALLOCATE(tmp)
      
    END FUNCTION GET_NEIGHS

    SUBROUTINE GET_POLARS(pos1, pos2, uc_len, uc_inv, theta, phi)
      ! ==================================== !
      ! Return spherical coordinates of atom !
      ! ==================================== !
      REAL(KIND=REAL64), DIMENSION(3), INTENT(IN) :: pos1, pos2
      REAL(KIND=REAL64), INTENT(IN)               :: uc_len, uc_inv
      REAL(KIND=REAL64), INTENT(OUT)              :: theta, phi
      REAL                                        :: x, y, z
      REAL(KIND=REAL64)                           :: twpi = 8.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(3)             :: bond
      INTEGER                                     :: j
      
      bond = pos1 - pos2
      
      DO j=1,3
         bond(j) = bond(j) - uc_len*ANINT(bond(j)*uc_inv)
      END DO

      ! Using physics convention, z=Rcos(theta)

      x = bond(1) ; y = bond(2) ; z = bond(3)

      ! Using ATAN2 to ensure correct quadrants for angles
      
      phi   = ATAN2(y*1.0_REAL64, x*1.0_REAL64)
      IF (phi .LT. 0.0_REAL64) THEN
         phi = phi+twpi
      END IF
      theta = ATAN2(SQRT((x*1.0_REAL64)**2+(y*1.0_REAL64)**2), z*1.0_REAL64)
      
    END SUBROUTINE GET_POLARS

    FUNCTION MEAN_qlm(pos, atom_number, angles, l, m) RESULT(mean)
      ! ================================== !
      ! Compute sum over qlm for all bonds !
      ! ================================== !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: angles
      INTEGER, INTENT(IN)                           :: atom_number, l, m
      REAL(KIND=REAL64)                             :: tot
      REAL(KIND=REAL64)                             :: mean
      INTEGER                                       :: neighs, i
      INTEGER, DIMENSION(2)                         :: nshape

      nshape = SHAPE(angles)
      neighs = nshape(1)

      tot    = 0.0_REAL64
      
      DO i=1, neighs
         tot = tot + SPHERICAL(l, m, angles(i,1), angles(i,2))
      END DO
      
      mean = tot/(1.0_REAL64*neighs)

    END FUNCTION MEAN_qlm

    SUBROUTINE ALL_MEAN_qlm(pos, atom_number, angles, l, m, ql_rl, ql_im)
      ! ================================== !
      ! Compute sum over qlm for all bonds !
      ! ================================== !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: angles
      INTEGER, INTENT(IN)                           :: atom_number, l, m
      REAL(KIND=REAL64)                             :: tot_rl, tot_im, rl, im
      REAL(KIND=REAL64)                             :: ql_rl, ql_im
      INTEGER                                       :: neighs, i
      INTEGER, DIMENSION(2)                         :: nshape
      

      nshape = SHAPE(angles)
      neighs = nshape(1)

      tot_rl    = 0.0_REAL64
      tot_im    = 0.0_REAL64

      DO i=1, neighs
         CALL ALL_SPHERICAL(l, m, angles(i, 1), angles(i, 2), rl, im)
         tot_im = tot_im + im
         tot_rl = tot_rl + rl
      END DO
      
      ql_rl = tot_rl/(1.0_REAL64*neighs)
      ql_im = tot_im/(1.0_REAL64*neighs)

    END SUBROUTINE ALL_MEAN_qlm

    FUNCTION Q_l_ATOM(l, pos, atom_number, neighbours, uc_len, uc_inv) RESULT(Q_l)
      !============================================================================== !
      ! Compute the Q_l number as in eq. 1.3, Steinhardt et al. 1983, PRB vol 28 no 2 !
      ! As eq.3 in Lechner and Dellago 2008, JCP 129                                  !
      ! Also as in eq. 7 ten Wolde 1996 Faraday Disc. 104                             !                     
      ! As used in LAMMPS                                                             !
      ! ============================================================================= !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)  :: pos
      REAL(KIND=REAL64), INTENT(IN)                  :: uc_len, uc_inv
      INTEGER, DIMENSION(:), INTENT(IN)              :: neighbours
      INTEGER, INTENT(IN)                            :: atom_number, l
      REAL(KIND=REAL64)                              :: sum, rl, Q_l, theta, phi
      REAL(KIND=REAL64), PARAMETER                   :: frpi = 16.0_REAL64*ATAN(1.0_REAL64)
      REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: angles
      INTEGER                                        :: m, neighs, i, j
      INTEGER, DIMENSION(1)                          :: nshape

      nshape = SHAPE(neighbours)
      neighs = nshape(1)
      
      ALLOCATE(angles(neighs, 2))
      
      DO i=1, neighs
         CALL GET_POLARS(pos(:,atom_number), pos(:,neighbours(i)), uc_len, uc_inv, theta, phi)
         angles(i, 1) = theta
         angles(i, 2) = phi
      END DO
      
      sum = 0.0_REAL64
 
      DO m=-1*l, l
         rl = MEAN_qlm(pos, atom_number, angles, l, m)
         sum = sum + rl*rl
      END DO
 
      sum = sum*frpi/(2_REAL64*l+1_REAL64)
      
      Q_l = SQRT(sum)
 
    END FUNCTION Q_l_ATOM

    FUNCTION Qlm_CLUSTER_MORONI(cluster, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(rl_cl)
      !================================================================================ !
      ! Compute the Q_l per cluster as in eq. 5 Jungblut et al. 2013, Mol Phys 111 3527 !
      ! Also as in Moroni et al. 2005 PRL 94 235703                                     !
      ! =============================================================================== !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN) :: pos
      REAL(KIND=REAL64), INTENT(IN)                 :: uc_len, uc_inv, distance_cutoff
      INTEGER, DIMENSION(:), INTENT(IN)             :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                           :: l
      REAL(KIND=REAL64), DIMENSION(-l:l)            :: rl_cl
      REAL(KIND=REAL64)                             :: theta, phi
      INTEGER                                       :: atom, neighs, i, j, Nbonds, m, Nq6
      INTEGER, DIMENSION(1)                         :: Nshape
      INTEGER, DIMENSION(:), ALLOCATABLE            :: neighbours

      ! For compuational efficiency, loop over all bonds twice (once from i, once from j)

      Nshape = SHAPE(cluster)
      Nq6    = Nshape(1)

      rl_cl  = 0.0_REAL64
      Nbonds = 0

      DO i=1, Nq6
         atom       = cluster(i)
         neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
         nshape     = SHAPE(neighbours)
         neighs     = nshape(1)
         DO j=1, neighs
            IF (COUNT(cluster .EQ. neighbours(j)) .EQ. 1) THEN
               ! This neighbour is in the cluster
               Nbonds = Nbonds + 1
               CALL GET_POLARS(pos(:,atom), pos(:,neighbours(j)), uc_len, uc_inv, theta, phi)
               DO m=-1*l, l
                  rl_cl(m) = rl_cl(m) + SPHERICAL(l, m, theta, phi)
               END DO
            END IF
         END DO
      END DO

      DO m=-1*l, l
         rl_cl(m) = rl_cl(m)/(Nbonds*1.0_REAL64)
      END DO
 
    END FUNCTION Qlm_CLUSTER_MORONI

    SUBROUTINE ALL_MEAN_qlm_CLUSTER(cluster, pos, uc_len, uc_inv, distance_cutoff, l, qlm_bar_rl, qlm_bar_im)
      ! =========================================================================== !
      ! Compute the mean Q6 vector as used in Q6cl Beckham and Peters, 2011, JPCL 2 !
      ! Uses equation equivalent to ten Wolde 1996 Faraday Disc. 104 eq 6           !
      ! =========================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)                           :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                                         :: l
      REAL(KIND=REAL64), INTENT(IN)                               :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:), ALLOCATABLE, INTENT(OUT)   :: qlm_bar_rl, qlm_bar_im
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)               :: pos
      REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE              :: angles
      REAL(KIND=REAL64)                                           :: theta, phi, ql_rl, ql_im
      INTEGER                                                     :: neighs, m, N, i, j, atom
      INTEGER, DIMENSION(1)                                       :: nshape
      INTEGER, DIMENSION(:), ALLOCATABLE                          :: neighbours
      
      Nshape = SHAPE(cluster)
      N      = Nshape(1)
      
      ALLOCATE(qlm_bar_rl(-l:l)) ; ALLOCATE(qlm_bar_im(-l:l))
      qlm_bar_rl = 0.0_REAL64 ; qlm_bar_im = 0.0_REAL64
      
      DO i=1, N

         atom       = cluster(i)
         neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
         nshape     = SHAPE(neighbours)
         neighs     = nshape(1)
         
         ALLOCATE(angles(neighs, 2))

         DO j=1, neighs
            CALL GET_POLARS(pos(:,atom), pos(:,neighbours(j)), uc_len, uc_inv, theta, phi)
            angles(j, 1) = theta
            angles(j, 2) = phi
         END DO
         
         DO m = -1*l, l
            CALL ALL_MEAN_qlm(pos, atom, angles, l, m, ql_rl, ql_im)
            qlm_bar_rl(m) = qlm_bar_rl(m) + ql_rl
            qlm_bar_im(m) = qlm_bar_im(m) + ql_im
         END DO
         
         DEALLOCATE(angles)
         
      END DO

      qlm_bar_rl = qlm_bar_rl / (1.0_REAL64*N)
      qlm_bar_im = qlm_bar_im / (1.0_REAL64*N)

    END SUBROUTINE ALL_MEAN_qlm_CLUSTER
    
    FUNCTION MEAN_qlm_CLUSTER(cluster, pos, uc_len, uc_inv, distance_cutoff, l) RESULT(qlm_bar)
      ! ====================================================================== !
      ! Compute the mean Q6 vector as used in Beckham and Peters, 2011, JPCL 2 !
      ! Uses equation equivalent to ten Wolde 1996 Faraday Disc. 104 eq 6      !
      ! ====================================================================== !
      INTEGER, DIMENSION(:), INTENT(IN)              :: cluster ! An array with the indexes of the atoms in the cluster
      INTEGER, INTENT(IN)                            :: l
      REAL(KIND=REAL64), INTENT(IN)                  :: uc_len, uc_inv, distance_cutoff      
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)  :: pos
      REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: angles
      REAL(KIND=REAL64), DIMENSION(-l:l)             :: qlm_bar
      REAL(KIND=REAL64)                              :: theta, phi
      INTEGER                                        :: neighs, m, N, i, j, atom
      INTEGER, DIMENSION(1)                          :: nshape
      INTEGER, DIMENSION(:), ALLOCATABLE             :: neighbours

      ! This is the function ALL_MEAN_qlm_cluster above for real spherical harmonics
      
      Nshape = SHAPE(cluster)
      N      = Nshape(1)

      qlm_bar = 0.0_REAL64

      DO i=1, N

         atom       = cluster(i)
         neighbours = GET_NEIGHS(pos, uc_len, uc_inv, distance_cutoff, atom)
         nshape     = SHAPE(neighbours)
         neighs     = nshape(1)
         
         ALLOCATE(angles(neighs, 2))

         DO j=1, neighs
            CALL GET_POLARS(pos(:,atom), pos(:,neighbours(j)), uc_len, uc_inv, theta, phi)
            angles(j, 1) = theta
            angles(j, 2) = phi
         END DO
         
         DO m = -1*l, l
            qlm_bar(m) = qlm_bar(m) + MEAN_Qlm(pos, atom, angles, l, m)
         END DO
         
         DEALLOCATE(angles)
         
      END DO

      qlm_bar = qlm_bar / (1.0_REAL64*N)
      
    END FUNCTION MEAN_qlm_CLUSTER

    FUNCTION VEC_Ql(l, pos, atom_number, uc_len, uc_inv, neighbours)
      ! ============================================================================== !
      ! Compute the q_l vectior as in eq. 10, ten Wolde et al. 1996, Faraday Disc, 104 !
      ! ============================================================================== !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)  :: pos
      REAL(KIND=REAL64), INTENT(IN)                  :: uc_len, uc_inv
      INTEGER, DIMENSION(:), INTENT(IN)              :: neighbours
      INTEGER, INTENT(IN)                            :: l, atom_number
      REAL(KIND=REAL64), DIMENSION(:), ALLOCATABLE   :: vec_ql
      REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE :: angles
      REAL(KIND=REAL64)                              :: sm, theta, phi, curr
      INTEGER                                        :: m, i, neighs
      INTEGER, DIMENSION(1)                          :: nshape
      
      ! vec_ql is given indexed by m

      nshape = SHAPE(neighbours)
      neighs = nshape(1)
     
      ALLOCATE(vec_ql(-l:l))
      ALLOCATE(angles(neighs, 2))

      vec_ql = 0.0_REAL64
      
      DO i=1, neighs
         CALL GET_POLARS(pos(:,atom_number), pos(:,neighbours(i)), uc_len, uc_inv, theta, phi)
         angles(i, 1) = theta
         angles(i, 2) = phi
      END DO
         
      DO m = -1*l, l
         vec_ql(m) = MEAN_Qlm(pos, atom_number, angles, l, m)
      END DO

      sm = SUM(vec_ql**2)
      sm = 1.0_REAL64/SQRT(sm)
      
      vec_ql = vec_ql*sm
  
    END FUNCTION VEC_Ql

    SUBROUTINE ALL_VEC_Ql(l, pos, atom_number, uc_len, uc_inv, neighbours, vec_ql_rl, vec_ql_im)
      ! ============================================================================== !
      ! Compute the q_l vectior as in eq. 10, ten Wolde et al. 1996, Faraday Disc, 104 !
      ! ============================================================================== !
      REAL(KIND=REAL64), DIMENSION(:,:), INTENT(IN)             :: pos
      REAL(KIND=REAL64), INTENT(IN)                             :: uc_len, uc_inv
      INTEGER, DIMENSION(:), INTENT(IN)                         :: neighbours
      INTEGER, INTENT(IN)                                       :: l, atom_number
      REAL(KIND=REAL64), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: vec_ql_rl, vec_ql_im
      REAL(KIND=REAL64), DIMENSION(:,:), ALLOCATABLE            :: angles
      REAL(KIND=REAL64)                                         :: sm, theta, phi, curr, ql_rl, ql_im
      INTEGER                                                   :: m, i, neighs
      INTEGER, DIMENSION(1)                                     :: nshape
      
      ! vec_ql is given indexed by m

      nshape = SHAPE(neighbours)
      neighs = nshape(1)
     
      ALLOCATE(vec_ql_rl(-l:l))
      ALLOCATE(vec_ql_im(-l:l))
      ALLOCATE(angles(neighs, 2))

      vec_ql_rl = 0.0_REAL64
      vec_ql_im = 0.0_REAL64
      
      DO i=1, neighs
         CALL GET_POLARS(pos(:,atom_number), pos(:,neighbours(i)), uc_len, uc_inv, theta, phi)
         angles(i, 1) = theta
         angles(i, 2) = phi
      END DO
         
      DO m = -1*l, l
         CALL ALL_MEAN_Qlm(pos, atom_number, angles, l, m, ql_rl, ql_im)
         vec_ql_rl(m) = ql_rl
         vec_ql_im(m) = ql_im
      END DO

      sm = SUM(vec_ql_im**2)+SUM(vec_ql_rl**2)
      sm = 1.0_REAL64/SQRT(sm)
      
      vec_ql_rl = vec_ql_rl*sm
      vec_ql_im = vec_ql_im*sm

    END SUBROUTINE ALL_VEC_Ql
   
END MODULE CLUSTER_FUNCS
