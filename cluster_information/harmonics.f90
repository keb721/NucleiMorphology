MODULE harmonics

  ! This module computes the REAL spherical harmonics when given theta and phi
  ! In order to determine this the associated Legendre polynomials and factorials must be 
  ! calculated

  INTEGER, PARAMETER :: REAL64 = SELECTED_REAL_KIND(15, 307)

  IMPLICIT NONE
  
  ! The function LEGENDRE follows the format laid out in "Numerical Recipes in C, Second Edition"
  ! to compute the associated Legendre polynomial in order to determine the spherical harmonics

  CONTAINS

    ! When passing to this function assure correctness of m

    FUNCTION LEGENDRE(x, l, m)
      REAL(KIND=REAL64), INTENT(IN) :: x
      INTEGER, INTENT(IN)           :: l, m
      REAL(KIND=REAL64)             :: legendre
      REAL(KIND=REAL64)             :: pmm, pmm1, pll, sx2, fact
      INTEGER                       :: i, ll

      pmm = 1.0_REAL64
      
      IF (m .NE. 0) THEN
         sx2  = SQRT((1.0_REAL64 - x)*(1.0_REAL64 + x))
         fact = 1.0_REAL64
         DO i = 1, m
            pmm  = -1.0_REAL64 * pmm * fact * sx2
            fact = fact + 2.0_REAL64
         ENDDO
      ENDIF
    
      IF (l .EQ. m) THEN
         legendre = pmm
      ELSE
         pmm1 = pmm*x*(2*m+1)
         IF (l .EQ. (m+1)) THEN
            legendre = pmm1
         ELSE
            DO ll = m+2, l
               pll  = (x*(2*ll-1.0_REAL64)*pmm1 - (ll+m-1_REAL64)*pmm)/(ll-m)
               pmm  = pmm1
               pmm1 = pll
            ENDDO
            legendre = pll
         ENDIF
      ENDIF
    END FUNCTION LEGENDRE


    ! Find the factorials from 1! to N! up to 24! (As many as are needed for q12)

    FUNCTION FACTORIAL(N)
      INTEGER, INTENT(IN)            :: N
      INTEGER, DIMENSION(0:23), SAVE :: factorials = 1
      INTEGER, SAVE                  :: max = 2
      INTEGER                        :: i, factorial
      
      DO i = max, N
         factorials(i) = factorials(i-1)*i
      ENDDO
      
      factorial = factorials(N)

    END FUNCTION FACTORIAL
    
    ! Define the REAL spherical harmonics Y 
    ! Definition from Blanco, Florez and Bermejo, 1997, J Mol Struc 419, 19-27

    
    FUNCTION SPHERICAL(l, m, theta, phi)
      INTEGER, INTENT(IN)           :: l, m
      REAL(KIND=REAL64), INTENT(IN) :: theta, phi
      REAL(KIND=REAL64)             :: spherical, legdr
      REAL(KIND=REAL64), PARAMETER  :: invfrpi = 1.0_REAL64/(16.0_REAL64*ATAN(1.0_REAL64)), &
                                       rttwo = SQRT(2.0_REAL64)
      INTEGER                       :: mabs

      IF (m .NE. 0) THEN
         mabs      = ABS(m)
         spherical = SQRT(invfrpi*(2*l+1)*FACTORIAL(l-mabs)/FACTORIAL(l+mabs))
         legdr     = LEGENDRE(COS(theta), l, mabs)
         spherical = spherical*legdr*rttwo
         IF (m .LT. 0) THEN
            spherical = spherical*MERGE(1, -1, MOD(mabs, 2) .EQ. 0)
            spherical = spherical*SIN(phi*mabs)
         ELSE
            spherical = spherical*COS(phi*mabs)
         END IF
      ELSE 
         legdr     = LEGENDRE(COS(theta), l, m)
         spherical = legdr*SQRT((2*l+1)*invfrpi)
      ENDIF
      
    END FUNCTION SPHERICAL


    SUBROUTINE ALL_SPHERICAL(l, m, theta, phi, rl, im)
      INTEGER, INTENT(IN)            :: l, m
      REAL(KIND=REAL64), INTENT(IN)  :: theta, phi
      REAL(KIND=REAL64), INTENT(OUT) :: rl, im
      REAL(KIND=REAL64)              :: legdr, spherical
      REAL(KIND=REAL64), PARAMETER   :: invfrpi = 1.0_REAL64/(16.0_REAL64*ATAN(1.0_REAL64))
      INTEGER                        :: mabs

      IF (m .NE. 0) THEN
         mabs      = ABS(m)
         spherical = SQRT(invfrpi*(2*l+1)*FACTORIAL(l-mabs)/FACTORIAL(l+mabs))
         legdr     = LEGENDRE(COS(theta), l, mabs)
         rl        = spherical*legdr*COS(mabs*phi)
         im        = spherical*legdr*SIN(mabs*phi)
         IF (m .LT. 0) THEN
            rl = rl*MERGE(1, -1, MOD(mabs, 2) .EQ. 0)
            im = im*MERGE(-1, 1, MOD(mabs, 2) .EQ. 0)
         END IF
      ELSE 
         legdr = LEGENDRE(COS(theta), l, m)
         rl    = legdr*SQRT((2*l+1)*invfrpi)
         im    = 0
      ENDIF
    END SUBROUTINE ALL_SPHERICAL
    
END MODULE harmonics


