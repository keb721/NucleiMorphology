module read_dcd

  IMPLICIT NONE

  INTEGER, PARAMETER :: REAL64 = SELECTED_REAL_KIND(15, 307) ! Read UC information as a double

  !##############################################################!
  !                                                              !
  !    MODULE TO READ IN DCDS FOR ORDER PARAMETER CALCULATION    !
  !                                                              !
  !##############################################################!

  ! This module is based on the inverse of D. Quigley's dcd writing code

  CONTAINS

    SUBROUTINE read_dcd_header(dcd_file, N, snapshots, uc_present)
      ! ================================ !
      ! Read in the header of a dcd file !
      ! ================================ !

      CHARACTER(LEN=*), INTENT(IN) :: dcd_file
      INTEGER, INTENT(OUT)         :: N, snapshots 
      CHARACTER(LEN=4)             :: hdr
      INTEGER, DIMENSION(20)       :: dcd_header_info     ! called icntrl in Q's code
      INTEGER                      :: ierr, dcd=202, i
      CHARACTER(LEN=80)            :: hlp
      LOGICAL, INTENT(OUT)         :: uc_present

      OPEN(UNIT=dcd, FILE=dcd_file, STATUS='OLD', IOSTAT=ierr, FORM='UNFORMATTED', POSITION='REWIND')
      IF (ierr /= 0 ) STOP "ERROR OPENING FILE - QUITTING"
      
      READ(dcd)hdr, dcd_header_info
      
      ! Based on the data in Q's code, hdr seems to contain no information
      ! The important information is contained in the dcd_header_info
      ! and the final line (which specifies number of atoms)
      
      ! SOME QUESTIONS TO BE ADDRESSED ABOUT THE FUNCTION OF SOME OF THE PARTS OF THIS

      snapshots  = dcd_header_info(1) ! How many snapshots are in the file
      uc_present = MERGE(.TRUE., .FALSE., dcd_header_info(11)==1)
      
      READ(dcd)i, hlp       ! Seems to give information about writing => irrelevant for our purposes
      READ(dcd)N            ! Gives the number of atoms
      CLOSE(dcd)

    END SUBROUTINE READ_DCD_HEADER


    SUBROUTINE UNIT_CELLS(xtlabc, uc_mat)
      REAL(KIND=REAL64), DIMENSION(6), INTENT(IN)    :: xtlabc
      REAL(KIND=REAL64), DIMENSION(3,3), INTENT(OUT) :: uc_mat
      INTEGER                                        :: i
      REAL(KIND=REAL64), PARAMETER                   :: Pi_deg = 4*ATAN(1.0)/180
      
      ! Need to arbirtraily assign one axis to one direction.
      ! For this purpose going to assume a = sza[1, 0, 0] - 
      ! there may be an issue as VMD may be the wrong way round!

      uc_mat(1, 1)   = xtlabc(1)
      uc_mat(2:3, 1) = 0
      
      ! b and c are a little more complex. Firstly we get x components from angle cosines

      uc_mat(1, 2) = xtlabc(2)
      uc_mat(1, 3) = xtlabc(4)

      ! We get to pick b based on the constraint of the angle xtlabc(2)
      
      uc_mat(3, 2) = 0
      uc_mat(2, 2) = SQRT(1-uc_mat(1, 2)**2)

      ! This then uniquely defines c

      uc_mat(2, 3) = (xtlabc(5) - uc_mat(1, 3)*uc_mat(1, 2))/uc_mat(2, 2)
      uc_mat(3, 3) = SQRT(1 - uc_mat(1,3)**2 - uc_mat(2,3)**2)

      ! Mulitply by lengths
      
      uc_mat(:, 3) = xtlabc(6)*uc_mat(:, 3)
      uc_mat(:, 2) = xtlabc(3)*uc_mat(:, 2)


    END SUBROUTINE UNIT_CELLS
      


    SUBROUTINE READ_DCD_SNAPSHOT(dcd_file, N, snapshot, uc_present, positions, uc_vecs)
      CHARACTER(LEN=*), INTENT(IN)                    :: dcd_file
      INTEGER                                         :: ierr, dcd=202, headerlines=3, nlines, i, current, l
      INTEGER, INTENT(IN)                             :: N, snapshot 
      REAL(KIND=REAL64), DIMENSION(3, 3), INTENT(OUT) :: uc_vecs
      REAL(KIND=REAL64), DIMENSION(6)                 :: XTLABC
      REAL, DIMENSION(:,:), INTENT(OUT)               :: positions
      LOGICAL, INTENT(IN)                             :: uc_present

      OPEN(UNIT=dcd, FILE=dcd_file, STATUS='OLD', POSITION='REWIND', IOSTAT=ierr, FORM='UNFORMATTED')
      IF (ierr /= 0 ) STOP "ERROR OPENING FILE - QUITTING"

      ! As yet I can't find a robust way to start at a specified position in the file - therefore
      ! need to skip over the header and previous snapshots every time. Hopefully this shouldn't
      ! be too much of a pain as the files should be truncated 

      ! Skip over the header - 3 READ commands

      READ(dcd)
      READ(dcd)
      READ(dcd)

      ! Now for each snapshot I need to skip the perviously read snapshots but I want to avoid if statements for the 
      ! first snapshot
      
      DO current=2, snapshot  
         IF(uc_present) READ(dcd)      ! Read the hermitian matrix for this snapshot if present
         DO l=1, 3
            READ(dcd)   ! Read in the position components for this snapshot
         END DO
      END DO


      ! Now we can read in the information for this snapshot


      ! xtlabc gives information about the simulation cell
      ! the next lines give information about atomic coordinates
     
      ! This should be passed back to the calling script for passing to analysis functions 
      ! A bit further down the line, but can I make this parallelisable on the spacebat nodes
      ! even if I want to write snapshot files to work with the TCL script?


      ! I need to be careful here with using uc_vecs if it UC data doesn't exist - this will be in a 
      ! higher level program calling this module
      
      IF(uc_present) THEN  
         READ(dcd)XTLABC
         CALL UNIT_CELLS(XTLABC, uc_vecs)     ! Gives the unit cell vectors of the space
      ENDIF


      READ(dcd)(positions(1, i), i=1,N)
      READ(dcd)(positions(2, i), i=1,N)
      READ(dcd)(positions(3, i), i=1,N)
   
      CLOSE(dcd)
      


    END SUBROUTINE READ_DCD_SNAPSHOT


  END module READ_DCD
  

