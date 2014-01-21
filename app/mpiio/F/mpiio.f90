program mpiio

  use mpi

  implicit none

!
!  The global data size is nx x ny
!

  integer, parameter :: nx = 480
  integer, parameter :: ny = 216

!
!  The processes are in a 2D array of dimension XPROCS x YPROCS, with
!  a total of NPROCS processes
!

  integer, parameter :: ndim = 2

  integer, parameter :: xprocs = 1
  integer, parameter :: yprocs = 1

  integer, parameter :: nprocs = xprocs*yprocs

!
!  The local data size is NXP x NYP
!

  integer, parameter :: nxp = nx/xprocs
  integer, parameter :: nyp = ny/yprocs

!
!  The maximum length of a file name
!

  integer, parameter :: maxfilename = 64

!
!  pcoords stores the grid positions of each process
!

  integer, dimension(ndim, nprocs) :: pcoords

!
!  buf is the large buffer for the master to read into
!  x contains the local data only
!

  real, dimension(nx,  ny)  :: buf
  real, dimension(nxp, nyp) :: x

  integer :: rank, size, ierr
  integer :: i, j

  character*(maxfilename) :: filename

  integer :: comm = MPI_COMM_WORLD

  call MPI_INIT(ierr)

  call MPI_COMM_SIZE(comm, size, ierr)
  call MPI_COMM_RANK(comm, rank, ierr)

!
!  Check we are running on the correct number of processes
!

  if (size .ne. nprocs) then

     if (rank .eq. 0) then
        write(*,*) 'ERROR: compiled for ', nprocs,    &
                   ' process(es), runnning on ', size
     end if

     call MPI_FINALIZE(ierr)
     stop

  end if

!
!  Work out the coordinates of all the processes in the grid and
!  print them out
!

  call initpgrid(pcoords, xprocs, yprocs)

  if (rank .eq. 0) then

     write(*,*) 'Running on ', nprocs, ' process(es) in a ', &
                xprocs, ' x ', yprocs, ' grid'
     write(*,*)

     do i = 0, nprocs-1
       write(*,*) 'Process ', i, ' has grid coordinates (', &
                  pcoords(1,i+1), ', ', pcoords(2,i+1), ')'
     end do

     write(*,*)

  end if
  
!
!  Initialise the arrays to a grey value
!

  call initarray(buf, nx,  ny )
  call initarray(x,   nxp, nyp)

!
!  Read the entire array on the master process
!  Passing "-1" as the rank argument means that the file name has no
!  trailing "_rank" appended to it, ie we read the global file
!

  if (rank .eq. 0) then

    call createfilename(filename, 'finput', nx, ny, -1)
    call ioread (filename, buf, nx*ny)
    write(*,*)

  end if

!
!  Simply copy the data from buf to x
!

  do i = 1, nxp
    do j = 1, nyp

      x(i,j) = buf(i,j)

     end do
  end do

!
!  Every process writes out its local data array x to an individually
!  named file which as the rank appended to the file name
!

  call createfilename(filename, 'foutput', nxp, nyp, rank)
  call iowrite(filename, x, nxp*nyp)

  call MPI_FINALIZE(ierr)

end

