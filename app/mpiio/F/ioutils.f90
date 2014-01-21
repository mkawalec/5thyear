subroutine createfilename(filename, basename, nx, ny, rank)

  implicit none

  character*(*) :: filename, basename

  integer :: nx, ny, rank

  if (rank .lt. 0) then

    write(filename, fmt='(i4.4,''x'',i4.4,''.dat'')') nx, ny

  else

    write(filename, fmt='(i4.4,''x'',i4.4,''_'',i2.2,''.dat'')')  nx, ny, rank

  end if

  filename = basename//filename

end

subroutine iosize(filename, nx, ny)

  implicit none

  character*(*) :: filename
  integer       :: nx, ny

  integer :: i

  do i = 1, len(filename)

    if (iachar(filename(i:i)) .ge. iachar('0') .and. &
        iachar(filename(i:i)) .le. iachar('9')         ) exit

  end do

  if (i > len(filename)) then

    write(*,*) 'iosize: error parsing filename ', filename

    nx = -1
    ny = -1

  else

    read(filename(i:len(filename)), fmt='(i4.4,''x'',i4.4,''.dat'')') nx, ny

  end if

end



subroutine ioread(filename, data, nreal)

  implicit none

  integer, parameter :: iounit   = 10
  integer, parameter :: realsize = 4

  character*(*) :: filename
  integer       :: nreal
  real          :: data(nreal)

  integer :: i

  write(*,*) 'ioread: reading ', filename

  open(unit=iounit, file=filename, form='unformatted', &
       access='direct', recl=realsize)

  do i = 1, nreal
    read(unit=iounit, rec=i) data(i)
  end do

  close(unit=iounit)

  write(*,*) '.. done'

end

subroutine iowrite(filename, data, nreal)

  implicit none

  integer, parameter :: iounit   = 10
  integer, parameter :: realsize = 4

  character*(*) :: filename
  integer       :: nreal
  real          :: data(nreal)

  integer :: i

  write(*,*) 'iowrite: writing ', filename

  open(unit=iounit, file=filename, form='unformatted', &
       access='direct', recl=realsize)

  do i = 1, nreal
    write(unit=iounit, rec=i) data(i)
  end do

  close(unit=iounit)

  write(*,*) '.. done'

end

subroutine initarray(data, nx, ny)

  implicit none

  real, parameter :: initdataval = 0.5

  integer :: nx, ny

  real data(nx*ny)

  integer :: i

  do i = 1, nx*ny
    data(i) = initdataval
  end do

end

subroutine initpgrid(pcoords, nxproc, nyproc)

  implicit none

  include 'mpif.h'

  integer, parameter :: ndim = 2

  integer :: nxproc, nyproc
  integer, dimension(ndim, nxproc*nyproc) :: pcoords

  integer, dimension(ndim) :: dims, periods

  integer :: i, ierr
  integer :: comm = MPI_COMM_WORLD
  integer :: gridcomm

  logical :: reorder

  periods = (/ 0, 0 /)
  reorder = .false.

  dims(1) = nxproc
  dims(2) = nyproc

  call MPI_CART_CREATE(comm, ndim, dims, periods, reorder, gridcomm, ierr)

  do i = 1, nxproc*nyproc

    call MPI_CART_COORDS(gridcomm, i-1, ndim, pcoords(1, i), ierr)

  end do

  call MPI_COMM_FREE(gridcomm, ierr)

end
