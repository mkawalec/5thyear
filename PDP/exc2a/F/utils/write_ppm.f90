module write_image
  implicit none
  integer, parameter :: MAX_COLOUR_VALS = 255
  integer, parameter :: R = 1
  integer, parameter :: G = 2
  integer, parameter :: B = 3
  private
  public :: write_ppm
contains

  ! Convert a single iteration count to an RGB value by treating it
  ! as an HSV triplet with saturation and value both set to 1
  function gray_to_rgb(gray, ncolours) result(rgb)
    integer, intent(in) :: gray
    integer, intent(in) :: ncolours
    integer, dimension(3) :: rgb
    real :: h
    real :: s
    real :: v
    real :: f, p, q, t

    s = 1
    v = ncolours

    h = (360.0 * gray) / (60 * ncolours)

    if ( h .lt. 0 ) then
       ! Invalid colour, set to black
       rgb = 0
       return
    end if

    f = h - int(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    select case ( int(h) )
    case (0)
       rgb(R) = int(v)
       rgb(G) = int(t)
       rgb(B) = int(p)
    case (1)
       rgb(R) = int(q)
       rgb(G) = int(v)
       rgb(B) = int(p)
    case (2)
       rgb(R) = int(p)
       rgb(G) = int(v)
       rgb(B) = int(t)
    case (3)
       rgb(R) = int(p)
       rgb(G) = int(q)
       rgb(B) = int(v)
    case (4)
       rgb(R) = int(t)
       rgb(G) = int(p)
       rgb(B) = int(v)
    case (5)
       rgb(R) = int(v)
       rgb(G) = int(p)
       rgb(B) = int(q)
    case default
       rgb = 0
    end select
  end function gray_to_rgb

  ! Write a PPM file containing the pixels in IMAGE to FILE
  !
  ! IMAGE is considered as a set of grayscale values that are
  ! converted to RGB by mapping them onto HSV.
  subroutine write_ppm(file, image, max_iter)
    character(len=*), intent(in):: file
    integer, dimension(:,:), intent(in) :: image
    integer, intent(in) :: max_iter
    integer :: i, j
    integer :: ncolours
    ncolours = MAX_COLOUR_VALS

    ! PPM format is:
    ! P3
    ! WIDTH HEIGHT
    ! MAX_COLOURS
    ! R G B
    ! R G B
    ! ...
    !
    ! All RGB values must be <= MAX_COLOURS
    if ( max_iter .lt. ncolours ) then
       ncolours = max_iter
    end if

    open(unit=10, file=file)

    write(10, '(A2)') 'P3'
    write(10, *) size(image, 1), size(image, 2)
    write(10, *) ncolours

    do j = size(image, 2), 1, -1
       do i = 1, size(image, 1)
          write(10, *)gray_to_rgb(image(i, j), ncolours)
       end do
    end do

    close(unit=10)

  end subroutine write_ppm
end module write_image
