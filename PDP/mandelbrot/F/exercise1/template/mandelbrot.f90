program mandelbrot

  use write_image
  use read_opt
  implicit none

  integer, allocatable, dimension(:,:) :: image

  integer :: grid_size_x, grid_size_y
  integer :: max_iter
  complex :: cmin
  complex :: cmax

  call read_options(grid_size_x, grid_size_y, max_iter, cmin, cmax)

  call initialise_image(image, grid_size_x, grid_size_y)

  ! Compute the mandelbrot set here and write results into image
  ! array.  Note that the image writing code assumes the following
  ! mapping from array entries to pixel positions in the image:
  !
  ! A == image(1, 1)
  ! B == image(1, grid_size_u)
  ! C == image(grid_size_x, grid_size_y)
  ! D == image(grid_size_x, 1)
  !
  !         B-----------------C
  !         |                 |
  !         |                 |
  !         A-----------------D

  call write_ppm("output.ppm", image, max_iter)

  deallocate(image)

contains

  ! Initialise the data for the image array.  Data is stored in
  ! "scanline order", i.e. x dimension varies fastest.  You get an
  ! array with shape (grid_size_x, grid_size_y) from this function
  subroutine initialise_image(image, grid_size_x, grid_size_y)
    integer, intent(out), allocatable, dimension(:,:) :: image
    integer, intent(in) :: grid_size_x, grid_size_y

    allocate(image(grid_size_x, grid_size_y))
    image = -1
  end subroutine initialise_image

end program mandelbrot
