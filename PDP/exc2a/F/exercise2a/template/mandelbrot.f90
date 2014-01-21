program mandelbrot

  use write_image
  use read_opt
  implicit none

  integer, allocatable, dimension(:,:) :: image

  integer :: gridsizex, gridsizey
  integer :: iter
  complex :: cmin
  complex :: cmax

  call read_options(gridsizex, gridsizey, iter, cmin, cmax)

  call initialise_image(image, gridsizex, gridsizey)

  call compute_mandelbrot_set(image, cmin, cmax, gridsizex, gridsizey, iter)

  call write_ppm("output.ppm", image, iter)

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

  subroutine copy_slice_to_image(image_slice, image, &
       slice, nslice)
    integer, dimension(:,:), intent(in) :: image_slice
    integer, dimension(:,:), intent(inout) :: image
    integer, intent(in) :: slice, nslice

    ! Copy the partial imagein IMAGE_SLICE into the global IMAGE.
    !
    ! The choice of how the slice number indexes the global image is
    ! up to you, but you should be consistent between
    ! compute_mandelbrot_slice and copy_slice_to_image.
  end subroutine copy_slice_to_image

  subroutine compute_mandelbrot_slice(image_slice, slice, nslice, &
       min, max, grid_size_x, grid_size_y, max_iter)
    integer, allocatable, dimension(:,:), intent(out) :: image_slice
    integer, intent(in) :: slice, nslice
    complex, intent(in) :: min, max
    integer, intent(in) :: grid_size_x, grid_size_y
    integer, intent(in) :: max_iter

    ! Compute the mandelbrot set in a SLICE of the whole domain.
    ! The total number of slices is given by NSLICE.
    !
    ! It is your choice whether you slice the domain in the x or the
    ! y direction, but think about which is going to be more
    ! efficient for data access.  Recall the image is stored in
    ! scanline order image(grid_size_x, grid_size_y).
    !
    ! This function should allocate the intent(out) variable
    ! image_slice and populate it with the correct values.
  end subroutine compute_mandelbrot_slice

  subroutine compute_mandelbrot_set(image, min, max, grid_size_x, &
       grid_size_y, max_iter)
    integer, dimension(:,:), intent(inout) :: image
    complex, intent(in) :: min, max
    integer, intent(in) :: grid_size_x, grid_size_y
    integer, intent(in) :: max_iter

    integer :: slice
    integer :: nslice
    integer, allocatable, dimension(:,:) :: image_slice

    ! Arbitrary number of slices
    nslice = 3

    do slice = 1, nslice
       call compute_mandelbrot_slice(image_slice, slice, nslice, &
            min, max, grid_size_x, grid_size_y, max_iter)
       call copy_slice_to_image(image_slice, image, slice, nslice)
       if ( allocated(image_slice) ) deallocate(image_slice)
    end do
  end subroutine compute_mandelbrot_set

end program mandelbrot
