#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <error.h>
#include "arralloc.h"
#include "write_ppm.h"
#include "read_options.h"

/* Initialise data for image array.  Data is stored in "scanline
 * order", i.e. x dimension varies fastest.  You get an array with
 * shape image[grid_size_y][grid_size_x] from this function. */
void initialise_image(int ***image, const int grid_size_x, const int grid_size_y)
{
    int i;
    int j;
    *image = (int**)arralloc(sizeof(int), 2, grid_size_y, grid_size_x);

    if ( NULL == *image ) {
        error(1, errno, "Unable to allocate memory for image\n");
    }
    /* initalise results array to black */
    for ( i = 0; i < grid_size_y; i++ ) {
        for ( j = 0; j < grid_size_x; j++ ) {
            (*image)[i][j] = -1;
        }
    }
}

void copy_slice_to_image(int **image_slice, int **image,
                         const int slice, const int nslice,
                         const int grid_size_x, const int grid_size_y)
{
    /*
     * Copy the partial image in IMAGE_SLICE into the global IMAGE.
     *
     * The choice of how the slice number indexes the global image is
     * up to you, but you should be consistent between
     * compute_mandelbrot_slice and copy_slice_to_image.
     */
}

int **compute_mandelbrot_slice(const int slice, const int nslice,
                               const float xmin, const float xmax,
                               const float ymin, const float ymax,
                               const int grid_size_x, const int grid_size_y,
                               const int max_iter)
{
    /*
     * Compute the mandelbrot set in a SLICE of the whole domain.
     * The total number of slices is given by NSLICE.
     *
     * It is your choice whether you slice the domain the x or the y
     * direction, but think about which is going to be more efficient
     * for data access.  Recall the image is stored in scanline order
     * image[grid_size_y][grid_size_x].
     *
     * This function should return a newly initialised image slice.
     */
    return NULL;
}

void compute_mandelbrot_set(int **image,
                            const float xmin,
                            const float xmax,
                            const float ymin,
                            const float ymax,
                            const int grid_size_x,
                            const int grid_size_y,
                            const int max_iter)
{
    int slice;
    int nslice;
    int **image_slice;

    /* Arbitrary number of slices */
    nslice = 3;
    for ( slice = 0; slice < nslice; slice++ ) {
        image_slice = compute_mandelbrot_slice(slice, nslice, xmin, xmax,
                                               ymin, ymax,
                                               grid_size_x, grid_size_y,
                                               max_iter);
        copy_slice_to_image(image_slice, image, slice, nslice,
                            grid_size_x, grid_size_y);
        free(image_slice);
    }
}

int main(int argc, char** argv)
{
    int grid_size_x;
    int grid_size_y;
    int max_iter;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    int **image;

    read_options(argc, argv, &grid_size_x, &grid_size_y, &max_iter,
                 &xmin, &xmax, &ymin, &ymax);

    initialise_image(&image, grid_size_x, grid_size_y);

    compute_mandelbrot_set(image, xmin, xmax, ymin, ymax,
                           grid_size_x, grid_size_y, max_iter);

    write_ppm("output.ppm", image, grid_size_x, grid_size_y, max_iter);

    free(image);
    return 0;
}
