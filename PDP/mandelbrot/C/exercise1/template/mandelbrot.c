#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <error.h>
#include <math.h>

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

void mandelbrot(int ***image,
        const int grid_size_x,
        const int grid_size_y,
        const int max_iter,
        const float x_min,
        const float x_max,
        const float y_min,
        const float y_max)
{
    int i, j, k;
    double tmp[2], norm;

    for (j = 0; j < grid_size_y; ++j) {
        for (i = 0; i < grid_size_x; ++i) {
            double re = x_min + i * (x_max - x_min) / (double)grid_size_x,
                   im = y_min + j * (y_max - y_min) / (double)grid_size_y;

            for (k = 0; k < max_iter; ++k) {
                tmp[0] = pow(re, 2) - pow(im, 2);
                tmp[1] = 2 * re * im;
                re = tmp[0];
                im = tmp[1];
                //printf("%lf\n", re);

                if (sqrt(pow(re, 2) + pow(im, 2)) > 1) {
                    (*image)[j][i] = k;
                    break;
                }
            }
        }
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

    mandelbrot(&image, grid_size_x, grid_size_y, max_iter,
            xmin, xmax, ymin, ymax);

    /* Compute the mandelbrot set here and write results into image
     * array.  Note that the image writing code assumes the following
     * mapping from array entries to pixel positions in the image:
     *
     * A == image[0][0]
     * B == image[grid_size_y-1][0]
     * C == image[grid_size_y-1][grid_size_x-1]
     * D == image[0][grid_size_x-1]
     *
     *         B-----------------C
     *         |                 |
     *         |                 |
     *         A-----------------D
     */

    write_ppm("output.ppm", image, grid_size_x, grid_size_y, max_iter);

    free(image);
    return 0;
}
