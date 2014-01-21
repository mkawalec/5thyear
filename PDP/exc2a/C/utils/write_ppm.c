#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <error.h>
#include <math.h>
#include "write_ppm.h"

/*
 * Convert a single iteration count to an RGB value by treating it as
 * an HSV triplet with saturation and value both set as 1
 */
static inline void gray_to_rgb(int gray, int rgb[3], int ncolours)
{
    double h;
    double s = 1;
    double v = ncolours;
    double f, p, q, t;

    h = (360.0 * gray) / (60 * ncolours);
    if ( h < 0 ) {
        /* Invalid colour, set to black */
        rgb[R] = 0;
        rgb[G] = 0;
        rgb[B] = 0;
        return;
    }
    f = h - (int)h;

    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));
    switch ( (int)h ) {
    case 0:
        rgb[R] = (int)v;
        rgb[G] = (int)t;
        rgb[B] = (int)p;
        break;
    case 1:
        rgb[R] = (int)q;
        rgb[G] = (int)v;
        rgb[B] = (int)p;
        break;
    case 2:
        rgb[R] = (int)p;
        rgb[G] = (int)v;
        rgb[B] = (int)t;
        break;
    case 3:
        rgb[R] = (int)p;
        rgb[G] = (int)q;
        rgb[B] = (int)v;
        break;
    case 4:
        rgb[R] = (int)t;
        rgb[G] = (int)p;
        rgb[B] = (int)v;
        break;
    case 5:
        rgb[R] = (int)v;
        rgb[G] = (int)p;
        rgb[B] = (int)q;
        break;
    default:
        rgb[R] = (int)0;
        rgb[G] = (int)0;
        rgb[B] = (int)0;
    }
}

/*
 * Write a PPM file containing the xsize x ysize pixels of IMAGE to FILE.
 *
 * IMAGE is considered as a set of grayscale values that are converted
 * to RGB by mapping them onto HSV.
 */
void write_ppm(const char *file, int **image, int xsize, int ysize, int max_iter)
{
    int i;
    int j;
    FILE *fp;
    int rgb[3];
    int ncolours = MAX_COLOUR_VALS;
    fp = fopen(file,"w");

    if ( NULL == fp ) {
        error(1, errno, "Unable to open file %s for output\n", file);
    }

    /*
     * PPM format is:
     * P3
     * WIDTH HEIGHT
     * MAX_COLOURS
     * R G B
     * R G B
     * R G B
     * ...
     *
     * All RGB values must be <= MAX_COLOURS
     */
    if ( max_iter < MAX_COLOUR_VALS ) {
        ncolours = max_iter;
    }

    fprintf(fp, "P3\n%d %d\n%d\n", xsize, ysize, ncolours);

    for ( i = ysize-1; i >= 0; i-- ) {
        for ( j = 0; j < xsize; j++ ) {
            gray_to_rgb(image[i][j], rgb, ncolours);
            fprintf (fp, "%d %d %d\n", rgb[R], rgb[G], rgb[B]);
        }
    }

    fclose(fp);
}
