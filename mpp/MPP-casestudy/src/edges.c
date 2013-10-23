#include <stdio.h>
#include <stdlib.h>
#include "pgmio.h"

/** Reads in the file data into a
 *  computation-ready array
 */
void read_input(float *buf, float *new, size_t dim_x, size_t dim_y)
{
        size_t i, j;
        for (i = 0; i < dim_x + 2; ++i) {
                for (j = 0; j < dim_y + 2; ++j) {
                        if (i == 0 || i == dim_x || j == 0 || j == dim_y) {
                                // Set the padding to 255
                                new[i + j * (dim_x + 2)] = 255;
                        } else {
                                new[i + j * (dim_x + 2)] = buf[i - 1 + (j -1) * (dim_x + 2)];
                        }
                }
        }
}

/** Set an array to the max value,
 *  including halos
 */
void initialize_array(float *array, size_t dim_x, size_t dim_y)
{
        size_t i, j;
        for (i = 0; i < dim_x + 2; ++i) {
                for (j = 0; j < dim_y + 2; ++j) {
                        array[i + j * (dim_x + 2)] = 255;
                }
        }
}

void write_output(float *buf, float *input, size_t dim_x, size_t dim_y)
{
        size_t i, j;
        for (i = 1; i < dim_x + 1; ++i) {
                for (j = 1; j < dim_y + 1; ++j) {
                        buf[i - 1 + (j - 1) * dim_x] = input[i + j * (dim_x + 2)];
                }
        }

        pgmwrite("output.pgm", buf, dim_x, dim_y);
}


int main(int argc, char *argv[])
{
        /** The image filename is provided as the first
         *  parameter to the program.
         */
        if (argc < 4) {
                printf("This program is run as follows: %s image_name dim_x dim_y",
                                argv[0]);
                return -1;
        }
        // Parsing the command line parameters
        char **char_ptr = NULL;
        char *image_name = argv[1];
        size_t dim_x = strtoul(argv[2], char_ptr, 10);
        size_t dim_y = strtoul(argv[3], char_ptr, 10);

        float *buf = malloc(sizeof(float) * dim_x * dim_y);
        float *edge = malloc(sizeof(float) * (dim_x + 2) * (dim_y + 2));
        float *new = malloc(sizeof(float) * (dim_x + 2) * (dim_y + 2));
        float *old = malloc(sizeof(float) * (dim_x + 2) * (dim_y + 2));
        pgmread(image_name, buf, dim_x, dim_y);

        read_input(buf, edge, dim_x, dim_y);
        initialize_array(old, dim_x, dim_y);

        // Main loop
        size_t iter;
        for (iter = 0; iter < 1000; ++iter) {
                size_t i, j;
                for (i = 1; i < dim_x; ++i) {
                        for (j = 1; j < dim_y; ++j) {
                                new[i + j * (dim_x + 2)] = 0.25 * (
                                                old[i - 1 + j * (dim_x + 2)] +
                                                old[i + 1 + j * (dim_x + 2)] +
                                                old[i + (j - 1) * (dim_x + 2)] +
                                                old[i + (j + 1) * (dim_x + 2)] -
                                                edge[i + j * (dim_x + 2)]);
                        }
                }
                float *temp = new;
                new = old;
                old = temp;
        }

        write_output(buf, old, dim_x, dim_y);
        return 0;
}
