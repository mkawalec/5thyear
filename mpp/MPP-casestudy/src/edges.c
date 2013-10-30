#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "pgmio.h"
#include "arralloc.h"

/** Reads in the file data into a
 *  computation-ready array
 */

int process_count = 2;

void read_input(float *buf, float **new, size_t dim_x, size_t dim_y)
{
    printf("Reading input...\n");
    size_t i, j;
    for (i = 0; i < dim_x + 2; ++i) {
        for (j = 0; j < dim_y + 2; ++j) {
            if (i == 0 || i == dim_x + 1 || j == 0 || j == dim_y + 1) {
                new[i][j] = 255;
            } else {
                new[i][j] = buf[(i - 1) * dim_y + (j - 1)];
            }
        }
    }
    printf("done\n");
}

/** Set an array to the max value,
 *  including halos
 */
void initialize_array(float **array, size_t dim_x, size_t dim_y)
{
    printf("Initializing an array...\n");
    size_t i, j;
    for (i = 0; i < dim_x + 2; ++i) {
        for (j = 0; j < dim_y + 2; ++j) {
            array[i][j] = 255;
        }
    }
    printf("done\n");
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
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    size_t new_dim_x = dim_x / process_count;
    float *buf = malloc(sizeof(float) * new_dim_x * dim_y);
    float *masterbuf = malloc(sizeof(float) * dim_x * dim_y);
    float **edge = arralloc(sizeof(float), 2, new_dim_x + 2, dim_y + 2);
    float **new = arralloc(sizeof(float), 2, new_dim_x + 2, dim_y + 2);
    float **old = arralloc(sizeof(float), 2, new_dim_x + 2, dim_y + 2);

    if (rank == 0) pgmread(image_name, masterbuf, dim_x, dim_y);

    MPI_Scatter(masterbuf, new_dim_x * dim_y, MPI_FLOAT,
            buf, new_dim_x * dim_y, MPI_FLOAT, 0,
            MPI_COMM_WORLD);

    read_input(buf, edge, new_dim_x, dim_y);
    initialize_array(old, new_dim_x, dim_y);
    printf("%d\n", rank);

    // Main loop
    size_t iter;
    for (iter = 0; iter < 1000; ++iter) {
        size_t i, j;
        for (i = 1; i < new_dim_x + 1; ++i) {
            for (j = 1; j < dim_y + 1; ++j) {
                new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                        old[i][j-1] + old[i][j+1] -
                        edge[i][j]);
            }
        }
        for (i = 1; i < new_dim_x + 1; ++i) {
            for (j = 1; j < dim_y + 1; ++j) {
                old[i][j] = new[i][j];
            }
        }
    }

    size_t i, j;
    for (i = 0; i < new_dim_x; ++i) {
        for (j = 0; j < dim_y; ++j) 
            buf[i * dim_y + j] = old[i+1][j+1];
    }

    MPI_Gather(buf, new_dim_x * dim_y, MPI_FLOAT,
            masterbuf, new_dim_x * dim_y, MPI_FLOAT, 0,
            MPI_COMM_WORLD);
    MPI_Finalize();

    if (rank == 0) pgmwrite("output.pgm", masterbuf, dim_x, dim_y);
    return 0;
}
