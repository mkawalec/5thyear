#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <limits.h>

#include "pgmio.h"
#include "arralloc.h"

/** Reads in the file data into a
 *  computation-ready array
 */

#define BUFFSIZE 100000

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

/*! Computes the total length of the circumference for
 *  a decomposition of different sizes
 */
size_t circ(size_t dim_x, size_t dim_y, size_t part_x, size_t part_y)
{
    size_t length = 0, x_position, y_position;
    for (x_position = 0; x_position < dim_x; x_position += part_x) {
        for (y_position = 0; y_position < dim_y; y_position += part_y) {
            if (y_position + part_y < dim_y) {
                length += 2 * part_y;
            } else {
                length += 2 * (dim_y - y_position);
            }

            if (x_position + part_x < dim_x) {
                length += 2 * part_x;
            } else {
                length += 2 * (dim_x - x_position);
            }
        }
    }

    return length;
}

/*! Returns the top upper corner and first bottom right coordinate 
 * that is not in its domain (useful for < comparison).
 */
void my_pos(int rank, size_t dim_x, size_t dim_y, size_t part_x, size_t part_y,
        size_t *start_x, size_t *start_y, size_t *end_x, size_t *end_y) {

    size_t square_num = 0, x_position, y_position;

    for (x_position = 0; x_position < dim_x; x_position += part_x) {
        for (y_position = 0; y_position < dim_y; y_position += part_y) {
            if (square_num == rank) {
                *start_x = x_position;
                *start_y = y_position;

                if (y_position + part_y < dim_y) {
                    *end_y = y_position + part_y;
                } else {
                    *end_y = dim_y - 1;
                }
                if (x_position + part_x < dim_x) {
                    *end_x = x_position + part_x;
                } else {
                    *end_x = dim_x - 1;
                }

                printf("Hi from rank %d %ld %ld %ld %ld\n", square_num,
                        *start_x, *start_y, *end_x, *end_y);
                return;
            }
            square_num += 1;
        }
    }
}


void scatter(float *input, size_t dim_x, size_t dim_y, MPI_Comm communicator)
{
    // Number of parts
    int comm_size, size_x, size_y, i, rank;
    size_t circumference = ULONG_MAX;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &rank);

    for (i = 1; i < sqrt(comm_size) + 1; ++i) {
        if (comm_size%i == 0) {
            int current_x, current_y;
            size_t temp_circ;
            current_x = ceil(dim_x / (double)i);
            current_y = ceil(dim_y / (double) (comm_size / i));

            if ((temp_circ = circ(dim_x, dim_y, current_x, current_y)) < circumference) {
                circumference = temp_circ;
                size_x = current_x;
                size_y = current_y;
            }
        }
    }

    size_t start_x, start_y, end_x, end_y;
    my_pos(rank, dim_x, dim_y, size_x, size_y,
           &start_x, &start_y, &end_x, &end_y);

    printf("%d %d\n", size_x, size_y);

}

int main(int argc, char *argv[])
{
    /** The image filename is provided as the first
     *  parameter to the program.
     */
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    double t1 = MPI_Wtime();
    scatter(NULL, 100, 100, MPI_COMM_WORLD);
    printf("Time %lf\n", MPI_Wtime() - t1);
    return 0;

    if (argc < 4) {
        if (rank == 0) 
            printf("This program is run as follows: %s image_name dim_x dim_y\n\n",
                argv[0]);
        MPI_Finalize();
        return -1;
    }
    // Parsing the command line parameters
    char *image_name = argv[1];
    size_t dim_x = strtoul(argv[2], NULL, 10);
    size_t dim_y = strtoul(argv[3], NULL, 10);

    MPI_Buffer_attach(malloc(BUFFSIZE), BUFFSIZE);

    int dims[1] = {process_count};
    int periods[1] = {0};
    MPI_Comm proc_topology;
    MPI_Cart_create(MPI_COMM_WORLD, 
            1, &(dims[0]), &(periods[0]), 1, &proc_topology);
    MPI_Comm_rank(proc_topology, &rank);

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

    int left_rank, right_rank;
    MPI_Cart_shift(proc_topology, 0, 1, &left_rank, &right_rank);

    // Main loop
    size_t iter, i, j;
    for (iter = 0; iter < 10000; ++iter) {
        // Sync the halos
        MPI_Request right_req, left_req;

        MPI_Ibsend(&(old[new_dim_x][0]), dim_y, MPI_FLOAT,
                right_rank, 0, proc_topology, &right_req);
        MPI_Ibsend(&(old[1][0]), dim_y, MPI_FLOAT,
                left_rank, 0, proc_topology, &right_req);

        MPI_Recv(&(old[0][0]), dim_y, MPI_FLOAT,
                left_rank, 0, proc_topology, NULL);
        MPI_Recv(&(old[new_dim_x + 1][0]), dim_y, MPI_FLOAT,
                right_rank, 0, proc_topology, NULL);


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
