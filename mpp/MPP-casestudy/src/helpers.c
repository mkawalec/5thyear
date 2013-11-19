#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>
#include <math.h>

#include "helpers.h"

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

/*! Returns the bottom left and top right hand corners
 *  of an area.
 */
void get_pos(int rank, size_t dim_x, size_t dim_y, size_t part_x, size_t part_y,
        size_t *start_x, size_t *start_y, size_t *end_x, size_t *end_y) 
{
    size_t square_num = 0, x_position, y_position;

    for (x_position = 0; x_position < dim_x; x_position += part_x) {
        for (y_position = 0; y_position < dim_y; y_position += part_y) {
            if (square_num == rank) {
                *start_x = x_position;
                *start_y = y_position;

                if (y_position + part_y < dim_y) {
                    *end_y = y_position + part_y - 1;
                } else {
                    *end_y = dim_y - 1;
                }
                if (x_position + part_x < dim_x) {
                    *end_x = x_position + part_x - 1;
                } else {
                    *end_x = dim_x - 1;
                }
//                printf("Hi from rank %d %ld %ld %ld %ld\n", square_num,
//                        *start_x, *start_y, *end_x, *end_y);
                return;
            }

            ++square_num;
        }
    }
}

MPI_Datatype create_dtype(int rank, size_t dim_x, size_t dim_y, 
        size_t size_x, size_t size_y)
{
    size_t start_x, start_y, end_x, end_y;
    printf("passed params: %ld %ld %ld %ld\n", dim_x, dim_y, size_x, size_y);
    get_pos(rank, dim_x, dim_y, size_x, size_y,
        &start_x, &start_y, &end_x, &end_y);

    MPI_Datatype current_type;
    int count = end_y - start_y;
    int blocklength = end_x - start_x;
    int stride = dim_x - blocklength;

    printf("dtype %d %d %d\n", count, blocklength, stride);
    MPI_Type_vector(count, blocklength, stride,
            MPI_FLOAT, &current_type);

    return current_type;
}

/*! Finds a decomposition that minimizes
 *  the total part circumference and by that
 *  requires the least communication.
 *
 *  Returns parts dimensions
 */
struct pair get_decomposition(size_t dim_x, size_t dim_y, int comm_size)
{
    int i;
    size_t circumference = ULONG_MAX;
    struct pair smallest;

    for (i = 1; i < sqrt(comm_size) + 1; ++i) {
        if (comm_size%i == 0) {
            // X and Y part sizes
            int current_x, current_y;
            size_t temp_circ;
            current_x = ceil(dim_x / (double)i);
            current_y = ceil(dim_y / (double) (comm_size / i));

            if ((temp_circ = circ(dim_x, dim_y, current_x, current_y)) < circumference) {
                circumference = temp_circ;
                smallest.first = current_x;
                smallest.second = current_y;
            }
        }
    }

    return smallest;
}

void my_scatter(float *input, size_t dim_x, size_t dim_y, MPI_Comm communicator,
                float *receive_buf, size_t *receive_x, size_t *receive_y)
{
    // Number of parts
    int comm_size, size_x, size_y, i, rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &rank);

    struct pair optimal = get_decomposition(dim_x, dim_y, comm_size);
//    printf("optimal %d %d\n", optimal.first, optimal.second);

    MPI_Request *requests;
    if (rank == 0) {
        requests = malloc(sizeof(MPI_Request) * comm_size);
        struct pair dims = get_dims(dim_x, dim_y, comm_size);

        for (i = 0; i < comm_size; ++i) {
            /* Creating a temprorary datatype
             * specially for the given exchange.
             */

            size_t start_x, start_y, end_x, end_y;
            get_pos(rank, dim_x, dim_y, optimal.first, optimal.second,
                &start_x, &start_y, &end_x, &end_y);
            int coords[2] = {i/dims.second, i%dims.second};

            int dest_rank;
            printf("coords: %d %d %d %d\n", coords[0], coords[1], dim_x, dim_y);
            MPI_Cart_rank(communicator, coords, &dest_rank); 
            printf("ranks: %d %d\n", i, dest_rank);
            MPI_Datatype exchange_dtype = create_dtype(i, dim_x, dim_y,
                    optimal.first, optimal.second);
            MPI_Type_commit(&exchange_dtype);
            MPI_Issend(&input[start_x + dim_x * start_y], 1, exchange_dtype, 
                       dest_rank, 0, communicator, &requests[i]);
        }
        printf("sent\n");
    }

    size_t start_x, start_y, end_x, end_y;
    get_pos(rank, dim_x, dim_y, optimal.first, optimal.second,
        &start_x, &start_y, &end_x, &end_y);

    size_t buffer_size = (end_x - start_x) * (end_y - start_y);
    receive_buf = 
        malloc(sizeof(float) * buffer_size);
    printf("about to rec\n");
    *receive_x = end_x - start_x;
    *receive_y = end_y - start_y;
    MPI_Recv(receive_buf, buffer_size, MPI_FLOAT, 0, 0, communicator, NULL);


    if (rank == 0) {
        for (i == 0; i < comm_size; ++i) 
            MPI_Wait(&requests[i], NULL);
        free(requests);
    }
    printf("received\n");
}

struct pair get_dims(size_t dim_x, size_t dim_y, int comm_size)
{
    struct pair sizes = get_decomposition(dim_x, dim_y, comm_size);
    struct pair dims;
    dims.first = 0;
    dims.second = 0;
    int x_pos, y_pos;

    for (x_pos = 0; x_pos < dim_x; x_pos += sizes.first) {
        dims.first += 1;
        dims.second = 0;

        for (y_pos = 0; y_pos < dim_y; y_pos += sizes.second)
            dims.second += 1;
    }

    return dims;
}
