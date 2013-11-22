#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "helpers.h"

void read_input(float *buf, float **new, size_t dim_x, size_t dim_y)
{
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
}

/** Set an array to the max value,
 *  including halos
 */
void initialize_array(float **array, size_t dim_x, size_t dim_y)
{
    size_t i, j;
    for (i = 0; i < dim_x + 2; ++i) {
        for (j = 0; j < dim_y + 2; ++j) {
            array[i][j] = 255;
        }
    }
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
    get_pos(rank, dim_x, dim_y, size_x, size_y,
        &start_x, &start_y, &end_x, &end_y);

    int count = end_x - start_x + 1;
    int blocklength = end_y - start_y + 1;
    int stride = dim_y;

    MPI_Datatype current_type;
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
                float **receive_buf, size_t *receive_x, size_t *receive_y)
{
    // Number of parts
    int comm_size, size_x, size_y, i, rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &rank);

    struct pair optimal = get_decomposition(dim_x, dim_y, comm_size);

    MPI_Request *requests;
    if (rank == 0) {
        requests = malloc(sizeof(MPI_Request) * comm_size);
        struct pair dims = get_dims(dim_x, dim_y, comm_size);

        for (i = 0; i < comm_size; ++i) {
            size_t start_x, start_y, end_x, end_y;

            MPI_Datatype exchange_dtype = create_dtype(i, dim_x, dim_y,
                    optimal.first, optimal.second);
            MPI_Type_commit(&exchange_dtype);

            get_pos(i, dim_x, dim_y, optimal.first, optimal.second,
                &start_x, &start_y, &end_x, &end_y);
            MPI_Issend(&input[start_x  * dim_y + start_y], 1, exchange_dtype, 
                       i, 0, communicator, &requests[i]);
        }
    }

    size_t start_x, start_y, end_x, end_y;
    get_pos(rank, dim_x, dim_y, optimal.first, optimal.second,
        &start_x, &start_y, &end_x, &end_y);

    size_t buffer_size = (end_x - start_x + 1) * (end_y - start_y + 1);
    float *tmp_buf = malloc(sizeof(float) * buffer_size);
    *receive_buf = malloc(sizeof(float) * buffer_size);
    *receive_x = end_x - start_x + 1;
    *receive_y = end_y - start_y + 1;
    MPI_Recv(tmp_buf, buffer_size, MPI_FLOAT, 0, 0, communicator, NULL);

    int row_height = end_y - start_y + 1;
    for (i = 0; i < buffer_size; ++i) {
        int index = row_height - 1 - i%row_height + (i / row_height) * row_height;
        (*receive_buf)[i] = tmp_buf[index];
    }
    free(tmp_buf);

    if (rank == 0) {
        for (i = 0; i < comm_size; ++i) 
            MPI_Wait(&requests[i], NULL);

        free(requests);
    }
}

void my_gather(float *input, size_t input_x, size_t input_y, MPI_Comm communicator,
               float *receive_buf, size_t receive_x, size_t receive_y)
{
    // Send the data in the buffer to rank 0
    int rank, i, comm_size;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &comm_size);
    struct pair optimal = get_decomposition(receive_x, receive_y, comm_size);

    size_t start_x, start_y, end_x, end_y;
    get_pos(rank, receive_x, receive_y, optimal.first, optimal.second,
        &start_x, &start_y, &end_x, &end_y);
    int buffer_size = (end_x - start_x + 1) * (end_y - start_y + 1);
    int row_height = end_y - start_y + 1;
    float *send_buf = malloc(sizeof(float) * buffer_size);

    for (i = 0; i < buffer_size; ++i) {
        int index = row_height - 1 - i%row_height + (i / row_height) * row_height;
        send_buf[i] = input[index];
    }

    MPI_Request send_request;
    MPI_Issend(send_buf, input_x * input_y, MPI_FLOAT, 0, 0, communicator,
            &send_request);

    if (rank == 0) {
        // Put the received data in the right places 
        // in the receive buffer
        size_t i;
        int comm_size;
        MPI_Comm_size(communicator, &comm_size);

        struct pair optimal = get_decomposition(receive_x, receive_y, comm_size);
        struct pair dims = get_dims(receive_x, receive_y, comm_size);

        for (i = 0; i < comm_size; ++i) {
            int coords[2];
            MPI_Cart_coords(communicator, i, 2, coords);

            MPI_Datatype exchange_dtype = create_dtype(i, receive_x,
                    receive_y, optimal.first, optimal.second);
            MPI_Type_commit(&exchange_dtype);

            size_t start_x, start_y, end_x, end_y;
            get_pos(i, receive_x, receive_y, optimal.first, optimal.second,
                &start_x, &start_y, &end_x, &end_y);

            MPI_Recv(&receive_buf[start_x * receive_y + start_y], 1, 
                    exchange_dtype, i, 0, communicator, NULL);
        }
    }

    MPI_Wait(&send_request, NULL);
    free(send_buf);
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
