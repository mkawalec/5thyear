#include <stdlib.h>
#include <mpi.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

#include "helpers.h"

void read_input(float *buf, float **new, size_t dim_x, size_t dim_y)
{
    size_t i, j;
    for (i = 0; i < dim_x + 2; ++i) {
        for (j = 0; j < dim_y + 2; ++j) {
            /* Set the halos to 255 and the inner
             * values to corresponding values from buf
             */
            if (i == 0 || i == dim_x + 1 || j == 0 || j == dim_y + 1) {
                new[i][j] = 255;
            } else {
                new[i][j] = buf[(i - 1) * dim_y + (j - 1)];
            }
        }
    }
}

void initialize_array(float **array, size_t dim_x, size_t dim_y)
{
    size_t i, j;
    for (i = 0; i < dim_x; ++i) {
        for (j = 0; j < dim_y; ++j) {
            array[i][j] = 255;
        }
    }
}

unsigned long long compute_sum(float **array, size_t dim_x, size_t dim_y)
{
    unsigned long long sum = 0;
    size_t i, j;
    for (i = 1; i < dim_x + 1; ++i) {
        for (j = 1; j < dim_y + 1; ++j) {
            sum += array[i][j];
        }
    }

    return sum;
}
            
size_t circ(size_t dim_x, size_t dim_y, size_t part_x, size_t part_y)
{
    size_t length = 0, x_position, y_position;
    for (x_position = 0; x_position < dim_x; x_position += part_x) {
        for (y_position = 0; y_position < dim_y; y_position += part_y) {
            // If there is enough room to put the whole block
            // add the whole block size, if there isn't just add the
            // amout that fits
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

void get_pos(int rank, size_t dim_x, size_t dim_y, size_t part_x, size_t part_y,
        size_t *start_x, size_t *start_y, size_t *end_x, size_t *end_y) 
{
    size_t square_num = 0, x_position, 
           y_position, in_column;

    in_column = ceil(dim_y / (double) part_y);
    x_position = floor(rank / in_column) * part_x;
    y_position = rank%in_column * part_y;

    // Set the start and end values, if they
    // are requested to be ignored then ignore them
    if (start_x != NULL)
        *start_x = x_position;
    if (start_y != NULL)
        *start_y = y_position;

    if (y_position + part_y < dim_y && end_y != NULL) {
        *end_y = y_position + part_y - 1;
    } else if (end_y != NULL) {
        *end_y = dim_y - 1;
    }
    if (x_position + part_x < dim_x && end_y != NULL) {
        *end_x = x_position + part_x - 1;
    } else if (end_y != NULL) {
        *end_x = dim_x - 1;
    }
}

MPI_Datatype create_dtype(int rank, size_t dim_x, size_t dim_y, 
        size_t size_x, size_t size_y)
{
    size_t start_x, start_y, end_x, end_y;
    get_pos(rank, dim_x, dim_y, size_x, size_y,
        &start_x, &start_y, &end_x, &end_y);

    // The description of a C-indexed rectangular
    // area as MPI virtual type
    int count = end_x - start_x + 1;
    int blocklength = end_y - start_y + 1;
    int stride = dim_y;

    MPI_Datatype current_type;
    MPI_Type_vector(count, blocklength, stride,
            MPI_FLOAT, &current_type);

    return current_type;
}

struct pair get_decomposition_length(size_t dim_x, size_t dim_y, int comm_size)
{
    int i;
    size_t circumference = ULONG_MAX;
    struct pair smallest;

    for (i = 1; i < sqrt(comm_size) + 1; ++i) {
        /*
         * If the current decomposition divides number
         * of nodes (thus is possible), compute its total
         * circumference and if it is a smallest circumference
         * so far, remember that it is a smallest circumference.
         */
        if (comm_size%i == 0) {
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

struct pair get_decomposition_size(size_t dim_x, size_t dim_y, int comm_size)
{
    struct pair sizes = get_decomposition_length(dim_x, dim_y, comm_size);
    struct pair dims;
    dims.first = ceil(dim_x / (double) sizes.first);
    dims.second = ceil(dim_y / (double) sizes.second);

    return dims;
}

void invert(float **buffer, int height, int size)
{
    float *tmp_buffer = malloc(sizeof(float) * size);

    int i;
    // Invert first
    for (i = 0; i < size; ++i) {
        int index = height - 1 - i%height + (i / height) * height;
        tmp_buffer[i] = (*buffer)[index];
    }

    // Then swap the buffers and free unneeded memory
    float *tmp = *buffer;
    *buffer = tmp_buffer;
    free(tmp);
}



float compute_max_change(float **old, float **new, size_t dim_x, size_t dim_y)
{
    float max_change = -1;
    size_t i, j;
    for (i = 1; i < dim_x + 1; ++i) {
        for (j = 1; j < dim_y + 1; j++) {
            if (fabs(new[i][j] - old[i][j]) > max_change)
                max_change = fabs(new[i][j] - old[i][j]);
        }
    }

    return max_change;
}

unsigned long long hash(float *buffer, int length)
{
    unsigned long long value = 0;
    srand(123);

    int i;
    for (i = 0; i < length / 100; ++i) {
        int index = rand()%length;
        value += buffer[index] * rand();
    }

    return value;
}
   
