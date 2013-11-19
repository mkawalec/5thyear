#ifndef helpers_h
#define helpers_h

#include <mpi.h>

struct pair {
    int first, second;
};

void read_input(float *buf, float **new, size_t dim_x, size_t dim_y);
void initialize_array(float **array, size_t dim_x, size_t dim_y);
size_t circ(size_t dim_x, size_t dim_y, size_t part_x, size_t part_y);
void get_pos(int rank, size_t dim_x, size_t dim_y, size_t part_x, size_t part_y,
        size_t *start_x, size_t *start_y, size_t *end_x, size_t *end_y);
MPI_Datatype create_dtype(int rank, size_t dim_x, size_t dim_y, 
        size_t size_x, size_t size_y);
void my_scatter(float *input, size_t dim_x, size_t dim_y, MPI_Comm communicator,
                float *receive_buf, size_t *receive_x, size_t *receive_y);

struct pair get_decomposition(size_t dim_x, size_t dim_y, int comm_size);

struct pair get_dims(size_t dim_x, size_t dim_y, int comm_size);
#endif
