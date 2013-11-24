#include "comm.h"
#include "helpers.h"

#include <mpi.h>
#include <stdlib.h>

void my_scatter(float *input, size_t dim_x, size_t dim_y, MPI_Comm communicator,
                float **receive_buf, size_t *receive_x, size_t *receive_y)
{
    int comm_size, size_x, size_y, i, rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &rank);
    MPI_Status tmp_status;

    // Find the optimal decomposition for the given
    // starting conditions
    struct pair optimal = get_decomposition_length(dim_x, dim_y, comm_size);

    // If the process has rank zero (is a 'master' process),
    // make it send different parts to different processes
    MPI_Request *requests;
    if (rank == 0) {
        requests = malloc(sizeof(MPI_Request) * comm_size);
        struct pair dims = get_decomposition_size(dim_x, dim_y, comm_size);

        for (i = 0; i < comm_size; ++i) {
            size_t start_x, start_y, end_x, end_y;

            MPI_Datatype exchange_dtype = create_dtype(i, dim_x, dim_y,
                    optimal.first, optimal.second);
            MPI_Type_commit(&exchange_dtype);

            get_pos(i, dim_x, dim_y, optimal.first, optimal.second,
                    &start_x, &start_y, NULL, NULL);
            MPI_Issend(&input[start_x  * dim_y + start_y], 1, exchange_dtype, 
                    i, 0, communicator, &requests[i]);
        }
    }

    size_t start_x, start_y, end_x, end_y;
    get_pos(rank, dim_x, dim_y, optimal.first, optimal.second,
            &start_x, &start_y, &end_x, &end_y);

    // Set sizes of this particular part
    *receive_x = end_x - start_x + 1;
    *receive_y = end_y - start_y + 1;

    // Receive and preprocess the data
    size_t buffer_size = (end_x - start_x + 1) * (end_y - start_y + 1);
    *receive_buf = malloc(sizeof(float) * buffer_size);
    MPI_Recv(*receive_buf, buffer_size, MPI_FLOAT, 0, 0, communicator, 
            &tmp_status);

    invert(receive_buf, end_y - start_y + 1, buffer_size);

    // Wait for completion of all the requests before continuing
    if (rank == 0) {
        for (i = 0; i < comm_size; ++i) 
            MPI_Wait(&requests[i], &tmp_status);

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
    MPI_Status tmp_status;

    // Gather the information needed for buffer inversion
    // and execute the vertical invertion
    struct pair optimal = get_decomposition_length(receive_x, receive_y, comm_size);
    size_t start_x, start_y, end_x, end_y;
    get_pos(rank, receive_x, receive_y, optimal.first, optimal.second,
            &start_x, &start_y, &end_x, &end_y);
    int buffer_size = (end_x - start_x + 1) * (end_y - start_y + 1);
    invert(&input, end_y - start_y + 1, buffer_size);

    MPI_Request send_request;
    MPI_Issend(input, input_x * input_y, MPI_FLOAT, 0, 0, communicator,
            &send_request);

    if (rank == 0) {
        for (i = 0; i < comm_size; ++i) {
            // Create a dynamic per-part datatype
            MPI_Datatype exchange_dtype = create_dtype(i, receive_x,
                    receive_y, optimal.first, optimal.second);
            MPI_Type_commit(&exchange_dtype);

            // Find out where this part should be put, disregard
            // the nonneeded coordinates
            size_t start_x, start_y;
            get_pos(i, receive_x, receive_y, optimal.first, optimal.second,
                    &start_x, &start_y, NULL, NULL);

            MPI_Recv(&receive_buf[start_x * receive_y + start_y], 1, 
                    exchange_dtype, i, 0, communicator, &tmp_status);
        }
    }

    MPI_Wait(&send_request, &tmp_status);
}

