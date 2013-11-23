#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#include "pgmio.h"
#include "arralloc.h"
#include "helpers.h"

#define PRINT_EVERY 250
#define END_THRESHOLD 0.1

int main(int argc, char *argv[])
{
    /** 
     *  The image filename is provided as the first
     *  parameter to the program.
     */
    int rank, process_count;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    double start_time = get_time();

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

    struct pair cart_dims = get_decomposition_size(dim_x, dim_y, process_count);

    // Check if the decomposition is possible
    if (cart_dims.first * cart_dims.second != process_count) {
        if (rank == 0) printf("It is impossible to divide the image"
                              " between the provided number of processes!"
                              " Terminating.\n");
        MPI_Finalize();
        return -1;
    }

    // Prepare for topology generation
    int dims[2] = {cart_dims.first, cart_dims.second};
    int periods[2] = {0, 0};
    size_t part_x, part_y;

    float *buf;
    float *masterbuf = malloc(sizeof(float) * dim_x * dim_y);

    // Generate the topology
    MPI_Comm proc_topology;
    MPI_Cart_create(MPI_COMM_WORLD, 
            2, &(dims[0]), &(periods[0]), 0, &proc_topology);
    MPI_Comm_rank(proc_topology, &rank);

    // Read in the data and spread it around the processes
    if (rank == 0) pgmread(image_name, masterbuf, dim_x, dim_y);
    my_scatter(masterbuf, dim_x, dim_y, proc_topology,
               &buf, &part_x, &part_y);

    // Allocating the computation arrays
    float **edge = arralloc(sizeof(float), 2, part_x + 2, part_y + 2);
    float **new = arralloc(sizeof(float), 2, part_x + 2, part_y + 2);
    float **old = arralloc(sizeof(float), 2, part_x + 2, part_y + 2);

    // Read in the input from the buffer into a 2D array and
    // initialize the arrays used in computations
    read_input(buf, edge, part_x, part_y);
    initialize_array(old, part_x + 2, part_y + 2);
    initialize_array(new, part_x + 2, part_y + 2);

    // Compute the ranks of the neighbours
    int left_rank, right_rank, up_rank, down_rank;
    MPI_Cart_shift(proc_topology, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(proc_topology, 1, 1, &up_rank, &down_rank);
    
    // Generate a data type for sending the data to
    // up/down neightbours
    MPI_Datatype vert_type;
    MPI_Type_vector(part_x, 1, part_y + 2,
            MPI_FLOAT, &vert_type);
    MPI_Type_commit(&vert_type);
    MPI_Status tmp_status;

    // Main loop
    size_t iter, i, j;
    for (iter = 0; 1; ++iter) {
        /*
         * Printing statistics and checking if 
         * an end condition is met
         */
        if (iter%PRINT_EVERY == 0 && iter != 0) {
            unsigned long long pixel_sum = compute_sum(old, part_x, part_y),
                               all_pixels;
            MPI_Reduce(&pixel_sum, &all_pixels, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_SUM, 0, proc_topology);

            if (rank == 0) printf("At iteration %ld the"
                                  " average pixel value is %lld\n", 
                                  iter, all_pixels / (dim_x * dim_y));

            float max_change = compute_max_change(old, new, part_x, part_y);
            float global_max_change;
            MPI_Allreduce(&max_change, &global_max_change, 1, MPI_FLOAT,
                    MPI_MAX, proc_topology);

            if (global_max_change < END_THRESHOLD) 
                break;
        }

        // Sync the halos
        MPI_Request right_req, left_req, up_req, down_req;

        // Send the halo to left/right/up/down neighbours
        // asynchronously
        MPI_Issend(&old[part_x][1], part_y, MPI_FLOAT,
                right_rank, 0, proc_topology, &right_req);
        MPI_Issend(&old[1][1], part_y, MPI_FLOAT,
                left_rank, 0, proc_topology, &left_req);
        MPI_Issend(&old[1][1], 1, vert_type, 
                down_rank, 0, proc_topology, &down_req);
        MPI_Issend(&old[1][part_y], 1, vert_type,
                up_rank, 0, proc_topology, &up_req);

        // Receives
        MPI_Recv(&old[0][1], part_y, MPI_FLOAT,
                left_rank, 0, proc_topology, &tmp_status);
        MPI_Recv(&old[part_x + 1][1], part_y, MPI_FLOAT,
                right_rank, 0, proc_topology, &tmp_status);
        MPI_Recv(&old[1][0], 1, vert_type,
                down_rank, 0, proc_topology, &tmp_status);
        MPI_Recv(&old[1][part_y + 1], 1, vert_type,
                up_rank, 0, proc_topology, &tmp_status);

        // Apply the transformation
        for (i = 1; i < part_x + 1; ++i) {
            for (j = 1; j < part_y + 1; ++j) {
                new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                        old[i][j-1] + old[i][j+1] -
                        edge[i][j]);
            }
        }

        // Make sure the requests complete before
        // modifying the old array
        MPI_Wait(&right_req, &tmp_status);
        MPI_Wait(&left_req, &tmp_status);
        MPI_Wait(&up_req, &tmp_status);
        MPI_Wait(&down_req, &tmp_status);

        // Swap the old and new arrays
        float **tmp = old;
        old = new;
        new = tmp;
    }

    // Copy the result to output buffer
    for (i = 0; i < part_x; ++i) {
        for (j = 0; j < part_y; ++j) 
            buf[i * part_y + j] = old[i+1][j+1];
    }

    // Gather!
    my_gather(buf, part_x, part_y, proc_topology,
              masterbuf, dim_x, dim_y);

    // Write the output data to a file and exit
    if (rank == 0) {
        pgmwrite("output.pgm", masterbuf, dim_x, dim_y);
        printf("Total time taken with %d processes: %lf s\n", 
                process_count, get_time() - start_time);
    }
    MPI_Finalize();

    return 0;
}
