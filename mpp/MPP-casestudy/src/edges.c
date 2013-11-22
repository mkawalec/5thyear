#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#include "pgmio.h"
#include "arralloc.h"
#include "helpers.h"

/** Reads in the file data into a
 *  computation-ready array
 */

// NOTE: We don't want to be using MPI_dims_create, as
// we have no guarantee that an image will be square-like
// and we want to minimize halo swaps

int main(int argc, char *argv[])
{
    /** The image filename is provided as the first
     *  parameter to the program.
     */
    int rank, process_count;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

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

    // Generating the topology
    struct pair cart_dims = get_dims(dim_x, dim_y, process_count);
    int dims[2] = {cart_dims.first, cart_dims.second};
    int periods[2] = {0, 0};
    size_t part_x, part_y;

    MPI_Comm proc_topology;
    MPI_Cart_create(MPI_COMM_WORLD, 
            2, &(dims[0]), &(periods[0]), 1, &proc_topology);
    MPI_Comm_rank(proc_topology, &rank);

    // Reading in the input data
    float *buf;
    float *masterbuf = malloc(sizeof(float) * dim_x * dim_y);

    if (rank == 0) pgmread(image_name, masterbuf, dim_x, dim_y);

    my_scatter(masterbuf, dim_x, dim_y, proc_topology,
               &buf, &part_x, &part_y);

    // Allocating the computation arrays
    float **edge = arralloc(sizeof(float), 2, part_x + 2, part_y + 2);
    float **new = arralloc(sizeof(float), 2, part_x + 2, part_y + 2);
    float **old = arralloc(sizeof(float), 2, part_x + 2, part_y + 2);

    read_input(buf, edge, part_x, part_y);
    initialize_array(old, part_x, part_y);
    initialize_array(new, part_x, part_y);

    int left_rank, right_rank, up_rank, down_rank;
    MPI_Cart_shift(proc_topology, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(proc_topology, 1, 1, &up_rank, &down_rank);
    
    MPI_Datatype vert_type;
    MPI_Type_vector(part_x, 1, part_y + 2,
            MPI_FLOAT, &vert_type);
    MPI_Type_commit(&vert_type);

    // Main loop
    size_t iter, i, j;
    for (iter = 0; iter < 1000; ++iter) {
        if (rank == 0 && iter%500 == 0)
            printf("Doing iteration %ld\n", iter);

        // Sync the halos
        MPI_Request right_req, left_req, up_req, down_req;

        // Sends
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
                left_rank, 0, proc_topology, NULL);
        MPI_Recv(&old[part_x + 1][1], part_y, MPI_FLOAT,
                right_rank, 0, proc_topology, NULL);
        MPI_Recv(&old[1][0], 1, vert_type,
                down_rank, 0, proc_topology, NULL);
        MPI_Recv(&old[1][part_y + 1], 1, vert_type,
                up_rank, 0, proc_topology, NULL);

        for (i = 1; i < part_x + 1; ++i) {
            for (j = 1; j < part_y + 1; ++j) {
                new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                        old[i][j-1] + old[i][j+1] -
                        edge[i][j]);
            }
        }

        MPI_Wait(&right_req, NULL);
        MPI_Wait(&left_req, NULL);
        MPI_Wait(&up_req, NULL);
        MPI_Wait(&down_req, NULL);

        for (i = 1; i < part_x + 1; ++i) {
            for (j = 1; j < part_y + 1; ++j) {
                old[i][j] = new[i][j];
            }
        }
    }

    for (i = 0; i < part_x; ++i) {
        for (j = 0; j < part_y; ++j) 
            buf[i * part_y + j] = old[i+1][j+1];
    }

    // Gather!
    my_gather(buf, part_x, part_y, proc_topology,
              masterbuf, dim_x, dim_y);

    if (rank == 0) pgmwrite("output.pgm", masterbuf, dim_x, dim_y);
    MPI_Finalize();
    /*
    char rank_str[4];
    sprintf(rank_str, "%d", rank);
    const char *name = "output";
    const char *ext = ".pgm";
    char *filename = malloc(strlen(rank_str) + strlen(name) + strlen(ext));
    strcpy(filename, name);
    strcat(filename, rank_str);
    strcat(filename, ext);
    pgmwrite(filename, buf, part_x, part_y);
    */
    return 0;
}
