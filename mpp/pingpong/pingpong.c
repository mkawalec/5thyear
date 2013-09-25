#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, i, iters,  array_size;
    unsigned int *to_send = malloc(sizeof(unsigned int) * array_size);
    unsigned int *temp_data = malloc(sizeof(unsigned int) * array_size);
    char **char_ptr;
    MPI_Status status;

    iters = strtol(argv[1], char_ptr, 10);
    array_size = strtol(argv[2], char_ptr, 10);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        memset(to_send, 0xDEADBEEF, sizeof(unsigned int) * array_size);
        double start_time = MPI_Wtime();

        for(i = 0; i < iters; ++i) {
            MPI_Ssend(&to_send, array_size, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&temp_data, array_size, MPI_UNSIGNED, 1, 0, 
                    MPI_COMM_WORLD, &status);
        }
        printf("Sends per second: %lf\n", (double) iters / (MPI_Wtime() - start_time));
    }
    else if (rank == 1) {
        for(i = 0; i < iters; ++i) {
            MPI_Recv(&temp_data, array_size, MPI_UNSIGNED, 0, 0, 
                    MPI_COMM_WORLD, &status);
            MPI_Ssend(&temp_data, array_size, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}


