#include <mpi.h>
#include <stdio.h>

main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int proc;
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    printf("hello %d!\n", proc);
    MPI_Finalize();
    return 0;
}
