#include <stdio.h>
#include <math.h>
#include <mpi.h>

double compute_pi(size_t lower_bound, size_t upper_bound, size_t total)
{
    size_t i;
    double pi = 0;

    for(i = lower_bound; i < upper_bound; ++i) 
        pi += 1 / (1 + pow((i - 0.5)/total, 2));

    return 4 * pi;
}


main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    double pi = 0;
    double start_time = MPI_Wtime();
    int rank, total_processes, j;
    int iterations = 840;
    int repeats = 1000000;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    if (rank > 0) {
        for (j = 0; j < repeats; ++j){
            int iter_per_rank = iterations/(total_processes - 1);
            int start_iter = iter_per_rank * (rank - 1) + 1;
            pi = compute_pi(start_iter, start_iter + iter_per_rank, iterations);

            MPI_Ssend(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    else {
        for (j = 0; j < repeats; ++j){
            double temp_pi;
            size_t i = 0;
            MPI_Status status;

            for(i = 0; i < total_processes - 1; ++i) {
                MPI_Recv(&temp_pi, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, 
                        MPI_COMM_WORLD, &status);
                pi += temp_pi;
            }
            pi /= iterations;
        }
        printf("Total time for %d repeats is %lf\n", repeats, MPI_Wtime() - start_time);
    }


    MPI_Finalize();

    return 0;
}

