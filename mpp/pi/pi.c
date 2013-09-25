#include <stdio.h>
#include <math.h>
#include <mpi.h>

double compute_pi(size_t upper_bound)
{
    size_t i;
    double pi = 0;

    for(i = 1; i <= upper_bound; ++i) 
        pi += 1 / (1 + pow((i - 0.5)/upper_bound, 2));

    return 4 * pi / upper_bound;
}


main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    printf("PI is: %lf\n", compute_pi(840));
    MPI_Finalize();

    return 0;
}

