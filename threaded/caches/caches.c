#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const size_t amount = 1e+7;
const size_t total_ops = 1e+8;

double sum(double *array, size_t count)
{
    double sum = 0.0;

    int i = count - 1;
    for (; i >= 0; --i) 
        sum += array[i];

    return sum;
}


int main(int argc, char *argv[])
{
    double *test_array = malloc(sizeof(double) * amount);
    size_t ops = 100;

    size_t i = 0;
    for (; ops < amount; ops *= 1.1) {
        double start_time = omp_get_wtime();
        double temp = 0.0;

        size_t j = 0;
        for (; j < total_ops/ops; ++j) {
            temp += sum(test_array, ops);
        }
        printf("%ld %lf %lf\n", ops * sizeof(double), omp_get_wtime() - start_time,
                temp);
    }


    return 0;
}
