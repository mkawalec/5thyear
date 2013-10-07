#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(){
    omp_set_num_threads(2);
#pragma omp parallel
    {
#pragma omp critical 
     {
      printf("hello from thread %d\n",omp_get_thread_num());
     }
    }
}
