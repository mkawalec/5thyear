#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 729
#define reps 100 
#include <omp.h> 

double a[N][N], b[N][N], c[N];
int jmax[N];  


void init1(void);
void init2(void);
void runloop(int); 
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);


int main(int argc, char *argv[]) { 

    double start1,start2,end1,end2;
    int r;

    init1(); 

    start1 = omp_get_wtime(); 

    for (r=0; r<reps; r++){ 
        runloop(1);
    } 

    end1  = omp_get_wtime();  

    valid1(); 

    printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1)); 


    init2(); 

    start2 = omp_get_wtime(); 

    for (r=0; r<reps; r++){ 
        runloop(2);
    } 

    end2  = omp_get_wtime(); 

    valid2(); 

    printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2)); 

} 

void init1(void){
    int i,j; 

    for (i=0; i<N; i++){ 
        for (j=0; j<N; j++){ 
            a[i][j] = 0.0; 
            b[i][j] = 3.142*(i+j); 
        }
    }

}

void init2(void){ 
    int i,j, expr; 

    for (i=0; i<N; i++){ 
        expr =  i%( 3*(i/30) + 1); 
        if ( expr == 0) { 
            jmax[i] = N;
        }
        else {
            jmax[i] = 1; 
        }
        c[i] = 0.0;
    }

    for (i=0; i<N; i++){ 
        for (j=0; j<N; j++){ 
            b[i][j] = (double) (i*j+1) / (double) (N*N); 
        }
    }

} 

struct Chunk {
    int start, end;
};

void runloop(int loopid)  {
    struct Chunk *chunks;
    omp_lock_t *chunk_locks;


#pragma omp parallel default(none) shared(loopid, chunks, chunk_locks) 
    {
        int myid  = omp_get_thread_num();
        int nthreads = omp_get_num_threads(); 
        int ipt = (int) ceil((double)N/(double)nthreads);
        int steal_from;

#pragma omp single
        {
            chunks = malloc(sizeof(struct Chunk) * nthreads);
            chunk_locks = malloc(sizeof(omp_lock_t) * nthreads);
        }

        int lo = myid*ipt;
        int hi = (myid+1)*ipt;
        if (hi > N) hi = N; 

        // Initialize the chunk sizes for all threads
        chunks[myid].start = lo;
        chunks[myid].end = hi;

#pragma omp barrier
        while (1) {
            int start, end, thread_id;
            omp_set_lock(&chunk_locks[myid]);
            if (chunks[myid].start < chunks[myid].end) {
                start = chunks[myid].start;
                end = start + (int)ceil((chunks[myid].end - chunks[myid].start)/
                        (double)nthreads);
                chunks[myid].start = end;
                thread_id = myid;

                omp_unset_lock(&chunk_locks[myid]);
            } else if ((steal_from = get_most_loaded(chunks, nthreads)) != -1) {
                omp_unset_lock(&chunk_locks[myid]);
                omp_set_lock(&chunk_locks[steal_from]);
                start = chunks[steal_from].start;
                end = start + (int)ceil((chunks[steal_from].end - chunks[steal_from].start)/
                        (double)nthreads);
                chunks[steal_from].start = end;

                omp_unset_lock(&chunk_locks[steal_from]);
            } else {
                omp_unset_lock(&chunk_locks[myid]);
                break;
            }

            // Since we don't want to put a critical section
            // on get_most_loaded
            if (start == end) 
                break;

            switch (loopid) { 
                case 1: loop1chunk(start, end); break;
                case 2: loop2chunk(start, end); break;
            } 
        }
    }

    // Let the memory graze free!
    free(chunks);
    free(chunk_locks);
}

int get_most_loaded(struct Chunk *chunks, int nthreads)
{
    int most_loaded = -1;
    int difference = 0;

    int i;
    for (i = 0; i < nthreads; ++i) {
        if (chunks[i].end - chunks[i].start > difference) {
            most_loaded = i;
            difference = chunks[i].end - chunks[i].start;
        }
    }

    return most_loaded;
}

void loop1chunk(int lo, int hi) { 
    int i,j; 

    for (i=lo; i<hi; i++){ 
        for (j=N-1; j>i; j--){
            a[i][j] += cos(b[i][j]);
        } 
    }

} 



void loop2chunk(int lo, int hi) {
    int i,j,k; 
    double rN2; 

    rN2 = 1.0 / (double) (N*N);  

    for (i=lo; i<hi; i++){ 
        for (j=0; j < jmax[i]; j++){
            for (k=0; k<j; k++){ 
                c[i] += (k+1) * log (b[i][j]) * rN2;
            } 
        }
    }

}

void valid1(void) { 
    int i,j; 
    double suma; 

    suma= 0.0; 
    for (i=0; i<N; i++){ 
        for (j=0; j<N; j++){ 
            suma += a[i][j];
        }
    }
    printf("Loop 1 check: Sum of a is %lf\n", suma);

} 


void valid2(void) { 
    int i; 
    double sumc; 

    sumc= 0.0; 
    for (i=0; i<N; i++){ 
        sumc += c[i];
    }
    printf("Loop 2 check: Sum of c is %f\n", sumc);
} 


