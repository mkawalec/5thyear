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

/**
 *  @brief          a definition of a iteration range, 
 *                  and a simple pair of integers
 */ 
struct Chunk {
    int start, end;
};

/**
 *  @brief          Finds a an id of the most loaded thread
 *  @param chunks   an array of unclaimed chunks
 *  @param nthreads number of threads
 *
 *
 *  This function does not perform any kind of locking
 *  and thus answer it gives is not guaranteed to be 
 *  'correct'. The only incorrectness that can occur
 *  that would have a significant impact on both execution
 *  time and the answer received occurs when a chunk
 *  to be claimed was already claimed by another process
 *  by the time this function completes.
 *
 *  This is an intended behaviour, and thus apply additional
 *  checks to the numbers returned by this function.
 */
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

void runloop(int loopid)  
{
    /* The total number of threads is acquired in this slightly odd 
     * way, as we want to allocate the later structures on the stack
     * and in a scope accessble to all threads
     */
    int nthreads; 
#pragma omp parallel default(none) shared(nthreads)
    {
#pragma omp single 
        {
            nthreads = omp_get_num_threads(); 
        }
    }
            
    /* chunks contains still claimable ranges for each
     * of the processes.
     *
     * chunk_locks is an array of locks. A lock
     * with array index the same as a process number
     * has to be set when a read or write, which results
     * will be used for computation on a given chunk range,
     * is attempted.
     */
    struct Chunk chunks[nthreads];
    omp_lock_t chunk_locks[nthreads];

    size_t i;
    for (i = 0; i < nthreads; ++i) omp_init_lock(&chunk_locks[i]);

#pragma omp parallel default(none) shared(loopid, chunks, chunk_locks, nthreads) 
    {
        int myid    = omp_get_thread_num();
        int ipt     = ceil(N / (double) nthreads);

        // Setting the upper and lower computation boundaries 
        // for this thread.
        int lo = myid * ipt;
        int hi = (myid + 1) * ipt;
        if (hi > N) hi = N; 

        /* The id of a process which chunk will be executed, if
         * the current process had finished its own chunks
         */
        int steal_from;

        /* Initialize the chunk sizes for all threads.
         * No locking is required, as each thread writes to a 
         * different part of the chunks array and the barrier ensures
         * that the threads will start computations only after
         * every thread has set its own computation region
         * boundaries.
         */
        chunks[myid].start = lo;
        chunks[myid].end = hi;

#pragma omp barrier
        while (1) {
            /* After the computation boundaries for the current iteration
             * code below finished, start will contain the computation
             * range start and end will contain computation range end.
             */
            int start, end;
            omp_set_lock(&chunk_locks[myid]);
            if (chunks[myid].start < chunks[myid].end) {
                /* If there is something to do in my own iterations 
                 * range, claim it.
                 */
                start = chunks[myid].start;
                end = start + (int)ceil((chunks[myid].end - chunks[myid].start)/
                        (double)nthreads);
                chunks[myid].start = end;

                omp_unset_lock(&chunk_locks[myid]);
            } else if ((steal_from = get_most_loaded(chunks, nthreads)) != -1) {
                /* Else, find the most loaded thread, 
                 * steal a correct amount of iterations
                 * and unset the locks
                 */
                omp_unset_lock(&chunk_locks[myid]);
                omp_set_lock(&chunk_locks[steal_from]);

                start = chunks[steal_from].start;
                end = start + (int)ceil((chunks[steal_from].end - chunks[steal_from].start)/
                        (double)nthreads);
                chunks[steal_from].start = end;

                omp_unset_lock(&chunk_locks[steal_from]);
            } else {
                /* If there are no iterations left in all the
                 * processes it means that the thread can finish
                 */
                omp_unset_lock(&chunk_locks[myid]);
                break;
            }

            /* There is a very small chance that in the 'else if' case
             * above a process selectred to steal_from had all of its iterations
             * claimed by the time the current process tries to claim its 
             * iterations. In such a case the while loop would not break,
             * but in the chunk range start would equal the end. As there may
             * still be unclaimed iterations left in other threads, 
             * the iteration-claiming process above should repeat.
             */
            if (start == end) 
                continue;

            switch (loopid) { 
                case 1: loop1chunk(start, end); break;
                case 2: loop2chunk(start, end); break;
            } 
        }
    }
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


