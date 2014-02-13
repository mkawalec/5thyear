#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

#include "ioutils.h"

/*
 *  The global data size is NX x NY
 */

#define NX 480
#define NY 216

/*
 *  The processes are in a 2D array of dimension XPROCS x YPROCS, with
 *  a total of NPROCS processes
 */

#define NDIM 2

#define XPROCS 4
#define YPROCS 1

#define NPROCS (XPROCS*YPROCS)

/*
 *  The local data size is NXP x NYP
 */

#define NXP (NX/XPROCS)
#define NYP (NY/YPROCS)

/*
 *  The maximum length of a file name
 */

#define MAXFILENAME 64

void main(void)
{
    /*
     *  pcoords stores the grid positions of each process
     */

    int pcoords[NPROCS][NDIM];

    /*
     *  buf is the large buffer for the master to read into
     *  x contains the local data only
     */

    float buf[NX][NY];
    float x[NXP][NYP];

    int rank, size;
    int i, j;

    char filename[MAXFILENAME];

    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    /*
     *  Check we are running on the correct number of processes
     */

    if (size != NPROCS)
    {
        if (rank == 0)
        {
            printf("ERROR: compiled for %d process(es), running on %d\n",
                    NPROCS, size);
        }

        MPI_Finalize();
        exit(-1);
    }

    /*
     *  Work out the coordinates of all the processes in the grid and
     *  print them out
     */

    initpgrid(pcoords, XPROCS, YPROCS);

    if (rank == 0)
    {
        printf("Running on %d process(es) in a %d x %d grid\n",
                NPROCS, XPROCS, YPROCS);
        printf("\n");

        for (i=0; i < NPROCS; i++)
        {
            printf("Process %2d has grid coordinates (%2d, %2d)\n",
                    i, pcoords[i][0], pcoords[i][1]);
        }
        printf("\n");
    }

    /*
     *  Initialise the arrays to a grey value
     */

    initarray(buf, NX,  NY );
    initarray(x  , NXP, NYP);


    /*
     *  Read the entire array on the master process
     *  Passing "-1" as the rank argument means that the file name has no
     *  trailing "_rank" appended to it, ie we read the global file
     */

    MPI_File blah;
    createfilename(filename, "cinput", NX, NY, -1);
    //ioread (filename, buf, NX*NY);

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &blah);
    MPI_File_read_all(blah, buf, NY * NX / XPROCS, MPI_FLOAT, NULL);
    MPI_File_close(&blah);

    /*
     *  Simply copy the data from buf to x
     */

    for (i=0; i < NXP; i++)
    {
        for (j=0; j < NYP; j++)
        {
            x[i][j] = buf[i][j];
        }
    }

    /*
     *  Every process writes out its local data array x to an individually
     *  named file which has the rank appended to the file name
     */

    createfilename(filename, "coutput", NXP, NYP, rank);
    iowrite(filename, x, NXP*NYP);

    MPI_Finalize();
}
