#!/bin/bash
#$ -V
#$ -cwd

mpd &
mpirun -np $NSLOTS data/morar-edges data/edge768x1152.pgm 768 1152