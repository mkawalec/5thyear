#!/bin/bash
#$ -V
#$ -cwd

mpd &
mpirun -np $NSLOTS data/morar-edges data/edge768x768.pgm 768 768
