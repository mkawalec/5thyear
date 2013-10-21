#!/bin/bash

export OMP_NUM_THREADS=$NSLOTS

#for i in 1 2 4 8 16 32 64; do ./loops $i; done
./loops
