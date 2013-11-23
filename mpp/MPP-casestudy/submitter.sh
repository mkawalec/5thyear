#
/bin/bash

export OMP_NUM_THREADS=$NSLOTS
build/edges data/edge768x1152 768 1152
