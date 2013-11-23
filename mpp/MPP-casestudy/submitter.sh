#
/bin/bash

export OMP_NUM_THREADS=$NSLOTS
data/static_edges data/edge768x1152 768 1152
