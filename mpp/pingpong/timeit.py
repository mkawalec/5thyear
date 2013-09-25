#!/usr/bin/env python2
from subprocess import check_output
import matplotlib.pyplot as plt

sizes = map(lambda x: pow(2, x), range(1, 13))
results = map(lambda x: check_output(["mpirun -np 2 ./pingpong 1000000 " + str(x)], shell=True),
        sizes)

plt.plot(sizes, map(lambda x: float(x), results))
plt.show()
