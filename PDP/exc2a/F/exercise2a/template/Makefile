# Leave alone below here


#
# lists of sources, etc
#
SRC = mandelbrot.f90

include ../../../config.mak

LIBDIR = ../../utils
FFLAGS := $(FFLAGS) -I$(LIBDIR)

LIBS = $(LIBDIR)/write_ppm.o $(LIBDIR)/read_options.o

all: mandelbrot

utils:
	$(MAKE) -C $(LIBDIR)

mandelbrot:	$(SRC) utils
	$(FC) $(FFLAGS) -o $@ $(SRC) $(LIBS)


clean: clean-utils
	rm -f mandelbrot *~ $(OBJ) $(EXE) core

clean-utils:
	$(MAKE) -C $(LIBDIR) clean

.PHONY: utils clean-utils
