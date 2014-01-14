# Leave alone below here


#
# lists of sources, etc
#
SRC = mandelbrot.c

OBJ=$(SRC:.c=.o)

include ../../../config.mak

LIBDIR = ../../utils
CFLAGS := $(CFLAGS) -I$(LIBDIR)

LIBS = $(LIBDIR)/arralloc.o $(LIBDIR)/write_ppm.o $(LIBDIR)/read_options.o

all: mandelbrot

$(OBJ):	$(MAKEFILE) $(SRC) $(INC) utils

utils:
	$(MAKE) -C $(LIBDIR)

mandelbrot:	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIBS)


clean: clean-utils
	rm -f mandelbrot *~ $(OBJ) $(EXE) core

clean-utils:
	$(MAKE) -C $(LIBDIR) clean

.PHONY: utils clean-utils
