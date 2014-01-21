MF=	Makefile

CC=	mpicc
CFLAGS=

LFLAGS=

EXE=	mpiio

SRC= \
	mpiio.c \
	ioutils.c

INC= \
	ioutils.h

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .c .o

OBJ=	$(SRC:.c=.o)

.c.o:
	$(CC) $(CFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(INC) $(MF)

clean:
	rm -f $(OBJ) $(EXE) core
