include ../../config.mak

OBJ = arralloc.o write_ppm.o read_options.o

all: $(OBJ)

$(OBJ): ../../config.mak

write_ppm.o: write_ppm.c write_ppm.h

arralloc.o: arralloc.c arralloc.h

read_options.o: read_options.c read_options.h
clean:
	-rm $(OBJ)
